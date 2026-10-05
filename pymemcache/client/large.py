"""Large-value access wrapper.

Memcached imposes a per-item size limit. The classes in this module work
around that limit by transparently splitting a serialized value into a number
of smaller chunks and publishing a *manifest* that describes how to
reassemble and verify them.

The manifest is the only piece that makes a generation of chunks visible to
readers. Reads only succeed when a complete, unmodified generation of chunks
is found and the reassembled value matches the digest stored in the manifest.
Publishing a manifest is a CAS operation: when another writer has published a
generation first, the current writer keeps the existing manifest intact
instead of replacing it with its own (possibly half-expired) generation.

Chunks from superseded generations are deliberately not deleted; they simply
age out through the normal Memcached expiry mechanism, so no scan of the
server is required.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from typing import Any
from typing import Optional

from pymemcache.exceptions import MemcacheClientError
from pymemcache.serde import pickle_serde


logger = logging.getLogger(__name__)

#: Version of the on-the-wire manifest format.
MANIFEST_VERSION = 1

#: Marker prepended to every key owned by this wrapper.
PREFIX = b"pmlv:"

#: Key of the manifest, relative to the logical key.
MANIFEST_SUFFIX = b":m"

#: Default maximum size of an individual chunk. 1 MiB fits comfortably under
#: Memcached's default 1 MiB per-item limit once protocol overhead is added.
DEFAULT_CHUNK_SIZE = 1024 * 1024 - 2048

#: Memcached treats relative exptime values of 30 days or more as absolute
#: Unix timestamps.
_EXPTIME_THRESHOLD = 30 * 24 * 60 * 60


def _as_bytes(key: Any) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("ascii")
    raise TypeError("keys must be bytes or str, got %r" % type(key))


def _key_hash(key: bytes) -> bytes:
    """A short, collision-resistant encoding of the logical key."""
    return hashlib.sha256(key).hexdigest().encode("ascii")[:32]


class LargeValueClient:
    """Wrap a Memcached client so values larger than the per-item limit can be
    stored as a manifest plus a set of chunks.

    Ordinary :meth:`get` and :meth:`set` calls are forwarded unchanged to the
    wrapped client. Use :meth:`get_large`, :meth:`set_large` and
    :meth:`delete_large` for the chunked representation.

    Args:
      client: any object implementing the usual pymemcache client surface
        (``get``, ``set``, ``add``, ``cas``, ``gets``, ``get_many`` and
        ``delete``).
      serde: optional serializer/deserializer object (same interface as the
        ``serde`` argument of :class:`pymemcache.client.base.Client`).
        Defaults to :data:`pymemcache.serde.pickle_serde`, which round-trips
        bytes, strings, integers and arbitrary picklable objects.
      chunk_size: maximum serialized size of an individual chunk.
      time_func: optional callable returning the current Unix timestamp;
        primarily useful for testing.
    """

    def __init__(
        self,
        client: Any,
        serde: Any = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        time_func: Any = time.time,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.client = client
        self.serde = serde or pickle_serde
        self.chunk_size = chunk_size
        self._time = time_func

    def _manifest_key(self, key: bytes) -> bytes:
        return PREFIX + _key_hash(key) + MANIFEST_SUFFIX

    def _chunk_key(self, key: bytes, generation: bytes, index: int) -> bytes:
        return (
            PREFIX
            + _key_hash(key)
            + b":"
            + generation
            + b":"
            + (b"%08x" % index)
        )

    def _absolute_expire(self, expire: int) -> Optional[int]:
        """Return an absolute Unix expiry timestamp, or None for no expiry.

        Memcached interprets relative values of 30 days or more as absolute
        timestamps, so they are passed through unchanged.
        """
        if not expire:
            return None
        if expire >= _EXPTIME_THRESHOLD:
            return expire
        return int(self._time() + expire)

    # Plain accessors: deliberately identical to the wrapped client.

    def get(self, key: Any, default: Any = None) -> Any:
        return self.client.get(key, default)

    def set(
        self,
        key: Any,
        value: Any,
        expire: int = 0,
        noreply: Optional[bool] = None,
        flags: Optional[int] = None,
    ) -> Any:
        return self.client.set(key, value, expire=expire, noreply=noreply, flags=flags)

    def delete(self, key: Any, noreply: Optional[bool] = None) -> Any:
        return self.client.delete(key, noreply=noreply)

    # Chunked accessors.

    def set_large(self, key: Any, value: Any, expire: int = 0) -> bool:
        """Store ``value`` as a generation of chunks plus a manifest.

        All chunks and the manifest use the same absolute expiry. Returns
        True if the manifest was published and False if the chunks could not
        all be written or another writer won the manifest publication.

        Raises:
          MemcacheClientError: if the supplied expiry is already in the past.
        """
        raw_key = _as_bytes(key)
        data, data_flags = self.serde.serialize(raw_key, value)
        if isinstance(data, str):
            data = data.encode("utf-8")
        if not isinstance(data, bytes):
            raise TypeError("serializer must return bytes, got %r" % type(data))

        absolute_expire = self._absolute_expire(expire)
        if absolute_expire is not None and absolute_expire <= self._time():
            # Publishing here would create a generation that is doomed to
            # lose chunks before any reader can use it.
            raise _expired_error(absolute_expire)

        chunks = [
            data[offset : offset + self.chunk_size]
            for offset in range(0, len(data), self.chunk_size)
        ]
        if not chunks:
            chunks = [b""]

        generation = secrets.token_hex(16).encode("ascii")

        # The expire value handed to the server is the same absolute instant
        # for every chunk and for the manifest.
        server_expire = absolute_expire if absolute_expire is not None else 0

        # Observe the current manifest before writing anything. Publication is
        # conditional on this exact state still holding afterwards, so a
        # concurrent writer that commits in between wins and keeps its
        # generation intact.
        _observed_value, observed_cas = self.client.gets(
            self._manifest_key(raw_key)
        )
        chunk_keys = [
            self._chunk_key(raw_key, generation, index)
            for index in range(len(chunks))
        ]

        for chunk_key, chunk in zip(chunk_keys, chunks):
            stored = self.client.set(
                chunk_key, chunk, expire=server_expire, noreply=False
            )
            if not stored:
                logger.warning(
                    "Failed to write chunk %s; manifest for %r left untouched",
                    chunk_key,
                    key,
                )
                return False

        # Re-read every chunk before publishing: this catches stores that
        # reported success but lost (or altered) the bytes, and guarantees we
        # never publish a manifest for a generation that cannot be reassembled.
        stored_chunks = self.client.get_many(chunk_keys)
        if [stored_chunks.get(chunk_key) for chunk_key in chunk_keys] != chunks:
            logger.warning(
                "Chunks for generation %s of %r failed verification; "
                "manifest left untouched",
                generation,
                key,
            )
            return False

        digest = hashlib.sha256(data).hexdigest()
        manifest = {
            "v": MANIFEST_VERSION,
            "gen": generation.decode("ascii"),
            "flags": data_flags,
            "length": len(data),
            "chunks": len(chunks),
            "digest": digest,
        }
        manifest_bytes = json.dumps(
            manifest, separators=(",", ":"), sort_keys=True
        ).encode("ascii")

        return self._publish_manifest(
            raw_key, manifest_bytes, server_expire, observed_cas
        )

    def _publish_manifest(
        self,
        raw_key: bytes,
        manifest_bytes: bytes,
        server_expire: int,
        observed_cas: Any,
    ) -> bool:
        """Commit the manifest using the state captured before chunk writes.

        Publication succeeds only when the manifest is still exactly what the
        writer observed: ``add`` when there was no manifest, ``cas``
        otherwise. Any other outcome means a concurrent writer committed a
        generation in the meantime, so the currently valid version is kept
        and our orphan chunks are left to expire.
        """
        manifest_key = self._manifest_key(raw_key)

        if observed_cas is None:
            committed = self.client.add(
                manifest_key, manifest_bytes, expire=server_expire, noreply=False
            )
        else:
            committed = self.client.cas(
                manifest_key,
                manifest_bytes,
                observed_cas,
                expire=server_expire,
                noreply=False,
            )

        if committed is True:
            return True

        logger.warning(
            "Manifest CAS conflict for %r; keeping the currently valid version",
            raw_key,
        )
        return False

    def _read_manifest(self, raw_key: bytes) -> Optional[dict]:
        raw = self.client.get(self._manifest_key(raw_key))
        if not isinstance(raw, bytes):
            return None
        try:
            manifest = json.loads(raw.decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("Unreadable manifest for %r", raw_key)
            return None
        if not isinstance(manifest, dict):
            return None
        if manifest.get("v") != MANIFEST_VERSION:
            return None
        generation = manifest.get("gen")
        digest = manifest.get("digest")
        if not isinstance(generation, str) or not isinstance(digest, str):
            return None
        if not isinstance(manifest.get("chunks"), int) or manifest["chunks"] <= 0:
            return None
        if not isinstance(manifest.get("length"), int) or manifest["length"] < 0:
            return None
        if not isinstance(manifest.get("flags"), int):
            return None
        return manifest

    def get_large(self, key: Any, default: Any = None) -> Any:
        """Reassemble the value published by :meth:`set_large`.

        Returns ``default`` unless exactly one complete generation of chunks
        is present and the reassembled bytes match the manifest's digest. A
        missing chunk, a chunk belonging to another generation or writer, or
        any corrupted byte therefore always degrades to a miss, never to a
        wrong business value.
        """
        raw_key = _as_bytes(key)
        manifest = self._read_manifest(raw_key)
        if manifest is None:
            return default

        generation = manifest["gen"].encode("ascii")
        count = manifest["chunks"]
        chunk_keys = [
            self._chunk_key(raw_key, generation, index) for index in range(count)
        ]

        found = self.client.get_many(chunk_keys)
        pieces = []
        for chunk_key in chunk_keys:
            piece = found.get(chunk_key)
            if not isinstance(piece, bytes):
                logger.warning(
                    "Missing chunk %s while reading %r", chunk_key, raw_key
                )
                return default
            pieces.append(piece)

        data = b"".join(pieces)
        if len(data) != manifest["length"]:
            logger.warning("Length mismatch reading %r", raw_key)
            return default
        if hashlib.sha256(data).hexdigest() != manifest["digest"]:
            logger.warning(
                "Digest mismatch reading %r; refusing to return corrupt value",
                raw_key,
            )
            return default

        try:
            return self.serde.deserialize(raw_key, data, manifest["flags"])
        except Exception:
            logger.warning("Deserializer failed for %r", raw_key, exc_info=True)
            return default

    def delete_large(self, key: Any) -> bool:
        """Delete the manifest and the chunks of the current generation.

        Chunks of older generations are left behind and expire on their own.
        Returns True if a manifest was removed.
        """
        raw_key = _as_bytes(key)
        manifest_key = self._manifest_key(raw_key)
        manifest = self._read_manifest(raw_key)
        if manifest is None:
            self.client.delete(manifest_key, noreply=False)
            return False

        self.client.delete(manifest_key, noreply=False)

        generation = manifest["gen"].encode("ascii")
        for index in range(manifest["chunks"]):
            self.client.delete(
                self._chunk_key(raw_key, generation, index), noreply=False
            )
        return True


def _expired_error(expire: int) -> MemcacheClientError:
    return MemcacheClientError(
        "Cannot publish large value with expiry already in the past: %d" % expire
    )
