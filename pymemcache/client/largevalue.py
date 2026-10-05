# Copyright 2012 Pinterest.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A wrapper for storing values larger than a single memcached item.

Serialized values are split into chunks namespaced by a per-write
generation id, and a small manifest (total length, chunk count and a
SHA-256 digest of the whole payload) is published last, atomically, via
add/cas. Readers only accept a manifest whose chunks are all present and
whose digest verifies, so a corrupted or partially overwritten value is
never returned.

Losing writers (chunk write failure or manifest cas conflict) leave the
previously published manifest untouched; their orphaned chunks are
reclaimed by memcached expiry, no scanning required. Chunks and the
manifest share a single expiry deadline: if the remaining TTL runs out
before the manifest can be published, the write is aborted instead of
publishing a manifest that is doomed to miss chunks.

The wrapped client's regular ``get``/``set`` behavior is unchanged; large
values go through :meth:`LargeValueClient.set_large`,
:meth:`LargeValueClient.get_large` and
:meth:`LargeValueClient.delete_large`.
"""

import hashlib
import json
import pickle
import secrets
import time
from typing import Any, Callable, Optional

from pymemcache.client.base import Key

#: Default chunk payload size, kept below memcached's 1MiB item limit
#: (slab overhead and key/metadata consume part of the item).
DEFAULT_CHUNK_SIZE = 900_000

#: Memcached treats expire values up to this many seconds as relative;
#: larger values are absolute unix timestamps.
_RELATIVE_EXPIRE_LIMIT = 60 * 60 * 24 * 30

_MANIFEST_VERSION = 1
_CHUNK_KEY_PREFIX = "__lvchunk__"


def _default_serializer(key: Key, value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _default_deserializer(key: Key, payload: bytes) -> Any:
    return pickle.loads(payload)


class LargeValueClient:
    """Stores oversized values as generation-tagged chunks plus a manifest.

    Args:
      client: a pymemcache-compatible client. It must support ``get``,
        ``set``, ``add``, ``gets``, ``cas`` and ``delete``.
      chunk_size: maximum number of payload bytes per chunk.
      serializer: callable ``(key, value) -> bytes`` used to serialize the
        business value before chunking. Defaults to pickle.
      deserializer: callable ``(key, payload) -> value`` used to rebuild the
        business value from the verified payload. Defaults to pickle.
      now: callable returning the current unix timestamp; injectable for
        testing.
    """

    def __init__(
        self,
        client: Any,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        serializer: Optional[Callable[[Key, Any], bytes]] = None,
        deserializer: Optional[Callable[[Key, bytes], Any]] = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self._client = client
        self._chunk_size = chunk_size
        self._serializer = serializer or _default_serializer
        self._deserializer = deserializer or _default_deserializer
        self._now = now

    # -- key helpers ------------------------------------------------------

    @staticmethod
    def _key_bytes(key: Key) -> bytes:
        if isinstance(key, str):
            return key.encode("utf8")
        return bytes(key)

    def _chunk_key(self, key: Key, generation: str, index: int) -> bytes:
        key_hash = hashlib.sha256(self._key_bytes(key)).hexdigest()[:32]
        return f"{_CHUNK_KEY_PREFIX}{key_hash}.{generation}.{index}".encode("ascii")

    # -- expiry helpers ---------------------------------------------------

    def _deadline(self, expire: int) -> Optional[float]:
        """Translate a memcache expire argument into an absolute deadline."""
        if not expire:
            return None
        if expire <= _RELATIVE_EXPIRE_LIMIT:
            return self._now() + expire
        return float(expire)

    def _remaining_ttl(self, deadline: Optional[float]) -> Optional[int]:
        """Whole seconds left before the deadline.

        Returns ``None`` when there is no deadline. A non-positive result
        means the remaining TTL is insufficient to safely publish.
        """
        if deadline is None:
            return None
        return int(deadline - self._now())

    # -- manifest helpers -------------------------------------------------

    def _build_manifest(
        self, generation: str, length: int, chunks: int, digest: str
    ) -> bytes:
        manifest = {
            "v": _MANIFEST_VERSION,
            "gen": generation,
            "len": length,
            "chunks": chunks,
            "sha256": digest,
        }
        return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )

    @staticmethod
    def _parse_manifest(raw: Any) -> Optional[dict]:
        if not isinstance(raw, (bytes, str)):
            return None
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return None
        if not isinstance(data, dict) or data.get("v") != _MANIFEST_VERSION:
            return None
        generation = data.get("gen")
        length = data.get("len")
        chunks = data.get("chunks")
        digest = data.get("sha256")
        if not (
            isinstance(generation, str)
            and isinstance(length, int)
            and isinstance(chunks, int)
            and isinstance(digest, str)
        ):
            return None
        if length < 0 or chunks <= 0:
            return None
        return data

    # -- public API -------------------------------------------------------

    def set_large(self, key: Key, value: Any, expire: int = 0) -> bool:
        """Serialize ``value`` and store it as chunks plus a manifest.

        The manifest is published only after every chunk of this write's
        generation has been stored, and only if the shared expiry deadline
        still has whole seconds remaining. A chunk write failure, an
        expired deadline, or a manifest add/cas conflict leaves any
        previously published manifest (and its chunks) as the valid
        version; this write's chunks are left to be reclaimed by expiry.

        Returns True if the new manifest was published, False otherwise.
        """
        payload = self._serializer(key, value)
        if not isinstance(payload, bytes):
            raise TypeError("serializer must return bytes")

        generation = secrets.token_hex(8)
        digest = hashlib.sha256(payload).hexdigest()
        chunks = [
            payload[offset : offset + self._chunk_size]
            for offset in range(0, len(payload), self._chunk_size)
        ] or [b""]

        deadline = self._deadline(expire)

        for index, chunk in enumerate(chunks):
            remaining = self._remaining_ttl(deadline)
            if remaining is not None and remaining <= 0:
                return False
            stored = self._client.set(
                self._chunk_key(key, generation, index),
                chunk,
                expire=remaining or 0,
                noreply=False,
            )
            if not stored:
                return False

        manifest = self._build_manifest(generation, len(payload), len(chunks), digest)
        return self._publish_manifest(key, manifest, deadline)

    def get_large(self, key: Key, default: Any = None) -> Any:
        """Fetch and verify a value stored with :meth:`set_large`.

        Returns ``default`` unless a complete, digest-verified generation
        is found: a missing manifest, an unparseable manifest, a missing
        chunk, or a length/digest mismatch all yield ``default`` rather
        than a partially or wrongly reassembled value.
        """
        raw = self._client.get(key)
        if raw is None:
            return default
        manifest = self._parse_manifest(raw)
        if manifest is None:
            return default

        generation = manifest["gen"]
        parts = []
        for index in range(manifest["chunks"]):
            chunk = self._client.get(self._chunk_key(key, generation, index))
            if chunk is None:
                return default
            parts.append(chunk)

        payload = b"".join(parts)
        if len(payload) != manifest["len"]:
            return default
        if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
            return default

        return self._deserializer(key, payload)

    def delete_large(self, key: Key) -> bool:
        """Delete the manifest for ``key``.

        Orphaned chunks are not scanned for; they are reclaimed by
        memcached expiry.
        """
        return bool(self._client.delete(key, noreply=False))

    # -- internal ---------------------------------------------------------

    def _publish_manifest(
        self, key: Key, manifest: bytes, deadline: Optional[float]
    ) -> bool:
        """Atomically replace the manifest, keeping the current version on
        conflict."""
        _, cas = self._client.gets(key)
        remaining = self._remaining_ttl(deadline)
        if remaining is not None and remaining <= 0:
            # Not enough TTL left for readers to fetch every chunk; do not
            # publish a manifest that is guaranteed to be missing chunks.
            return False
        expire = remaining or 0
        if cas is None:
            # No manifest published yet; add loses atomically to a
            # concurrent writer, in which case their version stands.
            return bool(self._client.add(key, manifest, expire=expire, noreply=False))
        # EXISTS (False) or a vanished key (None) both mean this write did
        # not win; the currently stored manifest remains the valid one.
        return bool(self._client.cas(key, manifest, cas, expire=expire, noreply=False))
