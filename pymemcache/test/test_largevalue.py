"""Acceptance tests for pymemcache.client.largevalue.LargeValueClient.

These tests use an in-memory fake memcached service (no sockets) and
focus on the core guarantee: a reader never receives a wrongly
reassembled business value, even when writes interleave, chunks are
deleted, or chunk contents are tampered with.
"""

import json

import pytest

from pymemcache.client.largevalue import LargeValueClient


class Clock:
    def __init__(self, start=1_700_000_000.0):
        self._now = start

    def time(self):
        return self._now

    def advance(self, seconds):
        self._now += seconds


class FakeMemcacheClient:
    """In-memory fake supporting the subset LargeValueClient relies on."""

    def __init__(self, clock=None):
        self.clock = clock or Clock()
        # key -> [expire_at_or_0, value, cas_id]
        self._contents = {}
        self._cas_counter = 0
        # test hooks
        self.fail_next_sets = 0
        self.on_gets = None

    # -- helpers ----------------------------------------------------------

    def _expired(self, entry):
        expire_at = entry[0]
        return expire_at != 0 and expire_at <= self.clock.time()

    def _lookup(self, key):
        entry = self._contents.get(key)
        if entry is None:
            return None
        if self._expired(entry):
            del self._contents[key]
            return None
        return entry

    def _store(self, key, value, expire):
        expire_at = self.clock.time() + expire if expire else 0
        self._cas_counter += 1
        self._contents[key] = [expire_at, value, self._cas_counter]
        return True

    def chunk_keys(self):
        return [
            k
            for k in self._contents
            if isinstance(k, bytes) and k.startswith(b"__lvchunk__")
        ]

    # -- memcache subset --------------------------------------------------

    def get(self, key, default=None):
        entry = self._lookup(key)
        return default if entry is None else entry[1]

    def gets(self, key):
        entry = self._lookup(key)
        snapshot = (None, None) if entry is None else (entry[1], entry[2])
        # The hook runs after the snapshot, modeling a concurrent writer
        # that acts between our gets and our subsequent add/cas.
        if self.on_gets is not None:
            self.on_gets(key)
        return snapshot

    def set(self, key, value, expire=0, noreply=False, flags=None):
        if self.fail_next_sets > 0:
            self.fail_next_sets -= 1
            return False
        return self._store(key, value, expire)

    def add(self, key, value, expire=0, noreply=False, flags=None):
        if self._lookup(key) is not None:
            return False
        return self._store(key, value, expire)

    def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
        entry = self._lookup(key)
        if entry is None:
            return None
        if entry[2] != cas:
            return False
        return self._store(key, value, expire)

    def delete(self, key, noreply=False):
        return self._contents.pop(key, None) is not None


def make_pair(chunk_size=64):
    clock = Clock()
    fake = FakeMemcacheClient(clock)
    wrapper = LargeValueClient(fake, chunk_size=chunk_size, now=clock.time)
    return fake, wrapper


def test_roundtrip_multi_chunk():
    fake, wrapper = make_pair(chunk_size=16)
    value = {"payload": "x" * 500, "n": 42}
    assert wrapper.set_large("k", value) is True
    assert len(fake.chunk_keys()) > 1
    assert wrapper.get_large("k") == value


def test_roundtrip_empty_and_small_values():
    fake, wrapper = make_pair()
    assert wrapper.set_large("empty", "") is True
    assert wrapper.get_large("empty") == ""
    assert wrapper.set_large("small", b"tiny") is True
    assert wrapper.get_large("small") == b"tiny"


def test_get_missing_key_returns_default():
    fake, wrapper = make_pair()
    sentinel = object()
    assert wrapper.get_large("nope", default=sentinel) is sentinel


def test_get_non_manifest_value_returns_default():
    fake, wrapper = make_pair()
    fake.set("plain", b"not a manifest")
    assert wrapper.get_large("plain") is None


def test_interleaved_writes_keep_valid_manifest():
    """Two writers race; the CAS loser must not clobber the winner and the
    reader must only ever see one complete generation."""
    fake, wrapper = make_pair(chunk_size=16)
    value_a = "A" * 200
    value_b = "B" * 200

    def interleave(key):
        # Writer B sneaks in a full write between writer A's chunk writes
        # and writer A's manifest publish. One-shot to avoid recursion.
        fake.on_gets = None
        assert wrapper.set_large("k", value_b) is True

    fake.on_gets = interleave
    # Writer A loses the manifest race: its publish must fail and must not
    # replace B's manifest.
    assert wrapper.set_large("k", value_a) is False

    # The reader gets B's complete value, never a mix of A and B chunks.
    assert wrapper.get_large("k") == value_b

    # Both generations' chunks exist (expiry reclaims the orphans), but
    # the published manifest points only at the winner's generation.
    manifest = json.loads(fake.get("k"))
    gens = {k.split(b".")[1].decode() for k in fake.chunk_keys()}
    assert len(gens) == 2
    assert manifest["gen"] in gens


def test_chunk_write_failure_keeps_previous_manifest():
    fake, wrapper = make_pair(chunk_size=16)
    old_value = "old" * 100
    assert wrapper.set_large("k", old_value) is True

    # The next write dies partway through its chunks.
    fake.fail_next_sets = 1
    assert wrapper.set_large("k", "new" * 100) is False

    # The previously published manifest is untouched and still readable.
    assert wrapper.get_large("k") == old_value


def test_deleted_chunk_never_returns_partial_value():
    fake, wrapper = make_pair(chunk_size=8)
    value = "z" * 100
    assert wrapper.set_large("k", value) is True

    doomed = fake.chunk_keys()[len(fake.chunk_keys()) // 2]
    fake.delete(doomed)

    assert wrapper.get_large("k") is None
    assert wrapper.get_large("k", default="fallback") == "fallback"


def test_tampered_chunk_never_returns_corrupt_value():
    fake, wrapper = make_pair(chunk_size=8)
    value = "q" * 100
    assert wrapper.set_large("k", value) is True

    victim = fake.chunk_keys()[0]
    entry = fake._contents[victim]
    corrupted = b"X" + entry[1][1:]
    entry[1] = corrupted

    assert wrapper.get_large("k") is None


def test_tampered_manifest_digest_rejected():
    fake, wrapper = make_pair(chunk_size=8)
    assert wrapper.set_large("k", "m" * 50) is True

    manifest = json.loads(fake.get("k"))
    manifest["len"] += 1
    fake.set("k", json.dumps(manifest).encode("ascii"))

    assert wrapper.get_large("k") is None


def test_insufficient_remaining_ttl_aborts_before_publish():
    fake, wrapper = make_pair(chunk_size=16)

    def burn_ttl(key):
        # By the time the manifest would be published, the deadline shared
        # by the chunks has passed.
        fake.on_gets = None
        fake.clock.advance(11)

    fake.on_gets = burn_ttl
    assert wrapper.set_large("k", "v" * 100, expire=10) is False

    # No manifest was published, so readers cannot be led to chunks that
    # are already gone.
    assert fake.get("k") is None
    assert wrapper.get_large("k") is None
    assert fake.chunk_keys() == [] or all(
        fake._lookup(k) is None for k in fake.chunk_keys()
    )


def test_shared_ttl_applies_to_chunks_and_manifest():
    fake, wrapper = make_pair(chunk_size=16)
    assert wrapper.set_large("k", "v" * 100, expire=30) is True

    manifest_expire = fake._contents["k"][0]
    for chunk_key in fake.chunk_keys():
        assert fake._contents[chunk_key][0] == manifest_expire

    # Everything expires together; after the deadline nothing is served.
    fake.clock.advance(31)
    assert wrapper.get_large("k") is None


def test_delete_large_removes_manifest_only():
    fake, wrapper = make_pair(chunk_size=16)
    assert wrapper.set_large("k", "v" * 100) is True
    assert wrapper.delete_large("k") is True
    assert wrapper.get_large("k") is None
    # Orphaned chunks remain for memcached expiry to reclaim; no scanning.
    assert fake.chunk_keys()


def test_serializer_must_return_bytes():
    fake, _ = make_pair()
    wrapper = LargeValueClient(fake, serializer=lambda k, v: "not-bytes")
    with pytest.raises(TypeError):
        wrapper.set_large("k", object())
