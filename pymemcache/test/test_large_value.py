import json
import threading
import time

import pytest

from pymemcache.client.large import LargeValueClient
from pymemcache.exceptions import MemcacheClientError


_EXPTIME_THRESHOLD = 30 * 24 * 60 * 60


class FakeMemcacheClient:
    """Thread-safe in-memory stand-in for a Memcached server.

    Unlike MockMemcacheClient it implements gets/cas, real Memcached expiry
    semantics and test injection hooks (``before_set`` / ``fail_on_call``),
    which the large-value protocol needs to be exercised.
    """

    def __init__(self, clock=time.time):
        self.clock = clock
        self._contents = {}
        self._next_cas = 0
        self._lock = threading.RLock()
        self.before_set = None
        self.fail_on_call = None
        self.set_calls = 0

    def _deadline(self, expire):
        if not expire:
            return None
        if expire >= _EXPTIME_THRESHOLD:
            return expire
        return int(self.clock() + expire)

    def _alive(self, entry):
        deadline = entry[0]
        return deadline is None or deadline > self.clock()

    def _mint_cas(self):
        self._next_cas += 1
        return self._next_cas

    def get(self, key, default=None):
        with self._lock:
            entry = self._contents.get(key)
            if entry is None or not self._alive(entry):
                self._contents.pop(key, None)
                return default
            return entry[1]

    def get_many(self, keys):
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result

    def gets(self, key, default=None, cas_default=None):
        with self._lock:
            entry = self._contents.get(key)
            if entry is None or not self._alive(entry):
                self._contents.pop(key, None)
                return default, cas_default
            return entry[1], entry[2]

    def set(self, key, value, expire=0, noreply=False, flags=None):
        if self.before_set is not None:
            self.before_set(key, value)
        with self._lock:
            self.set_calls += 1
            if self.fail_on_call is not None and self.set_calls == self.fail_on_call:
                return False
            self._contents[key] = (
                self._deadline(expire),
                value,
                self._mint_cas(),
            )
            return True

    def add(self, key, value, expire=0, noreply=False, flags=None):
        with self._lock:
            entry = self._contents.get(key)
            if entry is not None and self._alive(entry):
                return False
            self._contents[key] = (
                self._deadline(expire),
                value,
                self._mint_cas(),
            )
            return True

    def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
        with self._lock:
            entry = self._contents.get(key)
            if entry is None or not self._alive(entry):
                self._contents.pop(key, None)
                return None
            if entry[2] != cas:
                return False
            self._contents[key] = (
                self._deadline(expire),
                value,
                self._mint_cas(),
            )
            return True

    def delete(self, key, noreply=None):
        with self._lock:
            return self._contents.pop(key, None) is not None

    def raw(self, key):
        with self._lock:
            return self._contents.get(key)


def make_client(chunk_size=128, clock=time.time, serde=None):
    fake = FakeMemcacheClient(clock=clock)
    wrapper = LargeValueClient(
        fake, chunk_size=chunk_size, time_func=clock, serde=serde
    )
    return fake, wrapper


@pytest.mark.unit()
def test_roundtrip_large_value():
    fake, client = make_client(chunk_size=100)
    value = b"".join(bytes([i % 256]) * 137 for i in range(20))
    assert client.set_large(b"key", value) is True
    assert client.get_large(b"key") == value


@pytest.mark.unit()
def test_single_chunk_and_empty_value():
    fake, client = make_client(chunk_size=1024)
    assert client.set_large(b"small", b"abc") is True
    assert client.get_large(b"small") == b"abc"

    assert client.set_large(b"empty", b"") is True
    assert client.get_large(b"empty") == b""


@pytest.mark.unit()
def test_serde_types_roundtrip():
    fake, client = make_client(chunk_size=50)
    for value in ["a unicode value " * 20, 123456789, {"a": [1, 2, 3]}]:
        assert client.set_large(b"k", value) is True
        assert client.get_large(b"k") == value


@pytest.mark.unit()
def test_get_missing_returns_default():
    fake, client = make_client()
    assert client.get_large(b"nope", "dflt") == "dflt"


@pytest.mark.unit()
def test_manifest_records_metadata_and_uses_coded_chunk_keys():
    fake, client = make_client(chunk_size=100)
    value = bytes(range(250))
    assert client.set_large(b"key", value) is True

    raw_manifest = fake.get(client._manifest_key(b"key"))
    manifest = json.loads(raw_manifest.decode("ascii"))
    assert manifest["v"] == 1
    assert manifest["length"] == len(value)
    assert manifest["chunks"] == 3
    assert len(manifest["digest"]) == 64
    assert len(manifest["gen"]) == 32

    generation = manifest["gen"].encode("ascii")
    chunk_keys = [
        client._chunk_key(b"key", generation, i) for i in range(manifest["chunks"])
    ]
    assert all(fake.get(chunk_key) is not None for chunk_key in chunk_keys)


@pytest.mark.unit()
def test_deleted_chunk_never_returns_wrong_value():
    fake, client = make_client(chunk_size=100)
    value = b"payload-" * 100
    assert client.set_large(b"key", value) is True

    manifest = json.loads(fake.get(client._manifest_key(b"key")))
    victim = client._chunk_key(b"key", manifest["gen"].encode(), 1)
    assert fake.delete(victim) is True

    assert client.get_large(b"key", "miss") == "miss"


@pytest.mark.unit()
def test_tampered_chunk_never_returns_wrong_value():
    fake, client = make_client(chunk_size=100)
    value = b"payload-" * 100
    assert client.set_large(b"key", value) is True

    manifest = json.loads(fake.get(client._manifest_key(b"key")))
    victim = client._chunk_key(b"key", manifest["gen"].encode(), 0)
    original = fake.get(victim)
    tampered = bytes((b + 1) % 256 for b in original)
    assert tampered != original
    fake.set(victim, tampered)

    assert client.get_large(b"key", "miss") == "miss"


@pytest.mark.unit()
def test_tampered_manifest_never_returns_wrong_value():
    fake, client = make_client(chunk_size=100)
    value = b"payload-" * 100
    assert client.set_large(b"key", value) is True

    manifest_key = client._manifest_key(b"key")
    manifest = json.loads(fake.get(manifest_key))
    manifest["length"] += 1
    fake.set(manifest_key, json.dumps(manifest).encode("ascii"))

    assert client.get_large(b"key", "miss") == "miss"


@pytest.mark.unit()
def test_failed_chunk_write_keeps_existing_manifest():
    fake, client = make_client(chunk_size=100)
    first = b"first-value-" * 50
    second = b"second-value-is-different-" * 50
    assert client.set_large(b"key", first) is True
    first_manifest = fake.get(client._manifest_key(b"key"))

    # The very first chunk write of the second generation fails.
    fake.fail_on_call = fake.set_calls + 1
    assert client.set_large(b"key", second) is False

    assert fake.get(client._manifest_key(b"key")) == first_manifest
    assert client.get_large(b"key") == first


@pytest.mark.unit()
def test_mid_write_chunk_failure_keeps_existing_manifest():
    fake, client = make_client(chunk_size=100)
    first = b"first-value-" * 50
    second = b"second-value-is-different-" * 50
    assert client.set_large(b"key", first) is True
    first_manifest = fake.get(client._manifest_key(b"key"))

    # Fail the second chunk write; the first chunk is already on the server.
    fake.fail_on_call = fake.set_calls + 2
    assert client.set_large(b"key", second) is False

    assert fake.get(client._manifest_key(b"key")) == first_manifest
    assert client.get_large(b"key") == first


class _AlwaysCasConflict(FakeMemcacheClient):
    def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
        return False


@pytest.mark.unit()
def test_manifest_cas_conflict_keeps_current_version():
    fake = _AlwaysCasConflict()
    client = LargeValueClient(fake, chunk_size=100)
    first = b"first-value-" * 50
    second = b"second-value-is-different-" * 50
    assert client.set_large(b"key", first) is True
    first_manifest = fake.get(client._manifest_key(b"key"))

    # The chunks write fine, but publishing loses the CAS race.
    assert client.set_large(b"key", second) is False
    assert fake.get(client._manifest_key(b"key")) == first_manifest
    assert client.get_large(b"key") == first


@pytest.mark.unit()
def test_interleaved_writes_only_complete_generation_is_visible():
    fake = FakeMemcacheClient()
    writer_a = LargeValueClient(fake, chunk_size=50)
    writer_b = LargeValueClient(fake, chunk_size=50)

    value_a = b"A" * 230
    value_b = b"B" * 140

    parked = threading.Event()
    released = threading.Event()

    def park_mid_write(key, value):
        # Writer A is suspended immediately before its third chunk, while
        # writer B publishes a complete generation in the meantime.
        if threading.current_thread() is thread_a and key.endswith(b":00000002"):
            if not parked.is_set():
                parked.set()
                released.wait(timeout=10)

    fake.before_set = park_mid_write

    result_a = {}

    def run_a():
        result_a["ok"] = writer_a.set_large(b"key", value_a)

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    assert parked.wait(timeout=10)

    assert writer_b.set_large(b"key", value_b) is True

    released.set()
    thread_a.join(timeout=10)
    assert not thread_a.is_alive()

    # A refuses to install its manifest over B's valid generation.
    assert result_a["ok"] is False

    manifest = json.loads(fake.get(writer_b._manifest_key(b"key")))
    generation_b = manifest["gen"].encode("ascii")

    # Neither writer ever returns a stitched-together mixture.
    assert writer_a.get_large(b"key") == value_b
    assert writer_b.get_large(b"key") == value_b

    # A's orphan chunks are still on the server but the manifest does not
    # reference them; they are left to expire rather than being scanned away.
    orphan_keys = [
        key
        for key in fake._contents
        if key.startswith(b"pmlv:") and generation_b not in key
        and not key.endswith(b":m")
    ]
    assert orphan_keys
    assert writer_a.get_large(b"key") == value_b


@pytest.mark.unit()
def test_foreign_generation_chunk_is_never_stitched_in():
    fake, client = make_client(chunk_size=100)
    first = b"first-value-" * 50
    second = b"second-value-is-different-" * 50
    assert client.set_large(b"key", first) is True
    first_manifest = json.loads(fake.get(client._manifest_key(b"key")))

    # A second, complete generation exists as orphaned chunks after the
    # manifest was overwritten... simulate it by writing value two and then
    # restoring the first manifest.
    assert client.set_large(b"key", second) is True
    second_manifest = json.loads(fake.get(client._manifest_key(b"key")))
    fake.set(client._manifest_key(b"key"), json.dumps(first_manifest).encode())

    # Point the first manifest's index-0 key at the second generation's chunk.
    foreign_chunk = fake.get(
        client._chunk_key(b"key", second_manifest["gen"].encode(), 0)
    )
    victim = client._chunk_key(
        b"key", first_manifest["gen"].encode(), 0
    )
    fake.set(victim, foreign_chunk)

    # The digest cannot match: miss, not a mixed business value.
    assert client.get_large(b"key", "miss") == "miss"


@pytest.mark.unit()
def test_past_expiry_is_rejected():
    fake, client = make_client()
    with pytest.raises(MemcacheClientError):
        client.set_large(b"key", b"value", expire=int(time.time()) - 10)


@pytest.mark.unit()
def test_chunks_and_manifest_share_expiry():
    clock_value = [1_000_000_000]
    fake, client = make_client(chunk_size=100, clock=lambda: clock_value[0])
    assert client.set_large(b"key", b"x" * 450, expire=100) is True
    assert client.get_large(b"key") == b"x" * 450

    manifest_key = client._manifest_key(b"key")
    manifest = json.loads(fake.get(manifest_key))
    for index in range(manifest["chunks"]):
        chunk_key = client._chunk_key(b"key", manifest["gen"].encode(), index)
        deadline, _, _ = fake.raw(chunk_key)
        assert deadline == 1_000_000_100
    assert fake.raw(manifest_key)[0] == 1_000_000_100

    # Everything ages out at the same instant; a missing manifest or chunk
    # simply reads as a miss.
    clock_value[0] = 1_000_000_101
    assert client.get_large(b"key", "miss") == "miss"


@pytest.mark.unit()
def test_delete_large_removes_current_generation():
    fake, client = make_client(chunk_size=100)
    value = b"payload-" * 100
    assert client.set_large(b"key", value) is True
    manifest_key = client._manifest_key(b"key")
    prior_manifest = json.loads(fake.get(manifest_key))
    generation = prior_manifest["gen"].encode("ascii")
    chunk_count = prior_manifest["chunks"]
    assert client.delete_large(b"key") is True
    assert client.get_large(b"key", "miss") == "miss"

    assert fake.get(manifest_key) is None
    assert all(
        fake.get(client._chunk_key(b"key", generation, index)) is None
        for index in range(chunk_count)
    )


@pytest.mark.unit()
def test_plain_get_set_are_unchanged():
    fake, client = make_client()
    assert client.set(b"plain", b"value") is True
    assert client.get(b"plain") == b"value"
    assert client.delete(b"plain") is True
    assert client.get(b"plain", "miss") == "miss"

    # A large value does not masquerade as a plain value under the same key.
    assert client.set_large(b"large", b"z" * 500) is True
    assert client.get(b"large", "miss") == "miss"


@pytest.mark.unit()
def test_custom_serde_is_respected():
    class ReversedSerde:
        def serialize(self, key, value):
            return value[::-1], 7

        def deserialize(self, key, value, flags):
            assert flags == 7
            return value[::-1]

    fake, client = make_client(chunk_size=40, serde=ReversedSerde())
    value = b"serde-aware-value-" * 10
    assert client.set_large(b"key", value) is True
    assert client.get_large(b"key") == value
