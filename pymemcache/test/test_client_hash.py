from pymemcache.client.hash import CasToken, HashClient
from pymemcache.client.base import Client, PooledClient
from pymemcache.exceptions import (
    MemcacheError,
    MemcacheIllegalInputError,
    MemcacheNodeUnavailableError,
    MemcacheServerRemovedError,
    MemcacheUnknownError,
)
from pymemcache import pool

from .test_client import ClientTestMixin, MockSocket
import unittest
import os
import pytest
from unittest import mock
import socket


class TestHashClient(ClientTestMixin, unittest.TestCase):
    def make_client_pool(self, hostname, mock_socket_values, serializer=None, **kwargs):
        mock_client = Client(hostname, serializer=serializer, **kwargs)
        mock_client.sock = MockSocket(mock_socket_values)
        client = PooledClient(hostname, serializer=serializer)
        client.client_pool = pool.ObjectPool(lambda: mock_client)
        return mock_client

    def make_client(self, *mock_socket_values, **kwargs):
        current_port = 11012
        client = HashClient([], **kwargs)
        ip = "127.0.0.1"

        for vals in mock_socket_values:
            s = f"{ip}:{current_port}"
            c = self.make_client_pool((ip, current_port), vals, **kwargs)
            client.clients[s] = c
            client.hasher.add_node(s)
            current_port += 1

        return client

    def make_unix_client(self, sockets, *mock_socket_values, **kwargs):
        client = HashClient([], **kwargs)

        for socket_, vals in zip(sockets, mock_socket_values):
            c = self.make_client_pool(socket_, vals, **kwargs)
            client.clients[socket_] = c
            client.hasher.add_node(socket_)

        return client

    def test_setup_client_without_pooling(self):
        client_class = "pymemcache.client.hash.HashClient.client_class"
        with mock.patch(client_class) as internal_client:
            client = HashClient([], timeout=999, key_prefix="foo_bar_baz")
            client.add_server(("127.0.0.1", "11211"))

        assert internal_client.call_args[0][0] == ("127.0.0.1", "11211")
        kwargs = internal_client.call_args[1]
        assert kwargs["timeout"] == 999
        assert kwargs["key_prefix"] == "foo_bar_baz"

    def test_get_many_unix(self):
        pid = os.getpid()
        sockets = [
            "/tmp/pymemcache.1.%d" % pid,
            "/tmp/pymemcache.2.%d" % pid,
        ]
        client = self.make_unix_client(
            sockets,
            *[
                [
                    b"STORED\r\n",
                    b"VALUE key3 0 6\r\nvalue2\r\nEND\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VALUE key1 0 6\r\nvalue1\r\nEND\r\n",
                ],
            ],
        )

        def get_node(key):
            if key == b"key3":
                return "/tmp/pymemcache.1.%d" % pid
            else:
                return "/tmp/pymemcache.2.%d" % pid

        client.hasher.get_node = get_node

        result = client.set(b"key1", b"value1", noreply=False)
        result = client.set(b"key3", b"value2", noreply=False)
        result = client.get_many([b"key1", b"key3"])
        assert result == {b"key1": b"value1", b"key3": b"value2"}

    def test_get_many_unix_with_server_key(self):
        pid = os.getpid()
        sockets = [
            "/tmp/pymemcache.1.%d" % pid,
            "/tmp/pymemcache.2.%d" % pid,
        ]
        client = self.make_unix_client(
            sockets,
            *[
                [
                    b"STORED\r\n",
                    b"STORED\r\n",
                    b"VALUE key1 0 6\r\nvalue1\r\nVALUE key3 0 6\r\nvalue2\r\nEND\r\n",
                ],
                [],
            ],
        )

        def get_node(key):
            if key == b"server_key":
                return "/tmp/pymemcache.1.%d" % pid
            else:
                return "/tmp/pymemcache.2.%d" % pid

        client.hasher.get_node = get_node

        result = client.set((b"server_key", b"key1"), b"value1", noreply=False)
        result = client.set((b"server_key", b"key3"), b"value2", noreply=False)
        result = client.get_many([(b"server_key", b"key1"), (b"server_key", b"key3")])
        assert result == {b"key1": b"value1", b"key3": b"value2"}

    def test_get_many_all_found(self):
        client = self.make_client(
            *[
                [
                    b"STORED\r\n",
                    b"VALUE key3 0 6\r\nvalue2\r\nEND\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VALUE key1 0 6\r\nvalue1\r\nEND\r\n",
                ],
            ]
        )

        def get_node(key):
            if key == b"key3":
                return "127.0.0.1:11012"
            else:
                return "127.0.0.1:11013"

        client.hasher.get_node = get_node

        result = client.set(b"key1", b"value1", noreply=False)
        result = client.set(b"key3", b"value2", noreply=False)
        result = client.get_many([b"key1", b"key3"])
        assert result == {b"key1": b"value1", b"key3": b"value2"}

    def test_get_many_some_found(self):
        client = self.make_client(
            *[
                [
                    b"END\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VALUE key1 0 6\r\nvalue1\r\nEND\r\n",
                ],
            ]
        )

        def get_node(key):
            if key == b"key3":
                return "127.0.0.1:11012"
            else:
                return "127.0.0.1:11013"

        client.hasher.get_node = get_node
        result = client.set(b"key1", b"value1", noreply=False)
        result = client.get_many([b"key1", b"key3"])

        assert result == {b"key1": b"value1"}

    def test_get_many_bad_server_data(self):
        client = self.make_client(
            *[
                [
                    b"STORED\r\n",
                    b"VAXLUE key3 0 6\r\nvalue2\r\nEND\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VAXLUE key1 0 6\r\nvalue1\r\nEND\r\n",
                ],
            ]
        )

        def get_node(key):
            if key == b"key3":
                return "127.0.0.1:11012"
            else:
                return "127.0.0.1:11013"

        client.hasher.get_node = get_node

        with pytest.raises(MemcacheUnknownError):
            client.set(b"key1", b"value1", noreply=False)
            client.set(b"key3", b"value2", noreply=False)
            client.get_many([b"key1", b"key3"])

    def test_get_many_bad_server_data_ignore(self):
        client = self.make_client(
            *[
                [
                    b"STORED\r\n",
                    b"VAXLUE key3 0 6\r\nvalue2\r\nEND\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VAXLUE key1 0 6\r\nvalue1\r\nEND\r\n",
                ],
            ],
            ignore_exc=True,
        )

        def get_node(key):
            if key == b"key3":
                return "127.0.0.1:11012"
            else:
                return "127.0.0.1:11013"

        client.hasher.get_node = get_node

        client.set(b"key1", b"value1", noreply=False)
        client.set(b"key3", b"value2", noreply=False)
        result = client.get_many([b"key1", b"key3"])
        assert result == {}

    def test_gets_many(self):
        client = self.make_client(
            *[
                [
                    b"STORED\r\n",
                    b"VALUE key3 0 6 1\r\nvalue2\r\nEND\r\n",
                ],
                [
                    b"STORED\r\n",
                    b"VALUE key1 0 6 1\r\nvalue1\r\nEND\r\n",
                ],
            ]
        )

        def get_node(key):
            if key == b"key3":
                return "127.0.0.1:11012"
            else:
                return "127.0.0.1:11013"

        client.hasher.get_node = get_node

        assert client.set(b"key1", b"value1", noreply=False) is True
        assert client.set(b"key3", b"value2", noreply=False) is True
        result = client.gets_many([b"key1", b"key3"])
        assert result == {b"key1": (b"value1", b"1"), b"key3": (b"value2", b"1")}

    def test_cas_token_stays_bound_after_topology_changes(self):
        class FakeCasClient:
            def __init__(self, server, **kwargs):
                self.server = server
                self.calls = []

            def gets(self, key, default=None, cas_default=None):
                self.calls.append(("gets", key))
                return b"value", b"1"

            def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
                self.calls.append(("cas", key, value, cas))
                return True

        servers = [("127.0.0.1", 11012), ("127.0.0.1", 11013)]
        key = b"key"
        client = HashClient([])
        fakes = {}

        for server in servers:
            fake = FakeCasClient(server)
            server_key = client._make_client_key(server)
            fakes[server_key] = fake
            client.clients[server_key] = fake
            client.hasher.add_node(server_key)
            client._add_node_identity(server_key)

        original_server_key = client.hasher.get_node(key)
        original_client = fakes[original_server_key]
        remaining_server_key = next(
            server_key for server_key in fakes if server_key != original_server_key
        )
        original_server = next(
            server
            for server in servers
            if client._make_client_key(server) == original_server_key
        )

        value, token = client.gets_with_cas_token(key)
        assert value == b"value"
        assert isinstance(token, CasToken)
        assert token.cas == b"1"
        assert original_client.calls == [("gets", key)]

        assert client.cas_with_token(key, b"new-value", token) is True
        assert original_client.calls[-1] == ("cas", key, b"new-value", b"1")

        assert client.cas(key, b"overloaded-value", token) is True
        assert original_client.calls[-1] == (
            "cas",
            key,
            b"overloaded-value",
            b"1",
        )

        client.remove_server(original_server)
        assert client.hasher.get_node(key) == remaining_server_key

        with pytest.raises(MemcacheServerRemovedError):
            client.cas_with_token(key, b"another-value", token)
        assert fakes[remaining_server_key].calls == []

        assert client.cas(key, b"legacy-value", token.cas, noreply=False) is True
        assert fakes[remaining_server_key].calls == [
            ("cas", key, b"legacy-value", b"1")
        ]

        client.client_class = FakeCasClient
        client.add_server(*original_server)
        with pytest.raises(MemcacheServerRemovedError):
            client.cas_with_token(key, b"readded-value", token)

    def test_cas_token_rejects_other_key_and_untrusted_credentials(self):
        class FakeCasClient:
            def __init__(self, server, **kwargs):
                self.server = server
                self.calls = []

            def gets(self, key, default=None, cas_default=None):
                self.calls.append(("gets", key))
                return b"value", b"1"

            def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
                self.calls.append(("cas", key, value, cas))
                return True

        client = HashClient([])
        fake = FakeCasClient(("127.0.0.1", 11211))
        server_key = "127.0.0.1:11211"
        client.clients[server_key] = fake
        client.hasher.add_node(server_key)
        client._add_node_identity(server_key)

        _, token = client.gets_with_cas_token(b"key")
        with pytest.raises(MemcacheIllegalInputError):
            client.cas_with_token(b"other-key", b"value", token)
        with pytest.raises(MemcacheIllegalInputError):
            client.cas_with_token(b"key", b"value", "127.0.0.1:11211")
        with pytest.raises(TypeError):
            CasToken()
        assert fake.calls == [("gets", b"key")]

        class MissingFake:
            server = ("127.0.0.1", 11211)

            def gets(self, key, default=None, cas_default=None):
                return default, cas_default

        client.clients[server_key] = MissingFake()
        sentinel = object()
        assert client.gets_with_cas_token(
            b"missing", default=b"miss", cas_default=sentinel
        ) == (b"miss", sentinel)

    def test_temporarily_dead_node_keeps_cas_token_identity(self):
        class FakeCasClient:
            def __init__(self, server, **kwargs):
                self.server = server
                self.calls = []

            def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
                self.calls.append(("cas", key, value, cas))
                return True

        servers = [("127.0.0.1", 11012), ("127.0.0.1", 11013)]
        client = HashClient([], dead_timeout=3600)
        fakes = {}

        for server in servers:
            fake = FakeCasClient(server)
            server_key = client._make_client_key(server)
            fakes[server_key] = fake
            client.clients[server_key] = fake
            client.hasher.add_node(server_key)
            client._add_node_identity(server_key)

        token = CasToken._create(client._node_ids["127.0.0.1:11012"], b"key", b"1")
        client._cas_tokens.add(token)
        client._mark_server_dead(("127.0.0.1", 11012))

        assert client.hasher.get_node(b"key") == "127.0.0.1:11013"
        with pytest.raises(MemcacheNodeUnavailableError):
            client.cas_with_token(b"key", b"value", token)
        assert fakes["127.0.0.1:11013"].calls == []

        client._last_dead_check_time = 0
        client._dead_clients[("127.0.0.1", 11012)] = 0
        assert client.cas_with_token(b"key", b"recovered", token) is True
        assert fakes["127.0.0.1:11012"].calls == [
            ("cas", b"key", b"recovered", b"1")
        ]

    def test_manual_replacement_of_dead_node_invalidates_cas_token(self):
        class FakeCasClient:
            def __init__(self, server, **kwargs):
                self.server = server

        client = HashClient([], dead_timeout=3600)
        server = ("127.0.0.1", 11012)
        client.clients["127.0.0.1:11012"] = FakeCasClient(server)
        client.hasher.add_node("127.0.0.1:11012")
        client._add_node_identity("127.0.0.1:11012")
        token = CasToken._create(client._node_ids["127.0.0.1:11012"], b"key", b"1")
        client._cas_tokens.add(token)

        client._mark_server_dead(server)
        client.client_class = FakeCasClient
        client.add_server(*server)

        with pytest.raises(MemcacheServerRemovedError):
            client.cas_with_token(b"key", b"value", token)

    def test_actual_failure_marks_node_temporarily_dead_with_same_identity(self):
        class FakeCasClient:
            def __init__(self, server):
                self.server = server
                self.get = mock.Mock(side_effect=socket.timeout())

        client = HashClient([], retry_attempts=0, dead_timeout=3600)
        fake = FakeCasClient(("127.0.0.1", 11211))
        client.clients["127.0.0.1:11211"] = fake
        client.hasher.add_node("127.0.0.1:11211")
        client._add_node_identity("127.0.0.1:11211")

        with pytest.raises(socket.timeout):
            client.get(b"key")

        assert ("127.0.0.1", 11211) in client._dead_clients
        assert client._node_ids["127.0.0.1:11211"] in client._dead_node_ids

    def test_cas_token_returns_pooled_client_resources(self):
        class FakeCasClient:
            def __init__(self, server, **kwargs):
                self.server = server
                self.calls = []

            def gets(self, key, default=None, cas_default=None):
                self.calls.append(("gets", key))
                return b"value", b"1"

            def cas(self, key, value, cas, expire=0, noreply=False, flags=None):
                self.calls.append(("cas", key, value, cas))
                return True

            def close(self):
                pass

        server = ("127.0.0.1", 11211)
        pooled_client = PooledClient(server)
        pooled_client.client_pool = pool.ObjectPool(lambda: FakeCasClient(server))
        client = HashClient([])
        client.clients["127.0.0.1:11211"] = pooled_client
        client.hasher.add_node("127.0.0.1:11211")
        client._add_node_identity("127.0.0.1:11211")

        _, token = client.gets_with_cas_token(key=b"key")
        assert pooled_client.client_pool.used == ()
        raw_client = pooled_client.client_pool.free[0]

        assert client.cas_with_token(b"key", b"value", token) is True
        assert pooled_client.client_pool.used == ()
        assert pooled_client.client_pool.free == (raw_client,)
        assert raw_client.calls == [
            ("gets", b"key"),
            ("cas", b"key", b"value", b"1"),
        ]

    def test_touch_not_found(self):
        client = self.make_client([b"NOT_FOUND\r\n"])
        result = client.touch(b"key", noreply=False)
        assert result is False

    def test_touch_no_expiry_found(self):
        client = self.make_client([b"TOUCHED\r\n"])
        result = client.touch(b"key", noreply=False)
        assert result is True

    def test_touch_with_expiry_found(self):
        client = self.make_client([b"TOUCHED\r\n"])
        result = client.touch(b"key", 1, noreply=False)
        assert result is True

    def test_close(self):
        client = self.make_client([])
        assert all(c.sock is not None for c in client.clients.values())
        result = client.close()
        assert result is None
        assert all(c.sock is None for c in client.clients.values())

    def test_quit(self):
        client = self.make_client([])
        assert all(c.sock is not None for c in client.clients.values())
        result = client.quit()
        assert result is None
        assert all(c.sock is None for c in client.clients.values())

    def test_no_servers_left(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=True, timeout=1, connect_timeout=1
        )

        hashed_client = client._get_client("foo")
        assert hashed_client == (None, "foo")

    def test_no_servers_left_raise_exception(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=False, timeout=1, connect_timeout=1
        )

        with pytest.raises(MemcacheError) as e:
            client._get_client("foo")

        assert str(e.value) == "All servers seem to be down right now"

    def test_unavailable_servers_zero_retry_raise_exception(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [("example.com", 11211)],
            use_pooling=True,
            ignore_exc=False,
            retry_attempts=0,
            timeout=1,
            connect_timeout=1,
        )

        with pytest.raises(socket.error):
            client.get("foo")

    def test_no_servers_left_with_commands_return_default_value(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=True, timeout=1, connect_timeout=1
        )

        result = client.get("foo")
        assert result is None
        result = client.get("foo", default="default")
        assert result == "default"
        result = client.set("foo", "bar")
        assert result is False

    def test_no_servers_left_return_positional_default(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=True, timeout=1, connect_timeout=1
        )

        # Ensure compatibility with clients that pass the default as a
        # positional argument
        result = client.get("foo", "default")
        assert result == "default"

    def test_no_servers_left_with_set_many(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=True, timeout=1, connect_timeout=1
        )

        result = client.set_many({"foo": "bar"})
        assert result == ["foo"]

    def test_no_servers_left_with_get_many(self):
        from pymemcache.client.hash import HashClient

        client = HashClient(
            [], use_pooling=True, ignore_exc=True, timeout=1, connect_timeout=1
        )

        result = client.get_many(["foo", "bar"])
        assert result == {}

    def test_ignore_exec_set_many(self):
        values = {"key1": "value1", "key2": "value2", "key3": "value3"}

        with pytest.raises(MemcacheUnknownError):
            client = self.make_client(
                *[
                    [b"STORED\r\n", b"UNKNOWN\r\n", b"STORED\r\n"],
                    [b"STORED\r\n", b"UNKNOWN\r\n", b"STORED\r\n"],
                ]
            )
            client.set_many(values, noreply=False)

        client = self.make_client(
            *[
                [b"STORED\r\n", b"UNKNOWN\r\n", b"STORED\r\n"],
            ],
            ignore_exc=True,
        )
        result = client.set_many(values, noreply=False)

        assert len(result) == 0

    def test_noreply_set_many(self):
        values = {"key1": "value1", "key2": "value2", "key3": "value3"}

        client = self.make_client(
            *[
                [b"STORED\r\n", b"NOT_STORED\r\n", b"STORED\r\n"],
            ]
        )
        result = client.set_many(values, noreply=False)
        assert len(result) == 1

        client = self.make_client(
            *[
                [b"STORED\r\n", b"NOT_STORED\r\n", b"STORED\r\n"],
            ]
        )
        result = client.set_many(values, noreply=True)
        assert result == []

    def test_noreply_flush(self):
        client = self.make_client()
        client.flush_all(noreply=True)

    def test_stats_many_server(self):
        client = self.make_client(
            *[
                [b"STAT fake_stats 1\r\n", b"END\r\n"],
                [b"STAT fake_stats 2\r\n", b"END\r\n"],
            ]
        )
        result = client.stats()
        assert result == [
            ("127.0.0.1:11012", {b"fake_stats": 1}),
            ("127.0.0.1:11013", {b"fake_stats": 2}),
        ]

    def test_set_many_unix(self):
        values = {"key1": "value1", "key2": "value2", "key3": "value3"}

        pid = os.getpid()
        sockets = ["/tmp/pymemcache.%d" % pid]
        client = self.make_unix_client(
            sockets,
            *[
                [b"STORED\r\n", b"NOT_STORED\r\n", b"STORED\r\n"],
            ],
        )

        result = client.set_many(values, noreply=False)
        assert len(result) == 1

    def test_server_encoding_pooled(self):
        """
        test passed encoding from hash client to pooled clients
        """
        encoding = "utf8"
        from pymemcache.client.hash import HashClient

        hash_client = HashClient(
            [("example.com", 11211)], use_pooling=True, encoding=encoding
        )

        for client in hash_client.clients.values():
            assert client.encoding == encoding

    def test_server_encoding_client(self):
        """
        test passed encoding from hash client to clients
        """
        encoding = "utf8"
        from pymemcache.client.hash import HashClient

        hash_client = HashClient([("example.com", 11211)], encoding=encoding)

        for client in hash_client.clients.values():
            assert client.encoding == encoding

    @mock.patch("pymemcache.client.hash.HashClient.client_class")
    def test_dead_server_comes_back(self, client_patch):
        client = HashClient([], dead_timeout=0, retry_attempts=0)
        client.add_server(("127.0.0.1", 11211))

        test_client = client_patch.return_value
        test_client.server = ("127.0.0.1", 11211)

        test_client.get.side_effect = socket.timeout()
        with pytest.raises(socket.timeout):
            client.get(b"key", noreply=False)
        # Client gets removed because of socket timeout
        assert ("127.0.0.1", 11211) in client._dead_clients

        test_client.get.side_effect = lambda *_, **_kw: "Some value"
        # Client should be retried and brought back
        assert client.get(b"key") == "Some value"
        assert ("127.0.0.1", 11211) not in client._dead_clients

    @mock.patch("pymemcache.client.hash.HashClient.client_class")
    def test_failed_is_retried(self, client_patch):
        client = HashClient([], retry_attempts=1, retry_timeout=0)
        client.add_server(("127.0.0.1", 11211))

        assert client_patch.call_count == 1

        test_client = client_patch.return_value
        test_client.server = ("127.0.0.1", 11211)

        test_client.get.side_effect = socket.timeout()
        with pytest.raises(socket.timeout):
            client.get(b"key", noreply=False)

        test_client.get.side_effect = lambda *_, **_kw: "Some value"
        assert client.get(b"key") == "Some value"

        assert client_patch.call_count == 1

    def test_custom_client(self):
        class MyClient(Client):
            pass

        client = HashClient([])
        client.client_class = MyClient
        client.add_server(("host", 11211))
        assert isinstance(client.clients["host:11211"], MyClient)

    def test_custom_client_with_pooling(self):
        class MyClient(Client):
            pass

        client = HashClient([], use_pooling=True)
        client.client_class = MyClient
        client.add_server(("host", 11211))
        assert isinstance(client.clients["host:11211"], PooledClient)

        pool = client.clients["host:11211"].client_pool
        with pool.get_and_release(destroy_on_fail=True) as c:
            assert isinstance(c, MyClient)

    def test_mixed_inet_and_unix_sockets(self):
        expected = {
            f"/tmp/pymemcache.{os.getpid()}",
            ("127.0.0.1", 11211),
            ("::1", 11211),
        }
        client = HashClient(
            [
                f"/tmp/pymemcache.{os.getpid()}",
                "127.0.0.1",
                "127.0.0.1:11211",
                "[::1]",
                "[::1]:11211",
                ("127.0.0.1", 11211),
                ("::1", 11211),
            ]
        )
        assert expected == {c.server for c in client.clients.values()}

    def test_legacy_add_remove_server_signature(self):
        server = ("127.0.0.1", 11211)
        client = HashClient([])
        assert client.clients == {}
        client.add_server(*server)  # Unpack (host, port) tuple.
        assert ("%s:%s" % server) in client.clients
        client._mark_failed_server(server)
        assert server in client._failed_clients
        client.remove_server(*server)  # Unpack (host, port) tuple.
        assert server not in client._dead_clients
        assert server not in client._failed_clients
        assert "127.0.0.1:11211" not in client.hasher.nodes

        # Ensure that server is a string if passing port argument:
        with pytest.raises(TypeError):
            client.add_server(server, server[-1])
        with pytest.raises(TypeError):
            client.remove_server(server, server[-1])

    # TODO: Test failover logic
