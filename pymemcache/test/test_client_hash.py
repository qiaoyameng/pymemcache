from pymemcache.client.hash import CASToken, HashClient
from pymemcache.client.base import Client, PooledClient
from pymemcache.exceptions import (
    MemcacheClientError,
    MemcacheError,
    MemcacheUnknownError,
)
from pymemcache import pool

from .test_client import ClientTestMixin, MockSocket
import unittest
import os
import pytest
from unittest import mock
import socket


class FakeCASClient:
    def __init__(self, server, **kwargs):
        self.server = server
        self.gets_calls = []
        self.cas_calls = []
        self.closed = False

    def gets(self, key, default=None, cas_default=None, **kwargs):
        self.gets_calls.append(key)
        return b"value", b"42"

    def cas(self, key, value, cas, **kwargs):
        self.cas_calls.append((key, value, cas))
        return True

    def close(self):
        self.closed = True


class PooledHashClient(HashClient):
    client_class = FakeCASClient


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

    def test_cas_token_is_route_bound(self):
        ip = "127.0.0.1"
        server_a = (ip, 11011)
        server_b = (ip, 11012)
        server_c = (ip, 11013)
        key_a = b"key_a"
        key_b = b"key_b"

        client = PooledHashClient([], use_pooling=True)
        client.add_server(server_a)
        client.add_server(server_b)

        node_a = "127.0.0.1:11011"
        node_b = "127.0.0.1:11012"
        node_c = "127.0.0.1:11013"
        routing = {key_a: node_a, key_b: node_b}
        client.hasher.get_node = lambda key: routing[key]

        value_a, token_a = client.gets_with_token(key_a)
        value_b, token_b = client.gets_with_token(key_b)

        assert value_a == value_b == b"value"
        assert token_a.cas == token_b.cas == b"42"
        assert token_a.server_key == node_a
        assert token_b.server_key == node_b

        client.add_server(server_c)
        routing[key_a] = node_c
        routing[key_b] = node_c

        fake_a = client.clients[node_a].client_pool.free[0]
        fake_b = client.clients[node_b].client_pool.free[0]

        assert client.cas(key_a, b"raw", b"42", noreply=False) is True
        fake_c = client.clients[node_c].client_pool.free[0]
        assert fake_c.cas_calls == [(key_a, b"raw", b"42")]

        assert client.cas_with_token(key_a, b"from-a", token_a) is True
        assert client.cas(key_b, b"from-b", token_b) is True
        assert client.cas(key_b, b"from-b-kw", cas=token_b) is True

        tuple_key = (b"server_key", b"tuple_item")
        routing[b"server_key"] = node_b
        _, tuple_token = client.gets_with_token(tuple_key)
        routing[b"server_key"] = node_c
        assert client.cas_with_token(tuple_key, b"tuple", tuple_token) is True

        assert fake_a.cas_calls == [(key_a, b"from-a", b"42")]
        assert fake_b.cas_calls == [
            (key_b, b"from-b", b"42"),
            (key_b, b"from-b-kw", b"42"),
            (b"tuple_item", b"tuple", b"42"),
        ]
        assert client.clients[node_a].client_pool.used == ()

        before_wrong_key = len(fake_b.cas_calls)
        with pytest.raises(MemcacheClientError, match="different key"):
            client.cas_with_token(b"other", b"ignored", token_b)
        assert len(fake_b.cas_calls) == before_wrong_key

        with pytest.raises(TypeError):
            client.cas_with_token(key_a, b"ignored", node_a)

        forged_token = CASToken(
            token_b.client_id,
            token_b.key,
            node_a,
            token_b.generation,
            token_b.cas,
            token_b.signature,
        )
        with pytest.raises(MemcacheClientError, match="invalid"):
            client.cas_with_token(key_b, b"ignored", forged_token)
        with pytest.raises(AttributeError):
            token_b.server_key = node_a

        other_client = PooledHashClient([], use_pooling=True)
        other_client.add_server(server_a)
        _, foreign_token = other_client.gets_with_token(key_a)
        with pytest.raises(MemcacheClientError, match="different client"):
            client.cas_with_token(key_a, b"ignored", foreign_token)

        client.remove_server(server_a, permanent=True)
        with pytest.raises(MemcacheClientError, match="permanently removed"):
            client.cas_with_token(key_a, b"ignored", token_a)
        assert fake_a.closed is True
        assert fake_a.cas_calls == [(key_a, b"from-a", b"42")]

        client.add_server(server_a)
        with pytest.raises(MemcacheClientError, match="replaced"):
            client.cas_with_token(key_a, b"ignored", token_a)

        assert client.clients[node_b].client_pool.used == ()
        assert len(client.clients[node_b].client_pool.free) == 1
        assert client.clients[node_c].client_pool.used == ()
        assert len(client.clients[node_c].client_pool.free) == 1

    def test_cas_token_survives_temporary_server_recovery(self):
        ip = "127.0.0.1"
        server_a = (ip, 11021)
        server_b = (ip, 11022)
        key = b"key"
        client = PooledHashClient([], use_pooling=True, dead_timeout=60)
        client.add_server(server_a)
        client.add_server(server_b)

        node_a = "127.0.0.1:11021"
        node_b = "127.0.0.1:11022"
        client.hasher.get_node = lambda value: node_a if value == key else node_b

        _, token = client.gets_with_token(key)
        generation = token.generation
        fake_a = client.clients[node_a].client_pool.free[0]

        client.remove_server(server_a, permanent=False)
        assert node_a in client.clients
        assert node_a not in client.hasher.nodes
        with pytest.raises(MemcacheClientError, match="temporarily unavailable"):
            client.cas_with_token(key, b"too-soon", token)
        assert fake_a.cas_calls == []
        client._dead_clients[server_a] = -1
        client._last_dead_check_time = -100

        assert client.cas_with_token(key, b"recovered", token) is True
        assert token.generation == generation
        assert fake_a.cas_calls == [(key, b"recovered", b"42")]
        assert node_a in client.hasher.nodes
        assert client.clients[node_a].client_pool.used == ()

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
        assert server in client._dead_clients
        assert server not in client._failed_clients

        # Ensure that server is a string if passing port argument:
        with pytest.raises(TypeError):
            client.add_server(server, server[-1])
        with pytest.raises(TypeError):
            client.remove_server(server, server[-1])

    # TODO: Test failover logic
