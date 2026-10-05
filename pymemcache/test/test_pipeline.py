"""Tests for explicit command pipelining."""

import socket
from unittest import mock

import pytest

from pymemcache import pool
from pymemcache.client.base import Client, PooledClient
from pymemcache.exceptions import (
    MemcacheIllegalInputError,
    MemcacheServerError,
    MemcacheUnknownError,
)
from pymemcache.pipeline import Pipeline, PipelineResult
from pymemcache.serde import python_memcache_serializer, python_memcache_deserializer

from pymemcache.test.test_client import MockSocket


@pytest.mark.unit()
class TestPipeline:
    def make_client(self, recv_bufs, **kwargs):
        client = Client("localhost", **kwargs)
        sock = MockSocket(list(recv_bufs))
        client._connect = mock.Mock(
            side_effect=lambda: setattr(client, "sock", sock)
        )
        return client, sock

    def test_basic_results_in_order(self):
        client, sock = self.make_client(
            [
                b"VALUE a 0 1\r\n1\r\nEND\r\n",
                b"END\r\n",
                b"STORED\r\n",
                b"NOT_STORED\r\n",
                b"DELETED\r\n",
                b"NOT_FOUND\r\n",
                b"VERSION 1.6.0\r\n",
            ],
            default_noreply=False,
        )

        results = (
            client.pipeline()
            .get(b"a")
            .get(b"missing")
            .set(b"s1", b"v")
            .set(b"s2", b"v")
            .delete(b"d1")
            .delete(b"d2")
            .execute()
        )

        assert results == [b"1", None, True, False, True, False]
        assert results[2] is True
        assert results[3] is False

        sent = b"".join(sock.send_bufs)
        assert sent == (
            b"get a\r\n"
            b"get missing\r\n"
            b"set s1 0 0 1\r\nv\r\n"
            b"set s2 0 0 1\r\nv\r\n"
            b"delete d1\r\n"
            b"delete d2\r\n"
            b"version\r\n"
        )

    def test_replies_split_across_arbitrary_packets(self):
        # Responses arrive byte-by-byte-ish and glued together in odd
        # shapes; the buffered reader must still frame every reply.
        stream = (
            b"VALUE a 0 3\r\nabc\r\nEND\r\n"
            b"STORED\r\n"
            b"DELETED\r\n"
            b"VERSION 1.6.0\r\n"
        )
        chunks = [stream[i : i + 3] for i in range(0, len(stream), 3)]
        assert len(chunks) > 1

        client, _ = self.make_client(chunks, default_noreply=False)
        results = client.pipeline().get(b"a").set(b"s", b"v").delete(b"d").execute()
        assert results == [b"abc", True, True]

    def test_silent_writes_are_marked_silent_not_false(self):
        client, sock = self.make_client([b"VERSION 1.6.0\r\n"])

        results = (
            client.pipeline()
            .set(b"s", b"v", silent=True)
            .delete(b"d", silent=True)
            .execute()
        )

        assert results == [PipelineResult.SILENT, PipelineResult.SILENT]
        assert all(r is not False and r is not None for r in results)
        sent = b"".join(sock.send_bufs)
        assert b"set s 0 0 1 noreply\r\nv\r\n" in sent
        assert b"delete d noreply\r\n" in sent

    def test_silent_alias_noreply(self):
        client, _ = self.make_client([b"VERSION 1.6.0\r\n"])
        pipe = client.pipeline()
        pipe.set(b"a", b"b", noreply=True).delete(b"d", noreply=True)
        assert pipe.execute() == [PipelineResult.SILENT, PipelineResult.SILENT]

    def test_conflicting_silent_and_noreply(self):
        client, _ = self.make_client([])
        with pytest.raises(ValueError):
            client.pipeline().set(b"a", b"b", silent=True, noreply=False)
        with pytest.raises(ValueError):
            client.pipeline().delete(b"a", silent=True, noreply=False)

    def test_default_noreply_sets_silent(self):
        client, sock = self.make_client([b"VERSION 1.6.0\r\n"], default_noreply=True)
        results = client.pipeline().set(b"a", b"b").delete(b"d").execute()
        assert results == [PipelineResult.SILENT, PipelineResult.SILENT]
        assert b"noreply" in b"".join(sock.send_bufs)

    def test_disconnect_before_barrier_keeps_known_and_marks_unknown(self):
        # Write completed; some replies arrive, then the server closes the
        # connection before the barrier reply is sent.
        client, sock = self.make_client(
            [
                b"VALUE a 0 3\r\nabc\r\nEND\r\n",
                b"STORED\r\n",
                b"",
            ],
            default_noreply=False,
        )

        results = (
            client.pipeline()
            .get(b"a")
            .set(b"s", b"v")
            .delete(b"d")
            .set(b"silent", b"v", silent=True)
            .execute()
        )

        assert results[0] == b"abc"
        assert results[1] is True
        assert results[2] is PipelineResult.UNKNOWN
        assert results[3] is PipelineResult.SILENT
        assert sock.closed
        # The broken connection is dropped by the client.
        assert client.sock is None

    def test_disconnect_inside_value_body_marks_rest_unknown(self):
        client, _ = self.make_client(
            [b"VALUE a 0 5\r\nab", b""],
            default_noreply=False,
        )
        results = client.pipeline().get(b"a").set(b"s", b"v").execute()
        assert results == [PipelineResult.UNKNOWN, PipelineResult.UNKNOWN]
        assert client.sock is None

    def test_disconnect_at_barrier_replies_remain_confirmed(self):
        client, _ = self.make_client(
            [b"STORED\r\n", b"DELETED\r\n", b""],
            default_noreply=False,
        )
        results = client.pipeline().set(b"s", b"v").delete(b"d").execute()
        assert results == [True, True]
        assert client.sock is None

    def test_illegal_key_aborts_before_writing_anything(self):
        client, sock = self.make_client([], default_noreply=False)
        pipe = client.pipeline()
        pipe.set(b"fine", b"v")
        pipe.get(b"bad key")

        with pytest.raises(MemcacheIllegalInputError):
            pipe.execute()

        assert client.sock is None
        assert sock.send_bufs == []

    def test_illegal_value_aborts_before_writing_anything(self):
        client, sock = self.make_client([], default_noreply=False)
        pipe = client.pipeline()
        pipe.set(b"first", b"v")
        pipe.set(b"second", "\u00e9")

        with pytest.raises(MemcacheIllegalInputError):
            pipe.execute()

        assert client.sock is None
        assert sock.send_bufs == []

    def test_bad_expire_aborts_before_writing_anything(self):
        client, sock = self.make_client([], default_noreply=False)
        pipe = client.pipeline().set(b"a", b"v", expire="soon")
        with pytest.raises(MemcacheIllegalInputError):
            pipe.execute()
        assert client.sock is None
        assert sock.send_bufs == []

    def test_empty_pipeline_does_not_connect(self):
        client, sock = self.make_client([])
        assert client.pipeline().execute() == []
        assert client.sock is None
        assert sock.send_bufs == []

    def test_send_failure_marks_nothing_confirmed(self):
        client, sock = self.make_client([], default_noreply=False)
        sock.sendall = mock.Mock(side_effect=OSError("broken"))
        with pytest.raises(OSError):
            client.pipeline().set(b"a", b"b").delete(b"d").execute()
        assert client.sock is None

    def test_server_error_raises_and_closes(self):
        client, _ = self.make_client(
            [b"SERVER_ERROR boom\r\n"],
            default_noreply=False,
        )
        with pytest.raises(MemcacheServerError):
            client.pipeline().set(b"a", b"b").execute()
        assert client.sock is None

    def test_unknown_reply_line_raises(self):
        client, _ = self.make_client(
            [b"WAT\r\n", b"VERSION 1.6.0\r\n"],
            default_noreply=False,
        )
        with pytest.raises(MemcacheUnknownError):
            client.pipeline().set(b"a", b"b").execute()
        assert client.sock is None

    def test_bad_barrier_reply_raises(self):
        client, _ = self.make_client(
            [b"STORED\r\n", b"WAT\r\n"],
            default_noreply=False,
        )
        with pytest.raises(MemcacheUnknownError):
            client.pipeline().set(b"a", b"b").execute()

    def test_serde_round_trip(self):
        client, _ = self.make_client(
            [
                b"VALUE a 3 2\r\n42\r\nEND\r\n",
                b"STORED\r\n",
                b"VERSION 1.6.0\r\n",
            ],
            default_noreply=False,
            serializer=python_memcache_serializer,
            deserializer=python_memcache_deserializer,
        )
        results = client.pipeline().get(b"a").set(b"s", 42).execute()
        assert results[0] == 42
        sent = b"".join(client.sock.send_bufs)
        assert b"set s 2 0 2\r\n42\r\n" in sent

    def test_key_prefix_applied(self):
        client, sock = self.make_client(
            [b"VALUE pfx:a 0 1\r\nx\r\nEND\r\n", b"VERSION 1.6.0\r\n"],
            key_prefix=b"pfx:",
            default_noreply=False,
        )
        results = client.pipeline().get(b"a").execute()
        assert results == [b"x"]
        assert b"get pfx:a\r\n" in b"".join(sock.send_bufs)

    def test_context_manager_executes_on_exit(self):
        client, _ = self.make_client(
            [b"END\r\n", b"VERSION 1.6.0\r\n"],
            default_noreply=False,
        )
        pipe = client.pipeline()
        with pipe:
            pipe.get(b"a")
        assert pipe.results == [None]

    def test_pipeline_can_be_reused_after_execute(self):
        client, _ = self.make_client(
            [
                b"STORED\r\n",
                b"VERSION 1.6.0\r\n",
                b"STORED\r\n",
                b"VERSION 1.6.0\r\n",
            ],
            default_noreply=False,
        )
        pipe = client.pipeline()
        assert pipe.execute() == []
        assert pipe.queued == 0
        pipe.set(b"a", b"b")
        assert pipe.execute() == [True]
        pipe.set(b"c", b"d")
        assert pipe.execute() == [True]


@pytest.mark.unit()
class TestPooledPipeline:
    def make_pooled_client(self, recv_bufs, **kwargs):
        inner = Client("localhost", **kwargs)
        inner.sock = MockSocket(list(recv_bufs))
        pooled = PooledClient("localhost", **kwargs)
        pooled.client_pool = pool.ObjectPool(lambda: inner)
        return pooled, inner

    def test_connection_held_during_batch_and_released_after(self):
        pooled, inner = self.make_pooled_client(
            [b"STORED\r\n", b"VERSION 1.6.0\r\n"],
            default_noreply=False,
        )
        results = pooled.pipeline().set(b"a", b"b").execute()
        assert results == [True]
        assert inner.sock is not None
        assert len(pooled.client_pool.free) == 1
        assert len(pooled.client_pool.used) == 0

    def test_broken_connection_is_destroyed_not_returned_to_pool(self):
        pooled, inner = self.make_pooled_client(
            [b"STORED\r\n", b""],
            default_noreply=False,
        )
        results = pooled.pipeline().set(b"a", b"b").delete(b"d").execute()
        assert results == [True, PipelineResult.UNKNOWN]
        assert inner.sock is None
        assert len(pooled.client_pool.free) == 0
        assert len(pooled.client_pool.used) == 0

    def test_connection_exclusively_held_when_needed(self):
        pooled, inner = self.make_pooled_client([], default_noreply=False)
        # Empty batches never check out a connection.
        assert pooled.pipeline().execute() == []
        assert len(pooled.client_pool.used) == 0

    def test_pipeline_object_type(self):
        pooled, _ = self.make_pooled_client([])
        assert isinstance(pooled.pipeline(), Pipeline)
