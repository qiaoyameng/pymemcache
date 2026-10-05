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

import collections
import functools
from unittest import mock

import pytest

from pymemcache.client.base import Client, Pipeline, PipelineResult, PooledClient
from pymemcache.exceptions import (
    MemcacheIllegalInputError,
    MemcacheServerError,
)


def split_packets(data, pattern=(1, 4, 2, 7, 3, 5)):
    """Split one response blob into arbitrary recv()-sized pieces."""
    chunks = []
    offset = 0
    index = 0
    while offset < len(data):
        size = pattern[index % len(pattern)]
        chunks.append(data[offset : offset + size])
        offset += size
        index += 1
    return chunks


class PipelineMockSocket:
    """Socket mock that hands back arbitrary chunks and simulates a peer
    disconnect (EOF) as soon as the scripted chunks are exhausted."""

    def __init__(self, recv_chunks):
        self.recv_bufs = collections.deque(recv_chunks)
        self.send_bufs = []
        self.closed = False
        self.connections = []

    def sendall(self, value):
        self.send_bufs.append(value)

    def recv(self, size):
        if not self.recv_bufs:
            # The server dropped the connection mid-response.
            return b""
        return self.recv_bufs.popleft()

    def close(self):
        self.closed = True

    def settimeout(self, timeout):
        pass

    def connect(self, server):
        self.connections.append(server)

    def setsockopt(self, *args):
        pass


@pytest.mark.unit()
class TestPipeline:
    def make_client(self, recv_chunks, **kwargs):
        client = Client("localhost", **kwargs)
        sock = PipelineMockSocket(recv_chunks)
        client._connect = mock.Mock(
            side_effect=functools.partial(setattr, client, "sock", sock)
        )
        return client, sock

    def make_pooled(self, recv_chunks, **kwargs):
        sock = PipelineMockSocket(recv_chunks)
        client = Client("localhost", **kwargs)
        client._connect = mock.Mock(
            side_effect=functools.partial(setattr, client, "sock", sock)
        )
        pooled = PooledClient("localhost", **kwargs)
        pooled.client_pool._obj_creator = mock.Mock(return_value=client)
        return pooled, client, sock

    def test_get_set_delete_batched_with_barrier(self):
        responses = (
            b"VALUE k1 0 2\r\nv1\r\nEND\r\n"
            b"STORED\r\n"
            b"DELETED\r\n"
            b"END\r\n"
            b"VERSION 1.6.0\r\n"
        )
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline()
            .get(b"k1")
            .set(b"k2", b"v2")
            .delete(b"k3")
            .get(b"missing", default=b"d")
            .execute()
        )

        assert results == [b"v1", True, True, b"d"]

        # One write for the whole batch, with the barrier appended exactly
        # once at the very end.
        assert len(sock.send_bufs) == 1
        wire = sock.send_bufs[0]
        assert wire.count(b"version\r\n") == 1
        assert wire.endswith(b"version\r\n")
        assert wire.startswith(b"get k1\r\n")
        assert b"set k2 0 0 2\r\nv2\r\n" in wire
        assert b"delete k3\r\n" in wire
        # A successful barrier means the connection stays usable.
        assert not sock.closed

    def test_noreply_commands_report_silent_not_true(self):
        responses = b"VERSION 1.6.0\r\n"
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline()
            .set(b"k1", b"v1", noreply=True)
            .delete(b"k2", noreply=True)
            .execute()
        )

        assert results == [PipelineResult.SILENT, PipelineResult.SILENT]
        assert results[0] is not True
        assert results[0] is not False
        assert results[0] is not PipelineResult.UNKNOWN
        wire = sock.send_bufs[0]
        assert b"set k1 0 0 2 noreply\r\nv1\r\n" in wire
        assert b"delete k2 noreply\r\n" in wire

    def test_explicit_failures_stay_false(self):
        responses = b"NOT_STORED\r\nNOT_FOUND\r\nVERSION 1.6.0\r\n"
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline().set(b"k1", b"v1").delete(b"k2").execute()
        )

        assert results == [False, False]
        assert results[0] is False
        assert results[1] is False
        assert PipelineResult.UNKNOWN not in results

    def test_disconnect_before_barrier_keeps_known_and_unknown_apart(self):
        # Replies for the get and the replied set are fully available, but the
        # server closes before sending the barrier; no exception is raised.
        responses = b"VALUE k1 0 2\r\nv1\r\nEND\r\nSTORED\r\n"
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline()
            .get(b"k1")
            .delete(b"k3", noreply=True)
            .set(b"k2", b"v2")
            .set(b"k4", b"v4", noreply=True)
            .execute()
        )

        assert results[0] == b"v1"
        # The delete was already proven processed when the later STORED reply
        # arrived, so it is SILENT rather than unknown.
        assert results[1] is PipelineResult.SILENT
        assert results[2] is True
        # Nothing proves whether the trailing noreply set ran.
        assert results[3] is PipelineResult.UNKNOWN
        assert sock.closed

    def test_disconnect_after_write_all_unknown(self):
        # Peer closes immediately after the write, before any reply.
        client, sock = self.make_client([])

        results = (
            client.pipeline()
            .get(b"k1")
            .set(b"k2", b"v2", noreply=True)
            .delete(b"k3")
            .execute()
        )

        assert results == [
            PipelineResult.UNKNOWN,
            PipelineResult.UNKNOWN,
            PipelineResult.UNKNOWN,
        ]
        assert sock.closed

    def test_disconnect_mid_response_marks_remaining_unknown(self):
        # The first reply is cut off in the middle of its value.
        responses = b"VALUE k1 0 5\r\nv1"
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline()
            .get(b"k1")
            .set(b"k2", b"v2")
            .delete(b"k3", noreply=True)
            .execute()
        )

        assert results == [
            PipelineResult.UNKNOWN,
            PipelineResult.UNKNOWN,
            PipelineResult.UNKNOWN,
        ]
        assert sock.closed

    def test_validation_happens_before_anything_is_sent(self):
        client, sock = self.make_client([])

        with pytest.raises(MemcacheIllegalInputError):
            (
                client.pipeline()
                .set(b"good", b"data", noreply=True)
                .get(b"also-good")
                .get(b"bad key")
                .execute()
            )

        # No connection, no write: the two earlier writes were not sent.
        client._connect.assert_not_called()
        assert sock.send_bufs == []

    def test_bad_expire_validated_before_send(self):
        client, sock = self.make_client([])

        with pytest.raises(MemcacheIllegalInputError):
            client.pipeline().set(b"k", b"v", expire="soon").execute()

        client._connect.assert_not_called()
        assert sock.send_bufs == []

    def test_bad_value_validated_before_send(self):
        client, sock = self.make_client([])

        with pytest.raises(MemcacheIllegalInputError):
            client.pipeline().set(b"k", "\u0fff").execute()

        assert sock.send_bufs == []

    def test_pipeline_reuse_starts_fresh_batch(self):
        client, sock = self.make_client(
            split_packets(b"STORED\r\nVERSION 1.0\r\n")
            + split_packets(b"DELETED\r\nVERSION 1.0\r\n")
        )

        pipeline = client.pipeline()
        first = pipeline.set(b"k", b"v").execute()
        second = pipeline.delete(b"k").execute()

        assert first == [True]
        assert second == [True]
        assert len(sock.send_bufs) == 2

    def test_result_positions_follow_submission_order(self):
        responses = (
            b"END\r\n"
            b"VALUE b 0 1\r\nx\r\nEND\r\n"
            b"STORED\r\n"
            b"VERSION 1.0\r\n"
        )
        client, sock = self.make_client(split_packets(responses))

        results = (
            client.pipeline()
            .get(b"a")
            .get(b"b")
            .set(b"c", b"y")
            .execute()
        )

        assert results[0] is None
        assert results[1] == b"x"
        assert results[2] is True

    def test_server_error_raises_and_closes(self):
        client, sock = self.make_client(
            split_packets(b"SERVER_ERROR boom\r\n")
        )

        with pytest.raises(MemcacheServerError):
            client.pipeline().get(b"k").execute()

        assert sock.closed

    def test_returns_pipeline_instance_for_chaining(self):
        pipeline = Client("localhost").pipeline()
        assert isinstance(pipeline, Pipeline)
        assert pipeline.get(b"k").set(b"k2", b"v").delete(b"k3") is pipeline

    def test_pooled_pipeline_exclusively_borrows_one_connection(self):
        responses = (
            b"VALUE k1 0 2\r\nv1\r\nEND\r\n"
            b"STORED\r\n"
            b"VERSION 1.6.0\r\n"
        )
        pooled, client, sock = self.make_pooled(split_packets(responses))

        results = pooled.pipeline().get(b"k1").set(b"k2", b"v2").execute()

        assert results == [b"v1", True]
        assert len(sock.send_bufs) == 1
        # Barrier confirmed: connection is back in the pool, still open.
        assert not sock.closed
        assert pooled.client_pool.used == ()
        assert client in pooled.client_pool.free

    def test_pooled_pipeline_destroys_connection_on_disconnect(self):
        responses = b"VALUE k1 0 2\r\nv1\r\nEND\r\n"
        pooled, client, sock = self.make_pooled(split_packets(responses))

        results = pooled.pipeline().get(b"k1").set(b"k2", b"v2").execute()

        assert results == [b"v1", PipelineResult.UNKNOWN]
        # Read boundary never confirmed: connection must not return to pool.
        assert sock.closed
        assert pooled.client_pool.free == ()
        assert pooled.client_pool.used == ()

    def test_pooled_validation_failure_keeps_connection_unused(self):
        pooled, client, sock = self.make_pooled([])

        with pytest.raises(MemcacheIllegalInputError):
            pooled.pipeline().set(b"k", b"v").get(b"bad key").execute()

        # Nothing was sent; the untouched borrowed connection is returned.
        assert sock.send_bufs == []
        assert not sock.closed
        assert client in pooled.client_pool.free

    def test_pooled_pipeline_retries_with_fresh_connection(self):
        first_sock = PipelineMockSocket([])
        first_client = Client("localhost")
        first_client._connect = mock.Mock(
            side_effect=functools.partial(setattr, first_client, "sock", first_sock)
        )

        second_sock = PipelineMockSocket(
            split_packets(b"STORED\r\nVERSION 1.0\r\n")
        )
        second_client = Client("localhost")
        second_client._connect = mock.Mock(
            side_effect=functools.partial(
                setattr, second_client, "sock", second_sock
            )
        )

        pooled = PooledClient("localhost")
        pooled.client_pool._obj_creator = mock.Mock(
            side_effect=[first_client, second_client]
        )

        first = pooled.pipeline().set(b"k", b"v").execute()
        second = pooled.pipeline().set(b"k", b"v").execute()

        assert first == [PipelineResult.UNKNOWN]
        assert second == [True]
        assert first_sock.closed
        assert not second_sock.closed
        assert second_client in pooled.client_pool.free
