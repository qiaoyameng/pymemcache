"""
Explicit command pipelining.

A :class:`Pipeline` batches independent get/set/delete operations so that
they are written to a single connection in one round trip.  Commands that
don't need an acknowledgement may be queued as *silent* (``noreply``).

The end of the batch is delimited with a ``version`` command, which the
memcached text protocol always answers with exactly one line.  Reading
that line proves that every reply that precedes it has been fully
received, so the connection is only handed back to the pool once the
response boundary is known.

Every item keeps the position it was queued at.  A result slot can hold:

* the actual value returned by the server (``bytes``/deserialized value,
  ``True``/``False``, ...),
* :attr:`PipelineResult.SILENT` for a noreply write that was flushed to
  the socket without an acknowledgement,
* :attr:`PipelineResult.UNKNOWN` for an item whose fate could not be
  determined because the connection closed before its reply arrived.

Those three outcomes are intentionally never collapsed into a single
boolean.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from pymemcache.client.base import Client


class PipelineResult(Enum):
    """Sentinel values that may appear in a pipeline result list.

    ``SILENT`` marks a noreply command: it was written successfully but
    the server never acknowledges it.  ``UNKNOWN`` marks a command that
    was written but whose reply never arrived because the connection was
    lost, so the operation may or may not have happened.
    """

    SILENT = "silent"
    UNKNOWN = "unknown"


class Pipeline:
    """A batch of independent cache operations sent on one connection.

    Instances are normally created through :meth:`Client.pipeline` (also
    available on :class:`~pymemcache.client.base.PooledClient`).  Commands
    are queued with :meth:`get`, :meth:`set` and :meth:`delete`; all keys
    and serialized values are validated before anything is written, then
    :meth:`execute` flushes the batch and reads the replies.

    A pipeline can also be used as a context manager that executes on
    exit, storing the results on ``pipeline.results``::

        with client.pipeline() as pipe:
            pipe.set(b"a", b"1", silent=True)
            pipe.get(b"a")

    Args:
      client: the client that owns the connection the batch runs on.
    """

    def __init__(self, client: "Client"):
        self._client = client
        self._ops: list[tuple] = []
        self.results: Optional[list[Any]] = None

    @property
    def queued(self) -> int:
        """Number of commands currently waiting to be executed."""
        return len(self._ops)

    def get(self, key) -> "Pipeline":
        """Queue a ``get`` command for *key* and return self."""
        self._ops.append(("get", key))
        return self

    def set(
        self,
        key,
        value,
        expire: int = 0,
        silent: Optional[bool] = None,
        flags: Optional[int] = None,
        noreply: Optional[bool] = None,
    ) -> "Pipeline":
        """Queue a ``set`` command and return self.

        Args:
          key: str or bytes, see :class:`~pymemcache.client.base.Client`.
          value: value to store; it is serialized at execute time.
          expire: optional int, item lifetime in seconds (0 = no expiry).
          silent: optional bool, send ``noreply`` and record the result as
            :attr:`PipelineResult.SILENT`.  Defaults to the client's
            ``default_noreply`` setting.
          flags: optional int, server-specific flags; overrides serializer
            flags.
          noreply: deprecated alias for *silent*.
        """
        if silent is not None and noreply is not None and silent != noreply:
            raise ValueError("Pass only one of 'silent' and 'noreply'.")
        if silent is None:
            silent = noreply
        self._ops.append(("set", key, value, expire, silent, flags))
        return self

    def delete(
        self, key, silent: Optional[bool] = None, noreply: Optional[bool] = None
    ) -> "Pipeline":
        """Queue a ``delete`` command and return self.

        Args:
          key: str or bytes, see :class:`~pymemcache.client.base.Client`.
          silent: optional bool, send ``noreply`` and record the result as
            :attr:`PipelineResult.SILENT`.  Defaults to the client's
            ``default_noreply`` setting.
          noreply: deprecated alias for *silent*.
        """
        if silent is not None and noreply is not None and silent != noreply:
            raise ValueError("Pass only one of 'silent' and 'noreply'.")
        if silent is None:
            silent = noreply
        self._ops.append(("delete", key, silent))
        return self

    def execute(self) -> list[Any]:
        """Validate, send and read the whole batch.

        All queued keys and serialized values are checked before a single
        byte is written, so an illegal argument never leaves earlier
        writes on the wire.  Results are returned in submission order.

        Returns:
          A list aligned with the queued commands; see
          :class:`PipelineResult` for the sentinel values.
        """
        results = self._client._pipeline_execute(self._ops)
        self._ops = []
        self.results = results
        return results

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.execute()
        return False
