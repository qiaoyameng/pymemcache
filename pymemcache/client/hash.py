import collections
import socket
import time
import logging
import weakref

from pymemcache.client.base import (
    Client,
    PooledClient,
    check_key_helper,
    normalize_server_spec,
)
from pymemcache.client.rendezvous import RendezvousHash
from pymemcache.exceptions import (
    MemcacheError,
    MemcacheIllegalInputError,
    MemcacheNodeUnavailableError,
    MemcacheServerRemovedError,
)

logger = logging.getLogger(__name__)


class CasToken:
    """An opaque CAS credential bound to a key and a particular node."""

    __slots__ = ("_node_id", "_key", "_cas", "__weakref__")

    def __init__(self):
        raise TypeError("CasToken objects must be created by a HashClient")

    @classmethod
    def _create(cls, node_id, key, cas):
        token = object.__new__(cls)
        token._node_id = node_id
        token._key = key
        token._cas = cas
        return token

    @property
    def cas(self):
        return self._cas

    def __repr__(self):
        return "CasToken(***)"


CasCredential = CasToken


class HashClient:
    """
    A client for communicating with a cluster of memcached servers
    """

    #: :class:`Client` class used to create new clients
    client_class = Client

    def __init__(
        self,
        servers,
        hasher=RendezvousHash,
        serde=None,
        serializer=None,
        deserializer=None,
        connect_timeout=None,
        timeout=None,
        no_delay=False,
        socket_module=socket,
        socket_keepalive=None,
        key_prefix=b"",
        max_pool_size=None,
        pool_idle_timeout=0,
        lock_generator=None,
        retry_attempts=2,
        retry_timeout=1,
        dead_timeout=60,
        use_pooling=False,
        ignore_exc=False,
        allow_unicode_keys=False,
        default_noreply=True,
        encoding="ascii",
        tls_context=None,
    ):
        """
        Constructor.

        Args:
          servers: list() of tuple(hostname, port) or string containing a UNIX
                   socket path.
          hasher: optional class three functions ``get_node``, ``add_node``,
                  and ``remove_node``
                  defaults to Rendezvous (HRW) hash.

          use_pooling: use py:class:`.PooledClient` as the default underlying
                       class. ``max_pool_size`` and ``lock_generator`` can
                       be used with this. default: False

          retry_attempts: Amount of times a client should be tried before it
                          is marked dead and removed from the pool.
          retry_timeout (float): Time in seconds that should pass between retry
                                 attempts.
          dead_timeout (float): Time in seconds before attempting to add a node
                                back in the pool.
          encoding: optional str, controls data encoding (defaults to 'ascii').

        Further arguments are interpreted as for :py:class:`.Client`
        constructor.
        """
        self.clients = {}
        self.retry_attempts = retry_attempts
        self.retry_timeout = retry_timeout
        self.dead_timeout = dead_timeout
        self.use_pooling = use_pooling
        self.key_prefix = key_prefix
        self.ignore_exc = ignore_exc
        self.allow_unicode_keys = allow_unicode_keys
        self._failed_clients = {}
        self._dead_clients = {}
        self._last_dead_check_time = time.time()
        self._initialize_node_identity_state()
        self.hasher = hasher()

        self.default_kwargs = {
            "connect_timeout": connect_timeout,
            "timeout": timeout,
            "no_delay": no_delay,
            "socket_module": socket_module,
            "socket_keepalive": socket_keepalive,
            "key_prefix": key_prefix,
            "serde": serde,
            "serializer": serializer,
            "deserializer": deserializer,
            "allow_unicode_keys": allow_unicode_keys,
            "default_noreply": default_noreply,
            "encoding": encoding,
            "tls_context": tls_context,
        }

        if use_pooling is True:
            self.default_kwargs.update(
                {
                    "max_pool_size": max_pool_size,
                    "pool_idle_timeout": pool_idle_timeout,
                    "lock_generator": lock_generator,
                }
            )

        for server in servers:
            self.add_server(normalize_server_spec(server))
        self.encoding = encoding
        self.tls_context = tls_context

    def _make_client_key(self, server):
        if isinstance(server, (list, tuple)) and len(server) == 2:
            return "%s:%s" % server
        return server

    def _initialize_node_identity_state(self):
        self._next_node_id = 0
        self._node_ids = {}
        self._node_keys = {}
        self._active_node_ids = set()
        self._dead_node_ids = set()
        self._removed_node_ids = set()
        self._cas_tokens = weakref.WeakSet()

    def _add_node_identity(self, server_key):
        if not hasattr(self, "_node_ids"):
            self._initialize_node_identity_state()

        old_node_id = self._node_ids.get(server_key)

        node_id = self._next_node_id
        self._next_node_id += 1
        self._node_ids[server_key] = node_id
        self._node_keys[node_id] = server_key
        if old_node_id is not None:
            self._active_node_ids.discard(old_node_id)
            self._dead_node_ids.discard(old_node_id)
            self._removed_node_ids.add(old_node_id)
        self._active_node_ids.add(node_id)
        return node_id

    def _ensure_node_identity(self, server_key):
        node_id = self._node_ids.get(server_key)
        if node_id is None:
            node_id = self._add_node_identity(server_key)
        return node_id

    def _node_id_for_client(self, client):
        return self._ensure_node_identity(self._make_client_key(client.server))

    def _mark_node_dead(self, server):
        server_key = self._make_client_key(server)
        node_id = self._ensure_node_identity(server_key)
        if server_key in getattr(self.hasher, "nodes", ()):
            self.hasher.remove_node(server_key)
        self._active_node_ids.discard(node_id)
        self._dead_node_ids.add(node_id)

    def _mark_server_dead(self, server):
        self._failed_clients.pop(server, None)
        self._dead_clients[server] = time.time()
        self._mark_node_dead(server)

    def _restore_dead_node(self, server):
        server_key = self._make_client_key(server)
        node_id = self._node_ids[server_key]
        self.hasher.add_node(server_key)
        self._dead_node_ids.discard(node_id)
        self._active_node_ids.add(node_id)
        return node_id

    def add_server(self, server, port=None, preserve_identity=False) -> None:
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)

        _class = PooledClient if self.use_pooling else self.client_class
        client = _class(server, **self.default_kwargs)
        if self.use_pooling:
            client.client_class = self.client_class

        key = self._make_client_key(server)
        if not hasattr(self, "_node_ids"):
            self._initialize_node_identity_state()
        node_id = self._node_ids.get(key)
        if preserve_identity and node_id in self._dead_node_ids:
            self._restore_dead_node(key)
        elif not preserve_identity or node_id not in self._active_node_ids:
            self._add_node_identity(key)
        self._failed_clients.pop(server, None)
        self._dead_clients.pop(server, None)
        self.clients[key] = client
        self.hasher.add_node(key)

    def remove_server(self, server, port=None) -> None:
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)

        key = self._make_client_key(server)
        self._failed_clients.pop(server, None)
        self._dead_clients.pop(server, None)
        if key in self.hasher.nodes:
            self.hasher.remove_node(key)
        node_id = self._ensure_node_identity(key)
        self._active_node_ids.discard(node_id)
        self._dead_node_ids.discard(node_id)
        self._removed_node_ids.add(node_id)

    def _retry_dead(self) -> None:
        current_time = time.time()
        ldc = self._last_dead_check_time
        # We have reached the retry timeout
        if current_time - ldc > self.dead_timeout:
            candidates = []
            for server, dead_time in self._dead_clients.items():
                if current_time - dead_time > self.dead_timeout:
                    candidates.append(server)
            for server in candidates:
                logger.debug("bringing server back into rotation %s", server)
                self._restore_dead_node(server)
                del self._dead_clients[server]
            self._last_dead_check_time = current_time

    def _split_key(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return key
        return key, key

    def _get_client(self, key):
        server_key, key = self._split_key(key)

        check_key_helper(server_key, self.allow_unicode_keys, self.key_prefix)
        if self._dead_clients:
            self._retry_dead()

        server = self.hasher.get_node(server_key)
        # We've ran out of servers to try
        if server is None:
            if self.ignore_exc is True:
                return None, key
            raise MemcacheError("All servers seem to be down right now")

        self._ensure_node_identity(server)
        return self.clients[server], key

    def _safely_run_func(self, client, func, default_val, *args, **kwargs):
        try:
            if client.server in self._failed_clients:
                # This server is currently failing, lets check if it is in
                # retry or marked as dead
                failed_metadata = self._failed_clients[client.server]

                # we haven't tried our max amount yet, if it has been enough
                # time lets just retry using it
                if failed_metadata["attempts"] < self.retry_attempts:
                    failed_time = failed_metadata["failed_time"]
                    if time.time() - failed_time > self.retry_timeout:
                        logger.debug("retrying failed server: %s", client.server)
                        result = func(*args, **kwargs)
                        # we were successful, lets remove it from the failed
                        # clients
                        self._failed_clients.pop(client.server)
                        return result
                    return default_val
                else:
                    # We've reached our max retry attempts, we need to mark
                    # the sever as dead
                    logger.debug("marking server as dead: %s", client.server)
                    self._mark_server_dead(client.server)

            result = func(*args, **kwargs)
            return result

        # Connecting to the server fail, we should enter
        # retry mode
        except OSError:
            self._mark_failed_server(client.server)

            # if we haven't enabled ignore_exc, don't move on gracefully, just
            # raise the exception
            if not self.ignore_exc:
                raise

            return default_val
        except Exception:
            # any exceptions that aren't socket.error we need to handle
            # gracefully as well
            if not self.ignore_exc:
                raise

            return default_val

    def _safely_run_set_many(self, client, values, *args, **kwargs):
        failed = []
        succeeded = []
        try:
            if client.server in self._failed_clients:
                # This server is currently failing, lets check if it is in
                # retry or marked as dead
                failed_metadata = self._failed_clients[client.server]

                # we haven't tried our max amount yet, if it has been enough
                # time lets just retry using it
                if failed_metadata["attempts"] < self.retry_attempts:
                    failed_time = failed_metadata["failed_time"]
                    if time.time() - failed_time > self.retry_timeout:
                        logger.debug("retrying failed server: %s", client.server)
                        succeeded, failed, err = self._set_many(
                            client, values, *args, **kwargs
                        )
                        if err is not None:
                            raise err
                        # we were successful, lets remove it from the failed
                        # clients
                        self._failed_clients.pop(client.server)
                        return failed
                    return values.keys()
                else:
                    # We've reached our max retry attempts, we need to mark
                    # the sever as dead
                    logger.debug("marking server as dead: %s", client.server)
                    self._mark_server_dead(client.server)

            succeeded, failed, err = self._set_many(client, values, *args, **kwargs)
            if err is not None:
                raise err

            return failed

        # Connecting to the server fail, we should enter
        # retry mode
        except OSError:
            self._mark_failed_server(client.server)

            # if we haven't enabled ignore_exc, don't move on gracefully, just
            # raise the exception
            if not self.ignore_exc:
                raise

            return list(set(values.keys()) - set(succeeded))
        except Exception:
            # any exceptions that aren't socket.error we need to handle
            # gracefully as well
            if not self.ignore_exc:
                raise

            return list(set(values.keys()) - set(succeeded))

    def _mark_failed_server(self, server):
        # This client has never failed, lets mark it for failure
        if server not in self._failed_clients and self.retry_attempts > 0:
            self._failed_clients[server] = {
                "failed_time": time.time(),
                "attempts": 0,
            }
        # We aren't allowing any retries, we should mark the server as
        # dead immediately
        elif server not in self._failed_clients and self.retry_attempts <= 0:
            self._failed_clients[server] = {
                "failed_time": time.time(),
                "attempts": 0,
            }
            logger.debug("marking server as dead %s", server)
            self._mark_server_dead(server)
        # This client has failed previously, we need to update the metadata
        # to reflect that we have attempted it again
        else:
            failed_metadata = self._failed_clients[server]
            failed_metadata["attempts"] += 1
            failed_metadata["failed_time"] = time.time()
            self._failed_clients[server] = failed_metadata

    def _run_cmd(self, cmd, key, default_val, *args, **kwargs):
        client, key = self._get_client(key)

        if client is None:
            return default_val

        func = getattr(client, cmd)
        args = list(args)
        args.insert(0, key)
        return self._safely_run_func(client, func, default_val, *args, **kwargs)

    def _make_cas_token(self, client, key, cas):
        if not hasattr(self, "_cas_tokens"):
            self._initialize_node_identity_state()

        item_key = self._split_key(key)[1]
        bound_key = check_key_helper(
            item_key, self.allow_unicode_keys, self.key_prefix
        )
        token = CasToken._create(self._node_id_for_client(client), bound_key, cas)
        self._cas_tokens.add(token)
        return token

    def _get_cas_token_client(self, key, token):
        if not hasattr(self, "_cas_tokens"):
            self._initialize_node_identity_state()

        if not isinstance(token, CasToken) or token not in self._cas_tokens:
            raise MemcacheIllegalInputError("Invalid CAS token: %r" % (token,))

        item_key = self._split_key(key)[1]
        bound_key = check_key_helper(
            item_key, self.allow_unicode_keys, self.key_prefix
        )
        if token._key != bound_key:
            raise MemcacheIllegalInputError(
                "CAS token was issued for a different key"
            )

        if self._dead_clients:
            self._retry_dead()

        node_id = token._node_id
        if node_id in self._active_node_ids:
            return self.clients[self._node_keys[node_id]]
        if node_id in self._dead_node_ids:
            raise MemcacheNodeUnavailableError(
                "CAS token's node is temporarily unavailable"
            )
        if node_id in self._removed_node_ids:
            raise MemcacheServerRemovedError(
                "CAS token's node has been permanently removed"
            )
        raise MemcacheIllegalInputError("Invalid CAS token: %r" % (token,))

    def _set_many(self, client, values, *args, **kwargs):
        failed = []
        succeeded = []

        try:
            failed = client.set_many(values, *args, **kwargs)
        except Exception as e:
            if not self.ignore_exc:
                return succeeded, failed, e

        succeeded = [key for key in values if key not in failed]
        return succeeded, failed, None

    def close(self):
        for client in self.clients.values():
            self._safely_run_func(client, client.close, False)

    disconnect_all = close

    def set(self, key, *args, **kwargs):
        return self._run_cmd("set", key, False, *args, **kwargs)

    def get(self, key, default=None, **kwargs):
        return self._run_cmd("get", key, default, default=default, **kwargs)

    def gat(self, key, default=None, **kwargs):
        return self._run_cmd("gat", key, default, default=default, **kwargs)

    def gats(self, key, default=None, **kwargs):
        return self._run_cmd("gats", key, default, default=default, **kwargs)

    def incr(self, key, *args, **kwargs):
        return self._run_cmd("incr", key, None, *args, **kwargs)

    def decr(self, key, *args, **kwargs):
        return self._run_cmd("decr", key, None, *args, **kwargs)

    def set_many(self, values, *args, **kwargs):
        client_batches = collections.defaultdict(dict)
        failed = []

        for key, value in values.items():
            client, key = self._get_client(key)

            if client is None:
                failed.append(key)
                continue

            client_batches[client.server][key] = value

        for server, values in client_batches.items():
            client = self.clients[self._make_client_key(server)]
            failed += self._safely_run_set_many(client, values, *args, **kwargs)

        return failed

    set_multi = set_many

    def get_many(self, keys, gets=False, *args, **kwargs):
        client_batches = collections.defaultdict(list)
        end = {}

        for key in keys:
            client, key = self._get_client(key)

            if client is None:
                continue

            client_batches[client.server].append(key)

        for server, keys in client_batches.items():
            client = self.clients[self._make_client_key(server)]
            new_args = list(args)
            new_args.insert(0, keys)

            if gets:
                get_func = client.gets_many
            else:
                get_func = client.get_many

            result = self._safely_run_func(client, get_func, {}, *new_args, **kwargs)
            end.update(result)

        return end

    get_multi = get_many

    def gets(self, key, *args, **kwargs):
        return self._run_cmd("gets", key, None, *args, **kwargs)

    def gets_with_cas_token(self, key, default=None, cas_default=None):
        """Fetch a value with an opaque, route-bound CAS credential."""
        client, item_key = self._get_client(key)
        if client is None:
            return default, cas_default

        value, cas = self._safely_run_func(
            client,
            client.gets,
            (default, cas_default),
            item_key,
            default=default,
            cas_default=cas_default,
        )
        if cas is cas_default:
            return value, cas_default
        return value, self._make_cas_token(client, item_key, cas)

    gets_with_token = gets_with_cas_token
    gets_with_cas_credential = gets_with_cas_token
    gets_with_credential = gets_with_cas_token

    def gets_many(self, keys, *args, **kwargs):
        return self.get_many(keys, gets=True, *args, **kwargs)

    gets_multi = gets_many

    def add(self, key, *args, **kwargs):
        return self._run_cmd("add", key, False, *args, **kwargs)

    def prepend(self, key, *args, **kwargs):
        return self._run_cmd("prepend", key, False, *args, **kwargs)

    def append(self, key, *args, **kwargs):
        return self._run_cmd("append", key, False, *args, **kwargs)

    def delete(self, key, *args, **kwargs):
        return self._run_cmd("delete", key, False, *args, **kwargs)

    def delete_many(self, keys, *args, **kwargs) -> bool:
        for key in keys:
            self._run_cmd("delete", key, False, *args, **kwargs)
        return True

    delete_multi = delete_many

    def cas(
        self,
        key,
        value=None,
        cas=None,
        expire=0,
        noreply=False,
        flags=None,
        cas_token=None,
    ):
        if cas_token is None and isinstance(cas, CasToken):
            cas_token = cas

        if cas_token is not None:
            return self.cas_with_cas_token(
                key,
                value,
                cas_token,
                expire=expire,
                noreply=noreply,
                flags=flags,
            )

        return self._run_cmd(
            "cas",
            key,
            False,
            value,
            cas,
            expire=expire,
            noreply=noreply,
            flags=flags,
        )

    def cas_with_cas_token(
        self, key, value, cas_token, expire=0, noreply=False, flags=None
    ):
        """Perform CAS using a credential returned by gets_with_cas_token."""
        item_key = self._split_key(key)[1]
        client = self._get_cas_token_client(item_key, cas_token)
        return self._safely_run_func(
            client,
            client.cas,
            False,
            item_key,
            value,
            cas_token.cas,
            expire=expire,
            noreply=noreply,
            flags=flags,
        )

    cas_with_token = cas_with_cas_token
    cas_with_cas_credential = cas_with_cas_token
    cas_with_credential = cas_with_cas_token
    cas_token = cas_with_cas_token
    cas_credential = cas_with_cas_token

    def replace(self, key, *args, **kwargs):
        return self._run_cmd("replace", key, False, *args, **kwargs)

    def touch(self, key, *args, **kwargs):
        return self._run_cmd("touch", key, False, *args, **kwargs)

    def stats(self, *args, **kwargs):
        result = list()
        for key, client in self.clients.items():
            result.append(
                (
                    key,
                    self._safely_run_func(client, client.stats, False, *args, **kwargs),
                )
            )
        return result

    def flush_all(self, *args, **kwargs) -> None:
        for client in self.clients.values():
            self._safely_run_func(client, client.flush_all, False, *args, **kwargs)

    def quit(self) -> None:
        for client in self.clients.values():
            self._safely_run_func(client, client.quit, False)
