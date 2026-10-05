import collections
import hashlib
import hmac
import itertools
import secrets
import socket
import time
import logging

from pymemcache.client.base import (
    Client,
    PooledClient,
    check_key_helper,
    normalize_server_spec,
)
from pymemcache.client.rendezvous import RendezvousHash
from pymemcache.exceptions import MemcacheError
from pymemcache.exceptions import MemcacheClientError

logger = logging.getLogger(__name__)
_client_identities = itertools.count(1)
_missing_cas = object()


class CASToken:
    """
    An opaque CAS value bound to the key and node that produced it.

    Token objects are created by HashClient and must not be constructed by
    application code. They contain routing metadata, but that metadata is not
    trusted when a token is supplied to a different HashClient.
    """

    __slots__ = (
        "_frozen",
        "client_id",
        "key",
        "server_key",
        "generation",
        "cas",
        "signature",
    )

    def __init__(
        self, client_id, key, server_key, generation, cas, signature
    ):
        self._frozen = False
        self.client_id = client_id
        self.key = key
        self.server_key = server_key
        self.generation = generation
        self.cas = cas
        self.signature = signature
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("CASToken is immutable")
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if getattr(self, "_frozen", False):
            raise AttributeError("CASToken is immutable")
        super().__delattr__(name)

    def __repr__(self):
        return f"<{type(self).__name__}>"


CasToken = CASToken


class HashClient:
    """
    A client for communicating with a cluster of memcached servers
    """

    #: :class:`Client` class used to create new clients
    client_class = Client
    _client_id = None
    _generation_values = None
    _node_generations = None
    _token_secret = None

    def _ensure_routing_state(self):
        if self._client_id is None:
            self._client_id = next(_client_identities)
        if self._generation_values is None:
            self._generation_values = itertools.count(1)
        if self._node_generations is None:
            self._node_generations = {}
        if self._token_secret is None:
            self._token_secret = secrets.token_bytes(32)

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
        self._ensure_routing_state()
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

    def _new_generation(self, key):
        self._ensure_routing_state()
        generation = next(self._generation_values)
        self._node_generations[key] = generation
        return generation

    def _token_signature(self, token):
        message = hmac.new(self._token_secret, digestmod=hashlib.sha256)

        def update(value):
            if isinstance(value, bytes):
                message.update(value)
            else:
                message.update(str(value).encode("utf-8"))
            message.update(b"\0")

        update(token.client_id)
        update(token.key)
        update(token.server_key)
        update(token.generation)
        update(repr(token.cas))
        return message.digest()

    def add_server(self, server, port=None) -> None:
        """Add a server to the hash ring and create a new server generation."""
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)
        server = normalize_server_spec(server)
        self._dead_clients.pop(server, None)
        self._failed_clients.pop(server, None)

        key = self._make_client_key(server)
        _class = PooledClient if self.use_pooling else self.client_class
        client = _class(server, **self.default_kwargs)
        if self.use_pooling:
            client.client_class = self.client_class

        old_client = self.clients.pop(key, None)
        if old_client is not None:
            self._safely_run_func(old_client, old_client.close, False)

        self.clients[key] = client
        self._new_generation(key)
        self.hasher.add_node(key)

    def _remove_server_temporarily(self, server):
        key = self._make_client_key(server)
        try:
            self.hasher.remove_node(key)
        except ValueError:
            pass

    def _remove_server_permanently(self, server):
        key = self._make_client_key(server)
        client = self.clients.pop(key, None)
        if client is not None:
            self._safely_run_func(client, client.close, False)
        self._new_generation(key)
        self._remove_server_temporarily(server)

    def remove_server(self, server, port=None, permanent=False) -> None:
        """
        Remove a server from the hash ring.

        A temporary removal keeps the client and its node identity for use when
        ``dead_timeout`` expires. A permanent removal closes the client and
        invalidates CAS tokens issued by that node.
        """
        # To maintain backward compatibility, if a port is provided, assume
        # that server wasn't provided as a (host, port) tuple.
        if port is not None:
            if not isinstance(server, str):
                raise TypeError("Server must be a string when passing port.")
            server = (server, port)

        server = normalize_server_spec(server)
        self._failed_clients.pop(server, None)

        if permanent:
            self._dead_clients.pop(server, None)
            self._remove_server_permanently(server)
        else:
            self._dead_clients[server] = time.time()
            self._remove_server_temporarily(server)

    def _restore_server(self, server):
        key = self._make_client_key(server)
        self.hasher.add_node(key)

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
                self._restore_server(server)
                del self._dead_clients[server]
            self._last_dead_check_time = current_time

    def _get_client(self, key):
        # If key is tuple use first item as server key
        if isinstance(key, tuple) and len(key) == 2:
            server_key, key = key
        else:
            server_key = key

        check_key_helper(server_key, self.allow_unicode_keys, self.key_prefix)
        if self._dead_clients:
            self._retry_dead()

        server = self.hasher.get_node(server_key)
        # We've ran out of servers to try
        if server is None:
            if self.ignore_exc is True:
                return None, key
            raise MemcacheError("All servers seem to be down right now")

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
                    self.remove_server(client.server, permanent=False)

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
                    self.remove_server(client.server, permanent=False)

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
            self.remove_server(server, permanent=False)
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

    def _split_key(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            return key[1], key[0]
        return key, key

    def _get_token_client(self, key, token, retry=True):
        self._ensure_routing_state()
        if not isinstance(token, CASToken):
            raise TypeError("cas token must be a CASToken instance")
        if token.client_id != self._client_id:
            raise MemcacheClientError("CAS token was issued by a different client")
        if not hmac.compare_digest(
            token.signature, self._token_signature(token)
        ):
            raise MemcacheClientError("CAS token is invalid")

        item_key, server_key = self._split_key(key)
        if token.key != item_key:
            raise MemcacheClientError("CAS token is bound to a different key")

        server_key_name = token.server_key
        if retry and self._dead_clients:
            self._retry_dead()

        temporary_server = any(
            self._make_client_key(server) == server_key_name
            for server in self._dead_clients
        )
        if temporary_server:
            raise MemcacheClientError("CAS token's server is temporarily unavailable")

        if server_key_name not in self.clients:
            raise MemcacheClientError(
                "CAS token's server has been permanently removed"
            )
        if self._node_generations.get(server_key_name) != token.generation:
            raise MemcacheClientError("CAS token's server has been replaced")

        check_key_helper(server_key, self.allow_unicode_keys, self.key_prefix)
        return self.clients[server_key_name], item_key

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

    def gets_with_token(
        self, key, default=None, cas_default=_missing_cas, **kwargs
    ):
        """
        Fetch a key and return an opaque, route-bound CAS token on a hit.
        """
        client, item_key = self._get_client(key)
        effective_cas_default = (
            None if cas_default is _missing_cas else cas_default
        )
        if client is None:
            return default, effective_cas_default

        underlying_cas_default = object()
        defaults = (default, underlying_cas_default)
        result = self._safely_run_func(
            client,
            client.gets,
            defaults,
            item_key,
            default=default,
            cas_default=underlying_cas_default,
            **kwargs,
        )
        if result is defaults:
            return default, effective_cas_default
        value, cas = result
        if cas is underlying_cas_default:
            return default, effective_cas_default

        _, server_key = self._split_key(key)
        server_key_name = self._make_client_key(client.server)
        token = CASToken.__new__(CASToken)
        token._frozen = False
        token.client_id = self._client_id
        token.key = item_key
        token.server_key = server_key_name
        token.generation = self._node_generations[server_key_name]
        token.cas = cas
        token.signature = self._token_signature(token)
        token._frozen = True
        return value, token

    gets_token = gets_with_token
    get_cas_token = gets_with_token

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

    def cas(self, key, *args, **kwargs):
        """
        Perform a raw CAS update, or update by token when given a CASToken.
        """
        token = args[1] if len(args) > 1 else kwargs.get("cas")
        if isinstance(token, CASToken):
            if len(args) > 1:
                return self.cas_with_token(
                    key, args[0], token, *args[2:], **kwargs
                )
            value = args[0]
            kwargs.pop("cas")
            return self.cas_with_token(key, value, token, **kwargs)
        return self._run_cmd("cas", key, False, *args, **kwargs)

    def cas_with_token(self, key, value, token, *args, **kwargs):
        """
        Perform a CAS update using only the node and key bound to ``token``.
        """
        client, item_key = self._get_token_client(key, token)
        return self._safely_run_func(
            client,
            client.cas,
            False,
            item_key,
            value,
            token.cas,
            *args,
            **kwargs,
        )

    cas_token = cas_with_token

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
