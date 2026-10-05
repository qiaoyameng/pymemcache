class MemcacheError(Exception):
    "Base exception class"
    pass


class MemcacheClientError(MemcacheError):
    """Raised when memcached fails to parse the arguments to a request, likely
    due to a malformed key and/or value, a bug in this library, or a version
    mismatch with memcached."""

    pass


class MemcacheUnknownCommandError(MemcacheClientError):
    """Raised when memcached fails to parse a request, likely due to a bug in
    this library or a version mismatch with memcached."""

    pass


class MemcacheIllegalInputError(MemcacheClientError):
    """Raised when a key or value is not legal for Memcache (see the class docs
    for Client for more details)."""

    pass


class MemcacheServerError(MemcacheError):
    """Raised when memcached reports a failure while processing a request,
    likely due to a bug or transient issue in memcached."""

    pass


class MemcacheUnknownError(MemcacheError):
    """Raised when this library receives a response from memcached that it
    cannot parse, likely due to a bug in this library or a version mismatch
    with memcached."""

    pass


class MemcacheUnexpectedCloseError(MemcacheServerError):
    "Raised when the connection with memcached closes unexpectedly."
    pass


class MemcacheNodeUnavailableError(MemcacheServerError):
    """Raised when a route-bound request targets a node that is temporarily
    unavailable and may later return with the same identity."""

    pass


class MemcacheServerRemovedError(MemcacheClientError):
    """Raised when a route-bound credential was issued by a server that has
    been permanently removed from the HashClient."""

    pass
