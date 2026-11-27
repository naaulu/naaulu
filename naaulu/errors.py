class NoDataError(Exception):
    """Raised when expected data is missing or empty."""
    pass


class OfflineError(Exception):
    """No-network mode is active and the resource is not in the local cache."""
    pass
