"""Compatibility namespace for the shared KQ5034 business package."""

from kq5034 import __all__ as __all__  # noqa: F401


def __getattr__(name):
    import kq5034

    value = getattr(kq5034, name)
    globals()[name] = value
    return value
