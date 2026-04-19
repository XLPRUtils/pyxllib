"""Shared KQ5034 business package with legacy star-import compatibility."""

from importlib import import_module


def _import_legacy_module():
    return import_module("kq5034.kqtools")


def _load_extra_exports():
    extras = {}

    try:
        import fire as _fire
    except Exception:
        pass
    else:
        extras["fire"] = _fire

    try:
        from pyxllib.text.document import Document as _Document
    except Exception:
        pass
    else:
        extras["Document"] = _Document

    return extras


_EXTRA_EXPORTS = _load_extra_exports()
_LEGACY_MODULE = None
_LEGACY_EXPORTS = set()


def _update_legacy_exports(module):
    global __all__
    _LEGACY_EXPORTS.update(name for name in dir(module) if not name.startswith("_"))
    __all__ = sorted(_LEGACY_EXPORTS | set(_EXTRA_EXPORTS))


def _load_legacy_module():
    global _LEGACY_MODULE
    if _LEGACY_MODULE is None:
        _LEGACY_MODULE = _import_legacy_module()
        _update_legacy_exports(_LEGACY_MODULE)
    return _LEGACY_MODULE


try:
    _update_legacy_exports(_load_legacy_module())
except Exception:
    # Allow lightweight submodules such as `kq5034.order_ops` to be imported
    # even when the full legacy runtime dependency set is unavailable.
    __all__ = sorted(set(_EXTRA_EXPORTS))


def __getattr__(name):
    if name in _EXTRA_EXPORTS:
        value = _EXTRA_EXPORTS[name]
    else:
        module = _load_legacy_module()
        if not hasattr(module, name):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(module, name)

    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
