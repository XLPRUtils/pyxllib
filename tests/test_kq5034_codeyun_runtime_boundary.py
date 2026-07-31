from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
KQ_RUNTIME_ROOT = PROJECT_ROOT / "src" / "kq5034"


def test_kq5034_runtime_does_not_import_retired_table_stack():
    forbidden = (
        "WpsOnlineBook",
        "pyxllib.ext.wpsapi",
        "KqBook",
        "sync_kqbook_order_sheet",
    )
    violations = []
    for path in KQ_RUNTIME_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {marker}")
    assert not violations, "\n".join(violations)


def test_common_public_entrypoints_do_not_export_retired_table_stack():
    for relative in ("src/pyxllib/api.py", "src/pyxllib/xlwork.py"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "WpsOnlineBook" not in text
        assert "get_airscript_head2" not in text


def test_historical_interface_is_explicit_and_template_is_available():
    from pyxllib.legacy.wps_jsa import WpsOnlineBook, load_airscript_template

    assert WpsOnlineBook
    assert "function" in load_airscript_template()
