from __future__ import annotations

import os
from pathlib import Path
import runpy


COLOR_MODULE = Path(__file__).resolve().parents[3] / "src" / "wxautox" / "color.py"


def test_color_module_import_does_not_spawn_a_shell(monkeypatch) -> None:
    def fail_system(_command: str) -> int:
        raise AssertionError("wxautox.color must not call os.system during import")

    monkeypatch.setattr(os, "system", fail_system)

    namespace = runpy.run_path(os.fspath(COLOR_MODULE))

    assert "Print" in namespace
    assert "Input" in namespace
    assert "Warnings" in namespace
