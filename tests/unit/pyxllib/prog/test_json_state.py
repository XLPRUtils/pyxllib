from __future__ import annotations

import json

from pyxllib.prog.json_state import read_json_state, read_json_state_dict, write_json_state


def test_write_and_read_json_state_roundtrip(tmp_path):
    path = tmp_path / "state" / "runtime.json"

    write_json_state(path, {"message": "完成", "items": [1, 2]})

    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "完成", "items": [1, 2]}
    assert read_json_state(path, {}) == {"message": "完成", "items": [1, 2]}
    assert not list(path.parent.glob("*.tmp"))


def test_read_json_state_returns_default_for_missing_or_bad_json(tmp_path):
    path = tmp_path / "bad.json"

    assert read_json_state(path, ["default"]) == ["default"]

    path.write_text("{bad", encoding="utf-8")

    assert read_json_state(path, ["default"]) == ["default"]


def test_read_json_state_dict_filters_non_dict_payload(tmp_path):
    path = tmp_path / "state.json"

    path.write_text("[1, 2]", encoding="utf-8")

    assert read_json_state_dict(path, default={"ok": False}) == {"ok": False}
