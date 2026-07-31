from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from wps_dp.artifacts import (
    build_requested_actions,
    build_run_dir_name,
    resolve_save_dir,
)


def test_build_requested_actions_prefers_real_steps():
    actions = build_requested_actions(
        create_copy=True,
        open_now=True,
        rename_title="新名字",
        share_requested=True,
    )
    assert actions == ["copy", "open", "rename", "share"]


def test_build_requested_actions_falls_back_to_probe():
    assert build_requested_actions() == ["probe"]


def test_resolve_save_dir_uses_runs_root_when_not_explicit():
    save_dir = resolve_save_dir(
        explicit_save_dir=None,
        save_root=Path("wps_dp") / "output" / "runs",
        url="https://www.kdocs.cn/l/capdA7mscqov",
        actions=["copy", "open"],
        label="第39届念住",
        now=datetime(2026, 4, 9, 10, 15, 30),
    )
    assert save_dir == Path(
        "wps_dp/output/runs/20260409_101530__copy-open__第39届念住__capdA7mscqov"
    )


def test_build_run_dir_name_drops_invalid_path_chars():
    dirname = build_run_dir_name(
        "https://www.kdocs.cn/l/capdA7mscqov",
        actions=["share"],
        label='觉观45:/ fix?',
        now=datetime(2026, 4, 9, 10, 15, 30),
    )
    assert dirname == "20260409_101530__share__觉观45_fix__capdA7mscqov"
