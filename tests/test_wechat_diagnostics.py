import pytest

from pyxllib.autogui import wechat_diagnostics as diag


def test_sync_stage_records_bounded_totals_without_payload():
    with diag.sync_stage("sync.copy") as units:
        units["value"] = 3
    with pytest.raises(RuntimeError):
        with diag.sync_stage("sync.copy"):
            raise RuntimeError("stage failure is still counted")
    stats = diag.sync_stage_diagnostics()["sync.copy"]
    assert stats["count"] >= 2 and stats["units"] >= 3
    assert set(stats) == {"count", "seconds", "units", "private_delta_mb", "positive_delta_mb", "max_delta_mb"}


def test_unknown_stage_name_is_rejected():
    with pytest.raises(ValueError):
        with diag.sync_stage("updates.C:/private/path"):
            pass
