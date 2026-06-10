import os
import sys
from pathlib import Path

import pytest

from pyxllib.prog import process_runtime


def _current_python_process_names():
    return {Path(sys.executable).name.lower(), "python.exe", "python", "python3.exe", "python3"}


def test_process_candidates_by_name_finds_current_python_process():
    processes = process_runtime.process_candidates_by_name(_current_python_process_names())

    assert any(proc.pid == os.getpid() for proc in processes)


def test_windows_process_candidates_by_name_falls_back_to_psutil(monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows-specific fallback path")

    def fail_snapshot(_process_names):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(process_runtime, "_windows_process_ids_by_name", fail_snapshot)

    processes = process_runtime.process_candidates_by_name(_current_python_process_names())

    assert any(proc.pid == os.getpid() for proc in processes)


def test_root_process_records_excludes_children():
    records = [
        {"pid": 1, "parent_pid": 0, "name": "root-a"},
        {"pid": 2, "parent_pid": 1, "name": "child-a"},
        {"pid": 3, "parent_pid": 99, "name": "root-b"},
        {"pid": "bad", "parent_pid": 1, "name": "invalid-pid"},
    ]

    assert process_runtime.root_process_records(records) == [
        {"pid": 1, "parent_pid": 0, "name": "root-a"},
        {"pid": 3, "parent_pid": 99, "name": "root-b"},
    ]


def test_terminate_process_tree_tolerates_access_denied_wait(monkeypatch):
    import psutil

    calls = []

    class FakeProcess:
        pid = 123

        def children(self, recursive=True):
            return []

        def terminate(self):
            calls.append("terminate")

        def kill(self):
            calls.append("kill")

    fake_process = FakeProcess()
    monkeypatch.setattr(psutil, "Process", lambda pid: fake_process)
    monkeypatch.setattr(psutil, "wait_procs", lambda *_args, **_kwargs: (_ for _ in ()).throw(psutil.AccessDenied(pid=123)))
    monkeypatch.setattr(process_runtime, "_pid_exists", lambda pid: False)
    monkeypatch.setattr(process_runtime.time, "sleep", lambda _seconds: None)

    assert process_runtime.terminate_process_tree(123) is True
    assert calls == ["terminate"]
