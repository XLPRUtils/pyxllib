import os
import shlex
import subprocess
import sys
import time

from typing import Any, Dict, List, Set


WINDOWS_CREATE_NEW_PROCESS_GROUP = 0x00000200
WINDOWS_CREATE_BREAKAWAY_FROM_JOB = 0x01000000
WINDOWS_CREATE_NO_WINDOW = 0x08000000
WINDOWS_TH32CS_SNAPPROCESS = 0x00000002
WINDOWS_MAX_PATH = 260


def build_background_popen_kwargs(independent=False):
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "close_fds": True,
    }

    if sys.platform == "win32":
        creationflags = WINDOWS_CREATE_NO_WINDOW
        if independent:
            creationflags |= WINDOWS_CREATE_NEW_PROCESS_GROUP | WINDOWS_CREATE_BREAKAWAY_FROM_JOB
        kwargs["creationflags"] = creationflags
    elif independent:
        kwargs["start_new_session"] = True

    return kwargs


def parse_cmdline(cmdline):
    if sys.platform != "win32":
        return shlex.split(cmdline, posix=True)

    try:
        import ctypes
        from ctypes import wintypes

        ctypes.windll.shell32.CommandLineToArgvW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
        ctypes.windll.shell32.CommandLineToArgvW.restype = ctypes.POINTER(wintypes.LPWSTR)

        nargs = ctypes.c_int()
        res = ctypes.windll.shell32.CommandLineToArgvW(cmdline, ctypes.byref(nargs))
        if not res:
            return shlex.split(cmdline, posix=False)

        try:
            return [res[i] for i in range(nargs.value)]
        finally:
            ctypes.windll.kernel32.LocalFree(res)
    except Exception:
        return shlex.split(cmdline, posix=False)


def match_cmdline(target_cmd, proc_cmdline):
    try:
        try:
            target_args = shlex.split(target_cmd, posix=(sys.platform != "win32"))
        except Exception:
            target_args = target_cmd.split()

        if sys.platform == "win32":
            target_args = [arg.strip('"') for arg in target_args]

        if not target_args:
            return False

        n = len(target_args)
        if len(proc_cmdline) < n:
            return False

        for i in range(len(proc_cmdline) - n + 1):
            if proc_cmdline[i : i + n] == target_args:
                return True

        first_arg = target_args[0]
        if first_arg.startswith("python") or first_arg.endswith("python.exe") or first_arg.endswith("python"):
            rest_target = target_args[1:]
            if not rest_target:
                return False

            n_rest = len(rest_target)
            for i in range(len(proc_cmdline) - n_rest + 1):
                if proc_cmdline[i : i + n_rest] == rest_target:
                    return True

        return False
    except Exception:
        return False


def candidate_process_names_for_command(command):
    try:
        args = parse_cmdline(command)
    except Exception:
        args = command.split()

    if not args:
        return set()

    exe_name = os.path.basename(str(args[0]).strip('"')).lower()
    if not exe_name:
        return set()

    names = {exe_name}
    root, ext = os.path.splitext(exe_name)
    if sys.platform == "win32":
        if not ext:
            names.add("%s.exe" % exe_name)
            names.add("%s.cmd" % exe_name)
            names.add("%s.bat" % exe_name)
        if ext in {".cmd", ".bat"}:
            names.add("cmd.exe")
        if root in {"python", "python3", "py"} or exe_name.startswith("python"):
            names.update({"python.exe", "py.exe"})

    return names


def _windows_process_ids_by_name(process_names):
    wanted = {name.lower() for name in process_names if name}
    if not wanted:
        return []

    import ctypes
    from ctypes import wintypes

    class WindowsProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * WINDOWS_MAX_PATH),
        ]

    kernel32 = ctypes.windll.kernel32
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(WindowsProcessEntry32)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(WindowsProcessEntry32)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(WINDOWS_TH32CS_SNAPPROCESS, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        return []

    pids = []
    entry = WindowsProcessEntry32()
    entry.dwSize = ctypes.sizeof(WindowsProcessEntry32)
    entry_ptr = ctypes.pointer(entry)
    try:
        has_entry = kernel32.Process32FirstW(snapshot, entry_ptr)
        while has_entry:
            if entry.szExeFile.lower() in wanted:
                pids.append(int(entry.th32ProcessID))
            has_entry = kernel32.Process32NextW(snapshot, entry_ptr)
    finally:
        kernel32.CloseHandle(snapshot)
    return pids


def _psutil_process_candidates_by_name(process_names):
    import psutil

    wanted = {name.lower() for name in process_names if name}
    processes = []
    for proc in psutil.process_iter(["pid", "name", "create_time"]):
        try:
            info = getattr(proc, "info", {}) or {}
            proc_name = str(info.get("name") or "").lower()
            if proc_name in wanted:
                processes.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return processes


def process_candidates_by_name(process_names):
    if not process_names:
        return []

    import psutil

    if sys.platform == "win32":
        processes = []
        seen_pids = set()
        try:
            pids = _windows_process_ids_by_name(process_names)
        except Exception:
            return _psutil_process_candidates_by_name(process_names)
        for pid in pids:
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            try:
                processes.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        return processes

    return _psutil_process_candidates_by_name(process_names)


def process_candidates_for_command(command):
    return process_candidates_by_name(candidate_process_names_for_command(command))


def terminate_process_tree(pid, timeout=5.0):
    import psutil

    try:
        root = psutil.Process(int(pid))
        targets = [*root.children(recursive=True), root]
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError, ValueError):
        return True

    for proc in reversed(targets):
        try:
            proc.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    try:
        _, alive = psutil.wait_procs(targets, timeout=max(0.1, float(timeout)))
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        alive = [proc for proc in targets if _pid_exists(proc.pid)]
    for proc in alive:
        try:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass

    if alive:
        try:
            psutil.wait_procs(alive, timeout=max(0.1, float(timeout)))
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            pass
    time.sleep(0.1)
    return not any(_pid_exists(proc.pid) for proc in targets)


def _pid_exists(pid):
    import psutil

    try:
        return psutil.pid_exists(int(pid))
    except Exception:
        return False
