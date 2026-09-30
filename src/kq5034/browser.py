"""Resolve the existing attendance browser by profile, never the daily Chrome."""
import os
from pathlib import Path
import tempfile


def attendance_browser_address(processes, *, profile: str) -> str:
    """Match the browser parent process to an exact configured user-data directory."""
    expected = os.path.normcase(os.path.abspath(profile))
    matches = set()
    for args in processes:
        flags = dict(arg.split('=', 1) for arg in args if arg.startswith('--') and '=' in arg)
        if '--type' in flags:
            continue
        directory = flags.get('--user-data-dir', '').strip('"')
        if directory and os.path.normcase(os.path.abspath(directory)) == expected:
            port = flags.get('--remote-debugging-port', '')
            if port.isdigit() and 0 < int(port) < 65536:
                matches.add(f'127.0.0.1:{port}')
    if len(matches) != 1:
        raise RuntimeError(f'考勤 DP 浏览器未运行或身份不唯一：profile={profile}, addresses={sorted(matches)}')
    return matches.pop()


def connect_attendance_browser():
    """Reuse the running attendance profile; missing browser requires explicit recovery."""
    import psutil
    from DrissionPage import Chromium, ChromiumOptions
    profile = os.environ.get('KQ_BROWSER_USER_DATA_DIR') or str(
        Path(tempfile.gettempdir()) / 'DrissionPage/userData/9222')
    processes = []
    for process in psutil.process_iter(['name', 'cmdline']):
        if (process.info.get('name') or '').lower() in {'chrome.exe', 'msedge.exe'}:
            processes.append(process.info.get('cmdline') or [])
    address = attendance_browser_address(processes, profile=profile)
    options = ChromiumOptions().set_address(address).existing_only()
    return Chromium(options)
