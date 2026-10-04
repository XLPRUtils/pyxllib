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


def recover_attendance_browser():
    """Explicitly restore the existing attendance profile, preserving its cookies.

    Normal collection remains existing-only. Recovery refuses missing profiles
    and ambiguous identities; an occupied port is never treated as attendance.
    A profile-scoped lock makes concurrent recovery calls reuse one browser.
    """
    import psutil
    from filelock import FileLock
    from DrissionPage import Chromium, ChromiumOptions

    profile = Path(os.environ.get('KQ_BROWSER_USER_DATA_DIR') or str(
        Path(tempfile.gettempdir()) / 'DrissionPage/userData/9222'))
    if not profile.is_dir():
        raise RuntimeError(f'考勤 profile 不存在，禁止创建新身份：{profile}')
    with FileLock(str(profile.parent / (profile.name + '.recovery.lock')), timeout=10):
        processes = [p.info.get('cmdline') or [] for p in psutil.process_iter(['name', 'cmdline'])
                     if (p.info.get('name') or '').lower() in {'chrome.exe', 'msedge.exe'}]
        expected = os.path.normcase(os.path.abspath(profile))
        owners = [args for args in processes if any(
            arg.startswith('--user-data-dir=') and
            os.path.normcase(os.path.abspath(arg.split('=', 1)[1].strip('"'))) == expected
            for arg in args) and not any(arg.startswith('--type=') for arg in args)]
        if owners:
            address = attendance_browser_address(owners, profile=str(profile))
            return Chromium(ChromiumOptions().set_address(address).existing_only())
        occupied = {c.laddr.port for c in psutil.net_connections(kind='tcp') if c.status == 'LISTEN'}
        port = next((p for p in range(9222, 9233) if p not in occupied), None)
        if port is None:
            raise RuntimeError('考勤浏览器恢复端口均被占用')
        options = ChromiumOptions().set_local_port(port).set_user_data_path(str(profile))
        return Chromium(options)
