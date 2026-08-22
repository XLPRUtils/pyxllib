"""微信 4.1.12.55 进程内纯文本发送 API；没有 GUI 降级路径。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

WEIXIN_DLL = Path(r"C:\Program Files\Tencent\Weixin\4.1.12.55\Weixin.dll")
EXPECTED_SHA256 = "7AD9753D11C2BAF5C900AAC50DDF56A8170AA85C46129D661325FF88505BEFB1"
DEFAULT_CONTACT_DB = Path(r"D:\home\chenkunze\data\d2605微信逆向\decrypted\db_storage\contact\contact.db")
RECIPIENT_ALIASES = {"文件传输助手": "filehelper"}
NATIVE_SOURCE = Path(__file__).with_name("native") / "weixin_4_1_12_send.c"
NATIVE_ADAPTER = NATIVE_SOURCE.with_suffix(".dll")


class WeixinInstrumentationError(RuntimeError):
    """进程内 API 发送不能安全完成。"""


class WeixinInstrumentationUnavailable(WeixinInstrumentationError):
    """API 不可用；调用方必须失败关闭，禁止降级 GUI。"""


def _repair_legacy_text(value: str) -> str:
    try:
        repaired = value.encode("latin1").decode("gb18030")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value
    return repaired if repaired else value


def resolve_recipient_id(name: str, contact_db: str | Path = DEFAULT_CONTACT_DB) -> str:
    requested = str(name).strip()
    target = RECIPIENT_ALIASES.get(requested, requested)
    if target == "filehelper":
        return target
    path = Path(contact_db)
    if not path.exists():
        raise WeixinInstrumentationUnavailable(f"微信联系人库不存在：{path}")
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT username, alias, remark, nick_name FROM contact "
            "WHERE COALESCE(delete_flag, 0) = 0"
        ).fetchall()
    finally:
        conn.close()
    matches: set[str] = set()
    for username, alias, remark, nickname in rows:
        labels = {
            _repair_legacy_text(str(value or "")).strip()
            for value in (username, alias, remark, nickname)
        }
        if target in labels:
            matches.add(str(username))
    if len(matches) != 1:
        raise WeixinInstrumentationError(
            f"微信收件人必须唯一匹配：{requested!r}，实际 {len(matches)} 个"
        )
    return matches.pop()


SCRIPT_TEMPLATE = r"""
const module = Process.getModuleByName('Weixin.dll');
const base = module.base;
const offsets = {
  getCoro: 0x42010, getSvc: 0x339480, getCtx: 0x6cea20,
  doSend: 0x1734330, sendEntry: 0x197d500, msgCtor: 0x738e90
};
function hex(value) {
  return Array.from(new Uint8Array(value)).map(v => v.toString(16).padStart(2, '0')).join('');
}
function validate() {
  const expected = {
    getCoro: '564883ec404889ce',
    getSvc: '4889d0488b51304885d27414f0ff4208488b5130488b492848890848895008c331d2488b492848890848895008c3',
    getCtx: '5556574881ec90000000488d',
    doSend: '554157415641554154565753',
    sendEntry: '5556574881ecd0000000488d',
    msgCtor: '5556574883ec50488d6c2450'
  };
  for (const [name, signature] of Object.entries(expected)) {
    const actual = hex(base.add(offsets[name]).readByteArray(signature.length / 2));
    if (actual !== signature) throw new Error(name + ' runtime signature mismatch: ' + actual);
  }
}
validate();
const adapter = Module.load(__NATIVE_ADAPTER__);
const nativeSend = new NativeFunction(
  adapter.getExportByName('SendTextNow'), 'int',
  ['pointer', 'pointer', 'pointer', 'pointer']);
let pending = null;
let busy = false;
function sendOnce(to, text) {
  const recipient = Memory.allocUtf8String(to);
  const content = Memory.allocUtf8String(text);
  const diagnostics = Memory.alloc(24);
  const result = nativeSend(base, recipient, content, diagnostics);
  if (result !== 1) throw new Error('native adapter failed: ' + result);
  return {
    result: result,
    coro: diagnostics.readPointer().toString(),
    service: diagnostics.add(8).readPointer().toString(),
    context: diagnostics.add(16).readPointer().toString()
  };
}
Interceptor.attach(base.add(offsets.getCoro), {
  onEnter() {
    if (pending === null || busy) return;
    const task = pending;
    pending = null;
    busy = true;
    try { task.resolve(sendOnce(task.to, task.text)); }
    catch (error) { task.reject(error); }
    finally { busy = false; }
  }
});
rpc.exports = {
  probe() {
    return {pid: Process.id, path: module.path, mode: 'api-only-native-coroutine'};
  },
  sendtext(to, text) {
    if (pending !== null || busy) throw new Error('another send is pending');
    return new Promise((resolve, reject) => {
      const task = {to: to, text: text, resolve: resolve, reject: reject};
      pending = task;
      setTimeout(() => {
        if (pending === task) {
          pending = null;
          reject(new Error('coroutine safe-point timeout'));
        }
      }, 15000);
    });
  }
};
"""


def _ensure_native_adapter() -> Path:
    """Build the version-pinned native adapter without opening a console window."""
    if not NATIVE_SOURCE.exists():
        raise WeixinInstrumentationUnavailable(f"微信原生适配源码不存在：{NATIVE_SOURCE}")
    if NATIVE_ADAPTER.exists() and NATIVE_ADAPTER.stat().st_mtime >= NATIVE_SOURCE.stat().st_mtime:
        return NATIVE_ADAPTER
    compiler = shutil.which("gcc")
    if not compiler:
        raise WeixinInstrumentationUnavailable("微信原生适配器缺失，且未找到 gcc 编译器")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = [compiler, "-shared", "-O2", "-Wall", "-Wextra", "-o", str(NATIVE_ADAPTER), str(NATIVE_SOURCE)]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            creationflags=flags,
        )
    except OSError as exc:
        raise WeixinInstrumentationUnavailable(f"微信原生适配器编译失败：{exc}") from exc
    if completed.returncode or not NATIVE_ADAPTER.exists():
        detail = (completed.stderr or completed.stdout or "无编译器输出").strip()
        raise WeixinInstrumentationUnavailable(f"微信原生适配器编译失败：{detail}")
    return NATIVE_ADAPTER


def _script_source(adapter: Path) -> str:
    return SCRIPT_TEMPLATE.replace("__NATIVE_ADAPTER__", json.dumps(str(adapter)))


def _check_binary() -> None:
    if not WEIXIN_DLL.exists():
        raise WeixinInstrumentationUnavailable(f"未找到受支持的微信 DLL：{WEIXIN_DLL}")
    digest = hashlib.sha256(WEIXIN_DLL.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_SHA256:
        raise WeixinInstrumentationUnavailable(f"微信 DLL 版本校验失败：{digest}")


def send_text(
    recipient: str,
    text: str,
    *,
    contact_db: str | Path = DEFAULT_CONTACT_DB,
) -> dict:
    """通过进程内 API 向唯一解析的微信会话发送纯文本。"""
    _check_binary()
    adapter = _ensure_native_adapter()
    recipient_id = resolve_recipient_id(recipient, contact_db)
    try:
        import frida
    except ModuleNotFoundError as exc:
        raise WeixinInstrumentationUnavailable("当前 Python 环境未安装 frida") from exc
    matches = []
    diagnostics = []
    for process in frida.get_local_device().enumerate_processes():
        if process.name.lower() != "weixin.exe":
            continue
        session = None
        try:
            session = frida.attach(process.pid)
            script = session.create_script(_script_source(adapter))
            script.load()
            probe = script.exports_sync.probe()
            if probe:
                matches.append((session, script, probe))
                session = None
        except Exception as exc:
            diagnostics.append(f"pid={process.pid}: {type(exc).__name__}: {exc}")
        finally:
            if session is not None:
                session.detach()
    if len(matches) != 1:
        for session, _, _ in matches:
            session.detach()
        raise WeixinInstrumentationUnavailable(
            f"通过结构校验的微信主进程应为 1 个，实际 {len(matches)} 个；"
            f"诊断：{' | '.join(diagnostics) or '无候选进程'}"
        )
    session, script, probe = matches[0]
    try:
        result = script.exports_sync.sendtext(recipient_id, str(text))
    except Exception as exc:
        raise WeixinInstrumentationError(f"微信进程内 API 发送失败：{exc}") from exc
    finally:
        session.detach()
    if not isinstance(result, dict) or result.get("result") != 1:
        raise WeixinInstrumentationError(f"微信进程内 API 返回异常：{result!r}")
    return {"recipient_id": recipient_id, "probe": probe, "result": result}
