"""微信进程内纯文本发送 API；没有 GUI 降级路径。

版本兼容采用分层解析（对齐凡修接口层 Runtime 目标分层解析与恢复）：

- L0 版本快路径：DLL ``sha256`` 命中 :mod:`weixin4_offsets` 的不可变布局表。
- L1/L2 有界回退：未知版本用 ``getService`` 唯一签名 + 发送链拓扑自动重建偏移。
- L4 失败关闭：解析歧义或签名不符时抛出 :class:`WeixinInstrumentationUnavailable`，
  调用方必须失败关闭，禁止降级 GUI。

原生适配器 :file:`native/weixin_4_1_13_send.c` 不再硬编码偏移，函数地址与结构偏移
全部由解析结果在运行时传入，因此微信更新通常不需要重新编译适配器。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

from pyxllib.autogui.weixin4_offsets import RebindError, WeixinLayout, resolve

WEIXIN_DLL = Path(r"C:\Program Files\Tencent\Weixin\4.1.13.65\Weixin.dll")
DEFAULT_CONTACT_DB = Path(r"C:\home\chenkunze\data\d2605微信逆向\decrypted\db_storage\contact\contact.db")
RECIPIENT_ALIASES = {"文件传输助手": "filehelper"}
NATIVE_SOURCE = Path(__file__).with_name("native") / "weixin_send.c"
NATIVE_ADAPTER = NATIVE_SOURCE.with_suffix(".dll")

_LAYOUT_CACHE: dict[tuple[str, int, int], tuple[str, WeixinLayout]] = {}


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


def load_layout(path: str | Path = WEIXIN_DLL, *, refresh: bool = False) -> WeixinLayout:
    """按 ``(路径, mtime, 大小)`` 缓存布局解析结果；DLL 变化即重新解析。"""
    dll = Path(path)
    if not dll.exists():
        raise WeixinInstrumentationUnavailable(f"未找到微信 DLL：{dll}")
    stat = dll.stat()
    key = (str(dll.resolve()), stat.st_mtime_ns, stat.st_size)
    if not refresh and key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key][1]
    data = dll.read_bytes()
    digest = hashlib.sha256(data).hexdigest().upper()
    try:
        layout = resolve(data, digest)
    except RebindError as exc:
        raise WeixinInstrumentationUnavailable(f"微信 {digest[:12]} 发送链无法安全解析：{exc}") from exc
    _LAYOUT_CACHE.clear()
    _LAYOUT_CACHE[key] = (digest, layout)
    return layout


def preflight(path: str | Path = WEIXIN_DLL) -> dict:
    """启动自检：解析当前微信布局，返回可诊断结果而不发送任何消息。"""
    try:
        layout = load_layout(path)
    except WeixinInstrumentationError as exc:
        return {"ok": False, "error": str(exc), "dll": str(path)}
    return {
        "ok": True,
        "dll": str(path),
        "sha256": layout.sha256,
        "version": layout.version,
        "source": layout.source,
        "offsets": {name: hex(value) for name, value in layout.offsets.items()},
        "content_offset": hex(layout.struct["content"]),
        "evidence": layout.evidence,
    }


SCRIPT_TEMPLATE = r"""
const module = Process.getModuleByName('Weixin.dll');
const base = module.base;
const layout = __LAYOUT__;
const expected = {
  get_coro: '564883ec404889ce',
  get_service: '4889d0488b51304885d27414f0ff4208488b5130488b492848890848895008c331d2488b492848890848895008c3',
  get_context: '5556574881ec90000000488d',
  do_send: '554157415641554154565753',
  send_entry: '5556574881ecd0000000488d',
  message_ctor: '5556574883ec50488d6c2450'
};
function hex(value) {
  return Array.from(new Uint8Array(value)).map(v => v.toString(16).padStart(2, '0')).join('');
}
function validate() {
  for (const [name, signature] of Object.entries(expected)) {
    const rva = layout.offsets[name];
    if (rva === undefined) continue;
    const actual = hex(base.add(rva).readByteArray(signature.length / 2));
    if (actual !== signature) throw new Error(name + ' runtime signature mismatch: ' + actual);
  }
}
validate();
const adapter = Module.load(__NATIVE_ADAPTER__);
const nativeSend = new NativeFunction(
  adapter.getExportByName('SendTextNow'), 'int',
  ['pointer', 'pointer', 'pointer', 'pointer']);
const offsets = layout.offsets;
const struct = layout.struct;
const params = Memory.alloc(80);
const functions = [offsets.get_coro, offsets.get_service, offsets.get_context, offsets.message_ctor, offsets.do_send];
for (let i = 0; i < functions.length; i++) params.add(i * 8).writePointer(base.add(functions[i]));
const fields = [
  struct.object_size, struct.content, struct.wxid, struct.length, struct.flag,
  struct.kind, struct.subtype, struct.flag_value, struct.kind_value, struct.subtype_value
];
for (let i = 0; i < fields.length; i++) params.add(40 + i * 4).writeU32(fields[i]);
let pending = null;
let busy = false;
function sendOnce(to, text) {
  const recipient = Memory.allocUtf8String(to);
  const content = Memory.allocUtf8String(text);
  const diagnostics = Memory.alloc(24);
  const result = nativeSend(params, recipient, content, diagnostics);
  if (result !== 1) throw new Error('native adapter failed: ' + result);
  return {
    result: result,
    coro: diagnostics.readPointer().toString(),
    service: diagnostics.add(8).readPointer().toString(),
    context: diagnostics.add(16).readPointer().toString()
  };
}
Interceptor.attach(base.add(offsets.get_coro), {
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
    return {pid: Process.id, path: module.path, mode: 'api-only-native-coroutine', layout: layout.source};
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
    """Build the version-agnostic native adapter without opening a console window."""
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


def _script_source(adapter: Path, layout: WeixinLayout | None = None) -> str:
    if layout is None:
        layout = WeixinLayout(
            sha256="", version=None, offsets={}, struct={}, source="placeholder"
        )
    payload = {
        "offsets": layout.offsets,
        "struct": layout.struct,
        "source": layout.source,
        "version": layout.version,
    }
    return (
        SCRIPT_TEMPLATE
        .replace("__NATIVE_ADAPTER__", json.dumps(str(adapter)))
        .replace("__LAYOUT__", json.dumps(payload))
    )


def send_text(
    recipient: str,
    text: str,
    *,
    contact_db: str | Path = DEFAULT_CONTACT_DB,
) -> dict:
    """通过进程内 API 向唯一解析的微信会话发送纯文本。"""
    layout = load_layout()
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
        script = None
        script_messages: list[str] = []
        try:
            session = frida.attach(process.pid)
            script = session.create_script(_script_source(adapter, layout))
            script.on(
                "message",
                lambda message, _data, bucket=script_messages: bucket.append(str(message)),
            )
            script.load()
            probe = script.exports_sync.probe()
            if probe:
                matches.append((session, script, probe))
                session = None
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if script_messages:
                detail = f"{detail}；脚本错误：{script_messages[-1]}"
            diagnostics.append(f"pid={process.pid}: {detail}")
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
