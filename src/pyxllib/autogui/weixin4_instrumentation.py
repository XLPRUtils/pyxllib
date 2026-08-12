"""微信 4.1.11.55 进程内文本发送适配器。

只负责把已经确定的纯文本送到一个唯一会话。业务收件人、文案和调度仍由
上层考勤代码决定；版本、哈希、运行时结构或联系人解析不唯一时失败关闭。
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


WEIXIN_DLL = Path(r"C:\Program Files\Tencent\Weixin\4.1.11.55\Weixin.dll")
EXPECTED_SHA256 = "AB925B9428239DEF44B252D970C337034D75E66B27EB5529633DC10669FC796A"
DEFAULT_CONTACT_DB = Path(r"D:\home\chenkunze\data\d2605微信逆向\decrypted\db_storage\contact\contact.db")
RECIPIENT_ALIASES = {"文件传输助手": "filehelper"}


class WeixinInstrumentationError(RuntimeError):
    """动态发送不能安全完成。"""


class WeixinInstrumentationUnavailable(WeixinInstrumentationError):
    """当前环境不支持动态发送，可由调用方降级到 GUI。"""


def _repair_legacy_text(value: str) -> str:
    """修复旧解密快照中以 latin1 外观保存的 GBK 文本。"""
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
            "SELECT username, alias, remark, nick_name FROM contact WHERE COALESCE(delete_flag, 0) = 0"
        ).fetchall()
    finally:
        conn.close()
    matches: set[str] = set()
    for username, alias, remark, nickname in rows:
        labels = {_repair_legacy_text(str(value or "")).strip() for value in (username, alias, remark, nickname)}
        if target in labels:
            matches.add(str(username))
    if len(matches) != 1:
        raise WeixinInstrumentationError(f"微信收件人必须唯一匹配：{requested!r}，实际 {len(matches)} 个")
    return matches.pop()


SCRIPT = r"""
function validate(base) {
  const rels = (rva, count) => { const a=[]; for(let i=0;i<count;i++) a.push(base.add(rva+i*8).readPointer().sub(base).toString()); return a; };
  const expected={param1:['0x1741b60','0xaf90','0x1741cb0'],p21:['0x1b373f0','0x1b37500','0x1b37560','0x1b375e0'],p22:['0x1b36ff0','0x1b37030','0x1b37070','0x1b373e0','0x1ae83b0'],p23:['0x1b36da0','0x1b36de0','0x1b36e20','0x1b36f60','0x1ae83b0'],txt:['0xaf60','0xaf70','0xaf40','0xaf90']};
  const actual={param1:rels(0x8834ed8,3),p21:rels(0x88c9b88,4),p22:rels(0x88c9ac8,5),p23:rels(0x88c9a08,5),txt:rels(0x8c38b28,4)};
  if(JSON.stringify(actual)!==JSON.stringify(expected)) throw new Error('runtime vtable mismatch');
}
rpc.exports={
 probe(){const m=Process.findModuleByName('Weixin.dll');if(m===null)return null;validate(m.base);return {pid:Process.id,path:m.path};},
 sendtext(to,text){
  const base=Process.getModuleByName('Weixin.dll').base;validate(base);
  const k=Process.getModuleByName('kernel32.dll');const getHeap=new NativeFunction(k.getExportByName('GetProcessHeap'),'pointer',[]);const heapAlloc=new NativeFunction(k.getExportByName('HeapAlloc'),'pointer',['pointer','uint32','size_t']);const heap=getHeap();
  const alloc=n=>{const p=heapAlloc(heap,8,n*8);if(p.isNull())throw new Error('HeapAlloc failed');return p;};
  const setString=(p,s)=>{const src=Memory.allocUtf8String(s);let n=0;while(src.add(n).readU8()!==0)n++;p.writeByteArray(new Uint8Array(32));p.add(16).writeU64(n);if(n<16){Memory.copy(p,src,n);p.add(n).writeU8(0);p.add(24).writeU64(15);}else{const b=heapAlloc(heap,8,n+1);if(b.isNull())throw new Error('string HeapAlloc failed');Memory.copy(b,src,n+1);p.writePointer(b);p.add(24).writeU64((((n+1+15)&~15)-1));}return n;};
  const mb=alloc(0x768);mb.writePointer(base.add(0x8c38b28));mb.add(8).writeU64(0x200000005);const msg=mb.add(16);new NativeFunction(base.add(0x72e3a0),'pointer',['pointer'])(msg);setString(msg.add(0xb0),to);const len=setString(msg.add(0x708),text);msg.add(0x188).writeU64(len);msg.add(0xd8).writeU64(1);
  const data=alloc(0x20);data.writePointer(msg);data.add(8).writePointer(mb);data.add(16).writePointer(ptr(0));const a1=alloc(0x28);a1.writePointer(base.add(0x8834ed8));a1.add(8).writePointer(data);a1.add(16).writePointer(data.add(16));a1.add(24).writePointer(data.add(16));a1.add(32).writeU64(1);
  const a2=alloc(0xe8),buf=alloc(16),p2=alloc(64),p3=alloc(64),p4=alloc(64);p2.writePointer(base.add(0x88c9b88));p2.add(56).writePointer(p2);p3.writePointer(base.add(0x88c9ac8));p3.add(56).writePointer(p3);p4.writePointer(base.add(0x88c9a08));p4.add(56).writePointer(p4);new NativeFunction(base.add(0xe980),'int64',['pointer','pointer','pointer','pointer','pointer','uint64'])(a2,p2,p3,p4,buf,base.add(0xa5dcdc0).readU64());const ret=new NativeFunction(base.add(0x1741cb0),'int64',['pointer','pointer'])(a1,a2).toString();return {ret:ret,messageLength:len};
 }
};
"""


def _check_binary() -> None:
    if not WEIXIN_DLL.exists():
        raise WeixinInstrumentationUnavailable(f"未找到受支持的微信 DLL：{WEIXIN_DLL}")
    digest = hashlib.sha256(WEIXIN_DLL.read_bytes()).hexdigest().upper()
    if digest != EXPECTED_SHA256:
        raise WeixinInstrumentationUnavailable(f"微信 DLL 版本校验失败：{digest}")


def send_text(recipient: str, text: str, *, contact_db: str | Path = DEFAULT_CONTACT_DB) -> dict:
    """向唯一解析的微信会话发送纯文本，成功后返回底层结果。"""
    _check_binary()
    recipient_id = resolve_recipient_id(recipient, contact_db)
    try:
        import frida
    except ModuleNotFoundError as exc:
        raise WeixinInstrumentationUnavailable("当前 Python 环境未安装 frida") from exc
    matches = []
    for process in frida.get_local_device().enumerate_processes():
        if process.name.lower() != "weixin.exe":
            continue
        session = None
        try:
            session = frida.attach(process.pid)
            script = session.create_script(SCRIPT)
            script.load()
            probe = script.exports_sync.probe()
            if probe:
                matches.append((session, script, probe))
                session = None
        except Exception:
            pass
        finally:
            if session is not None:
                session.detach()
    if len(matches) != 1:
        for session, _, _ in matches:
            session.detach()
        raise WeixinInstrumentationUnavailable(f"通过结构校验的微信主进程应为 1 个，实际 {len(matches)} 个")
    session, script, probe = matches[0]
    try:
        result = script.exports_sync.sendtext(recipient_id, str(text))
    except Exception as exc:
        raise WeixinInstrumentationError(f"微信动态文本发送失败：{exc}") from exc
    finally:
        session.detach()
    if str(result.get("ret")) != "1":
        raise WeixinInstrumentationError(f"微信动态文本发送返回异常：{result!r}")
    return {"recipient_id": recipient_id, "probe": probe, "result": result}
