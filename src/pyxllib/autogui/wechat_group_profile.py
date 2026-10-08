"""Account-scoped group nickname API; native operation with archive verification.

Weixin 4.1.15.50: ChatroomManager wrapper 0x10543b0 gets the active
session's oplog service, then serializes ModChatRoomMemberDisplayName
(room, member, nickname), opcode 48. The UI's 0x7700a0 callback supplies
the logged-in member ID. This adapter always supplies the verified account
ID, so it cannot modify another member or the group's title. No GUI.
Offsets are valid only for the pinned DLL hash; upgrades fail closed.
Dispatch runs at the existing get_coro safe point: service lookup depends on
the client thread's session context and cannot run on Frida's RPC thread.
"""
from __future__ import annotations

import hashlib
import re
import time

from filelock import FileLock

from pyxllib.autogui.wechat_accounts import get_account_storage
from pyxllib.autogui.weixin4_instrumentation import (
    WeixinInstrumentationError, WeixinInstrumentationUnavailable,
    resolve_sender, running_weixin_dll,
    load_layout,
)

DLL_SHA256 = "0cf6379405432385a1c442766443450f26a7689f45e53c1b4ad2822061f3113b"
_SCRIPT = r"""
const module = Process.getModuleByName('Weixin.dll');
const base = module.base;
const safePoint = __SAFE_POINT__;
function check(offset, hex) {
    const bytes = new Uint8Array(base.add(offset).readByteArray(hex.length / 2));
    const actual = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
    if (actual !== hex) throw new Error('Nickname API code mismatch at ' + offset.toString(16));
}
check(0x10543b0, '555657534883ec58488d6c245048c74500feffffff4c89ce4c89c74889d3488d4df0');
check(0x19aeb3d, '488d55a04889f141b830000000e8b197ffff');
check(safePoint, '564883ec404889ce');
const modify = new NativeFunction(base.add(0x10543b0), 'void', ['pointer', 'pointer', 'pointer', 'pointer'], 'win64');
function string(value) {
    const bytes = Memory.allocUtf8String(value);
    const length = unescape(encodeURIComponent(value)).length;
    const object = Memory.alloc(32);
    object.writeByteArray(new Uint8Array(32));
    if (length <= 15) object.writeByteArray(bytes.readByteArray(length + 1));
    else object.writePointer(bytes);
    object.add(16).writeU64(length);
    object.add(24).writeU64(length <= 15 ? 15 : length);
    return {object, bytes};
}
let pending = null, busy = false;
Interceptor.attach(base.add(safePoint), {
    onEnter() {
        if (pending === null || busy) return;
        const task = pending;
        pending = null;
        busy = true;
        try {
            const r = string(task.room), m = string(task.member), n = string(task.nickname);
            modify(ptr(0), r.object, m.object, n.object);
            task.resolve({dispatched: true});
        } catch (error) { task.reject(error); }
        finally { busy = false; }
    }
});
rpc.exports = {
    path() { return module.path; },
    setNickname(room, member, nickname) {
        if (pending !== null || busy) throw new Error('Nickname operation already pending');
        return new Promise((resolve, reject) => {
            const task = {room, member, nickname, resolve, reject};
            pending = task;
            setTimeout(() => {
                if (pending === task) {
                    pending = null;
                    reject(new Error('Nickname coroutine safe-point timeout; not dispatched'));
                }
            }, 15000);
        });
    }
};
"""


def get_group_nickname(account_id: str, chat_id: str, *, refresh: bool = True) -> dict:
    """Read this account's own group nickname; missing member fails visibly."""
    storage = get_account_storage(account_id)
    if refresh:
        storage.sync_from_live(export_media=False)
    matches = [m for m in storage.group_members(chat_id) if m["username"] == account_id]
    if len(matches) != 1:
        raise WeixinInstrumentationUnavailable("目标群没有唯一的当前账号成员")
    return {"account_id": account_id, "chat_id": chat_id, "nickname": matches[0]["display_name"]}


def set_group_nickname(account_id: str, chat_id: str, nickname: str, *, timeout_seconds: float = 30) -> dict:
    """Set only this online account's nickname in one exact group, then verify.

    Idempotent when already equal. Dispatch once; an unverified result raises
    with an explicit unknown outcome. Call get_group_nickname before retrying.
    The timeout bounds archive polling after dispatch, not account sync locks.
    """
    if not re.fullmatch(r"wxid_[A-Za-z0-9]+", account_id):
        raise ValueError("必须使用精确的微信账号 ID")
    if not re.fullmatch(r"[0-9]+@chatroom", chat_id):
        raise ValueError("必须使用精确的微信群 ID")
    if not isinstance(nickname, str) or not nickname.strip() or "\0" in nickname or len(nickname) > 32:
        raise ValueError("群昵称必须为 1–32 字且不能含 NUL")
    storage = get_account_storage(account_id)
    with FileLock(str(storage.root.parent / "group-profile.lock"), timeout=60):
        sender = resolve_sender(account_id)
        previous = get_group_nickname(account_id, chat_id)["nickname"]
        if previous == nickname:
            return {"account_id": account_id, "chat_id": chat_id, "nickname": nickname,
                    "previous_nickname": previous, "changed": False, "verified": True}
        dll = running_weixin_dll(account_id)
        if hashlib.sha256(dll.read_bytes()).hexdigest() != DLL_SHA256:
            raise WeixinInstrumentationUnavailable("当前微信版本未验证群昵称接口，未执行修改")
        layout = load_layout(dll)
        import frida
        session = frida.attach(sender["pid"])
        try:
            script = session.create_script(_SCRIPT.replace("__SAFE_POINT__", str(layout.offsets["get_coro"])))
            script.load()
            from pathlib import Path
            if Path(script.exports_sync.path()).resolve() != dll.resolve() or resolve_sender(account_id) != sender:
                raise WeixinInstrumentationUnavailable("昵称修改前账号进程或 DLL 已变化，未执行修改")
            script.exports_sync.set_nickname(chat_id, account_id, nickname)
        except Exception as exc:
            raise WeixinInstrumentationError(f"群昵称原生调用失败，请先回读确认结果：{exc}") from exc
        finally:
            session.detach()
        deadline = time.monotonic() + max(1, min(float(timeout_seconds), 60))
        while True:
            current = get_group_nickname(account_id, chat_id)
            if current["nickname"] == nickname:
                return {**current, "previous_nickname": previous, "changed": True, "verified": True}
            if time.monotonic() >= deadline:
                raise WeixinInstrumentationError("修改请求已提交，但群昵称回读未确认；请先回读再重试")
            time.sleep(1)
