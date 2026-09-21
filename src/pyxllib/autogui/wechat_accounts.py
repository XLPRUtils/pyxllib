"""微信账号隔离的归档入口。账号发现、密钥准备和同步均由提供方负责。"""
from pathlib import Path
import re

from pyxllib.autogui.weixin4_instrumentation import (
    DEFAULT_CONTACT_DB, DEFAULT_SENDER_ACCOUNT_ID, WeixinInstrumentationUnavailable,
    account_id_from_root, resolve_sender,
)
from pyxllib.autogui.wechat_db import WeChatDbStorage


def account_archive_root(account_id: str) -> Path:
    if not re.fullmatch(r"wxid_[A-Za-z0-9]+", account_id):
        raise ValueError("无效的微信账号 ID")
    return DEFAULT_CONTACT_DB.parents[3] / "accounts" / account_id


def get_account_storage(account_id: str) -> WeChatDbStorage:
    """获取账号专属归档；未初始化时拒绝借用其他账号数据。"""
    root = (DEFAULT_CONTACT_DB.parent.parent if account_id == DEFAULT_SENDER_ACCOUNT_ID
            else account_archive_root(account_id) / "wechat_db" / "decrypted" / "db_storage")
    storage = WeChatDbStorage(root)
    owner = storage.status().get("live_account_root")
    if not owner or account_id_from_root(owner) != account_id:
        raise WeixinInstrumentationUnavailable(f"账号 {account_id} 的独立归档尚未初始化")
    return storage


def list_account_archive_roots() -> list[Path]:
    """列出已完成绑定的独立账号归档根目录。"""
    parent = DEFAULT_CONTACT_DB.parents[3] / "accounts"
    if not parent.exists():
        return []
    return [path for path in parent.iterdir() if path.is_dir()
            and re.fullmatch(r"wxid_[A-Za-z0-9]+", path.name)
            and (path / "wechat_db" / "decrypted" / "sync_state.json").exists()]


def prepare_account(account_id: str, *, export_media: bool = False) -> dict:
    """只读访问指定在线账号，建立独立快照；不发送消息或切换登录。"""
    sender = resolve_sender(account_id)
    root = (DEFAULT_CONTACT_DB.parent.parent if account_id == DEFAULT_SENDER_ACCOUNT_ID
            else account_archive_root(account_id) / "wechat_db" / "decrypted" / "db_storage")
    from filelock import FileLock

    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root.parent / "account-initialize.lock"), timeout=120):
        return WeChatDbStorage(root).initialize_from_live(sender, export_media=export_media)


def check_account(account_id: str, recipients: list[str]) -> dict:
    """只读核验发信进程、归档归属和精确收件人；绝不发送消息。"""
    from pyxllib.autogui.weixin4_instrumentation import resolve_recipient_id, preflight

    sender = resolve_sender(account_id)
    storage = get_account_storage(account_id)
    return {"sender": sender, "archive": storage.status(), "layout": preflight(),
            "recipients": {name: resolve_recipient_id(name, storage.root / "contact" / "contact.db") for name in recipients}}
