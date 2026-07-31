from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from urllib.parse import urlparse


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_ROOT = MODULE_DIR / "output"
DEFAULT_RUNS_ROOT = DEFAULT_ARTIFACT_ROOT / "runs"


def extract_doc_token(url: str) -> str:
    path = urlparse(url).path
    match = re.search(r"/l/([^/?#]+)", path)
    if not match:
        raise ValueError(f"无法从 url 提取文档 token：{url}")
    return match.group(1)


def sanitize_label(text: str | None, *, max_length: int = 48) -> str:
    if not text:
        return ""

    sanitized = re.sub(r"\s+", "_", text.strip())
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("._ ")
    if not sanitized:
        return ""

    return sanitized[:max_length].rstrip("._ ")


def build_requested_actions(
    *,
    create_copy: bool = False,
    open_now: bool = False,
    rename_title: str | None = None,
    share_requested: bool = False,
) -> list[str]:
    actions: list[str] = []
    if create_copy:
        actions.append("copy")
    if open_now:
        actions.append("open")
    if rename_title:
        actions.append("rename")
    if share_requested:
        actions.append("share")
    if not actions:
        actions.append("probe")
    return actions


def build_run_dir_name(
    url: str,
    *,
    actions: list[str],
    label: str | None = None,
    now: datetime | None = None,
) -> str:
    now = now or datetime.now()
    parts = [now.strftime("%Y%m%d_%H%M%S"), "-".join(actions)]

    sanitized_label = sanitize_label(label)
    if sanitized_label:
        parts.append(sanitized_label)

    parts.append(extract_doc_token(url))
    return "__".join(parts)


def resolve_save_dir(
    *,
    explicit_save_dir: str | Path | None,
    save_root: str | Path | None,
    url: str,
    actions: list[str],
    label: str | None = None,
    now: datetime | None = None,
) -> Path:
    if explicit_save_dir:
        return Path(explicit_save_dir)

    root = Path(save_root) if save_root else DEFAULT_RUNS_ROOT
    return root / build_run_dir_name(
        url,
        actions=actions,
        label=label,
        now=now,
    )
