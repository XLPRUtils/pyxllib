"""订单操作共享逻辑。"""

from __future__ import annotations

import datetime
import logging
from typing import Any, Iterable, Literal, Sequence, cast

logger = logging.getLogger(__name__)


ORDER_SHEET_COLUMNS = [
    "编号",
    "学员名称",
    "微信支付订单号",
    "商户订单号",
    "订单金额",
    "已返款",
    "退款原因",
    "退款额度",
    "执行退款",
]
ORDER_NUMERIC_COLUMNS = ["订单金额", "已返款", "退款额度"]
OrderLookupMode = Literal["hybrid", "db_only", "browser_only"]
DEFAULT_ORDER_LOOKUP_MODE: OrderLookupMode = "hybrid"
ORDER_LOOKUP_MODES = frozenset({"hybrid", "db_only", "browser_only"})
RefundQueryType = Literal["auto", "pay_order", "merchant_order", "refund_id"]
DEFAULT_REFUND_QUERY_TYPE: RefundQueryType = "auto"
REFUND_QUERY_TYPES = frozenset({"auto", "pay_order", "merchant_order", "refund_id"})


class OrderAutomationError(RuntimeError):
    """Raised when order automation input or state is unsafe."""


def _normalize_lookup_mode(value: Any, *, strict: bool = False) -> OrderLookupMode:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return DEFAULT_ORDER_LOOKUP_MODE
    if normalized in ORDER_LOOKUP_MODES:
        return cast(OrderLookupMode, normalized)
    if strict:
        raise OrderAutomationError(f"不支持的订单查单模式：{value}")
    return DEFAULT_ORDER_LOOKUP_MODE


def _normalize_refund_query_type(value: Any, *, strict: bool = False) -> RefundQueryType:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return DEFAULT_REFUND_QUERY_TYPE

    alias_map = {
        "wechat_order_id": "pay_order",
        "wechat_order": "pay_order",
        "transaction_id": "pay_order",
        "flow_order": "pay_order",
        "merchant_order_id": "merchant_order",
        "merchant": "merchant_order",
        "voucher_id": "merchant_order",
        "refund": "refund_id",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized in REFUND_QUERY_TYPES:
        return cast(RefundQueryType, normalized)
    if strict:
        raise OrderAutomationError(f"不支持的退款详情查询类型：{value}")
    return DEFAULT_REFUND_QUERY_TYPE


def _normalize_order_id(order_id: Any) -> str:
    return str(order_id or "").lstrip("`'").strip()


def _coerce_number(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_order_row(row: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(row or {})
    normalized = {col: source.get(col, "") for col in ORDER_SHEET_COLUMNS}
    for col in ORDER_NUMERIC_COLUMNS:
        normalized[col] = _coerce_number(normalized.get(col), 0.0)
    for col in ["编号", "学员名称", "微信支付订单号", "商户订单号", "退款原因", "执行退款"]:
        value = normalized.get(col, "")
        normalized[col] = "" if value is None else str(value)
    return normalized


def _is_blank_order_row(row: dict[str, Any]) -> bool:
    return not (_normalize_order_id(row.get("微信支付订单号")) or _normalize_order_id(row.get("商户订单号")))


def _ensure_weipay(weipay=None, *, weipay_login_users: Sequence[str] | None = None):
    if weipay is not None:
        return weipay

    from .weipay import Weipay

    users = [str(user).strip() for user in (weipay_login_users or []) if str(user).strip()]
    return Weipay(users or None)


def _format_order_record(row: dict[str, Any] | None) -> dict[str, Any]:
    row = dict(row or {})
    dt = row.get("datetime")
    if hasattr(dt, "strftime"):
        order_month = dt.strftime("%Y%m")
    elif isinstance(dt, str) and len(dt) >= 7:
        order_month = dt[:7].replace("-", "")
    else:
        order_month = ""

    return {
        "订单日期": order_month,
        "微信支付订单号": ("`" + str(row["flow_order"])) if row.get("flow_order") else "",
        "商户订单号": str(row["voucher_id"]) if row.get("voucher_id") else "",
        "订单金额": _coerce_number(row.get("money"), ""),
        "已返款": _coerce_number(row.get("refund"), 0.0),
    }


def _generate_zero_o_variants(order_id: str) -> list[str]:
    zero_positions = [i for i, char in enumerate(order_id) if char == "0"]
    if not zero_positions:
        return [order_id]

    variants = []
    for mask in range(2 ** len(zero_positions)):
        chars = list(order_id)
        for index, pos in enumerate(zero_positions):
            if mask & (1 << index):
                chars[pos] = "O"
        variants.append("".join(chars))
    return variants


def _generate_candidate_order_ids(order_id: Any) -> list[str]:
    normalized_order_id = _normalize_order_id(order_id)
    if "-" in normalized_order_id and "0" in normalized_order_id:
        return _generate_zero_o_variants(normalized_order_id)
    return [normalized_order_id]


def _normalize_refund_detail_row(row: dict[str, Any] | None) -> dict[str, Any]:
    source = dict(row or {})
    return {
        "wechat_order_id": _normalize_order_id(source.get("交易单号")),
        "merchant_order_id": _normalize_order_id(source.get("商户单号")),
        "refund_id": _normalize_order_id(source.get("退款单号")),
        "refund_amount": _coerce_number(source.get("退款金额"), 0.0),
        "refund_status": str(source.get("退款状态") or "").strip(),
        "applicant": str(source.get("申请人") or "").strip(),
        "submitted_at": str(source.get("提交时间") or "").strip(),
        "completed_at": str(source.get("退款完成时间") or "").strip(),
    }


def query_order_refund_details(
    order_id: Any,
    *,
    query_type: Any = DEFAULT_REFUND_QUERY_TYPE,
    weipay=None,
    weipay_login_users: Sequence[str] | None = None,
) -> dict[str, Any]:
    normalized_order_id = _normalize_order_id(order_id)
    if not normalized_order_id:
        raise OrderAutomationError("订单号不能为空")

    normalized_query_type = _normalize_refund_query_type(query_type, strict=True)
    candidate_ids = (
        _generate_candidate_order_ids(normalized_order_id)
        if normalized_query_type in {"auto", "merchant_order"}
        else [normalized_order_id]
    )

    weipay = _ensure_weipay(weipay, weipay_login_users=weipay_login_users)
    matched_order_id = normalized_order_id
    raw_rows: list[dict[str, Any]] = []
    last_error = ""
    had_successful_query = False

    for candidate_id in candidate_ids:
        matched_order_id = candidate_id
        try:
            rows = weipay.search_refund_details(candidate_id, query_type=normalized_query_type)
        except Exception as exc:
            last_error = str(exc)
            logger.warning(
                "退款详情查询失败，尝试下一候选订单号：order_id=%s candidate=%s error=%s",
                normalized_order_id,
                candidate_id,
                exc,
            )
            continue
        had_successful_query = True

        raw_rows = [dict(row) for row in rows if isinstance(row, dict)]
        if raw_rows:
            break

    if not raw_rows and last_error and not had_successful_query:
        raise OrderAutomationError(f"退款详情查询失败：{last_error}")

    rows = [_normalize_refund_detail_row(row) for row in raw_rows]
    refund_statuses: list[str] = []
    for row in rows:
        status = row["refund_status"]
        if status and status not in refund_statuses:
            refund_statuses.append(status)

    first_row = rows[0] if rows else {}
    return {
        "summary": {
            "order_id": normalized_order_id,
            "matched_order_id": matched_order_id,
            "query_type": normalized_query_type,
            "row_count": len(rows),
            "refund_amount_total": round(sum(row["refund_amount"] for row in rows), 2),
            "wechat_order_id": str(first_row.get("wechat_order_id") or ""),
            "merchant_order_id": str(first_row.get("merchant_order_id") or ""),
            "refund_statuses": refund_statuses,
        },
        "rows": rows,
    }


def find_order_in_db(order_id: Any, *, kqdb=None) -> dict[str, Any]:
    """Only query the local weipay materialized view; do not open browser automation."""

    normalized_order_id = _normalize_order_id(order_id)
    if not normalized_order_id:
        return {}

    try:
        if kqdb is None:
            from .db import get_kqdb

            kqdb = get_kqdb()

        rows = kqdb.exec2dict(
            "SELECT * FROM weipay_matview WHERE flow_order=%s OR voucher_id=%s",
            [normalized_order_id, normalized_order_id],
        ).fetchall()
    except Exception as exc:
        logger.warning(f"订单数据库查询失败，已跳过数据库通道：order_id={normalized_order_id} error={exc}")
        return {}

    return _format_order_record(rows[0] if rows else {})


def lookup_order(
    order_id: Any,
    *,
    kqdb=None,
    lookup_mode: Any = DEFAULT_ORDER_LOOKUP_MODE,
    use_browser: bool = True,
    weipay=None,
    weipay_login_users: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Query order details from DB first, then optionally fall back to WeChat Pay web automation."""

    normalized_order_id = _normalize_order_id(order_id)
    if not normalized_order_id:
        return {}

    normalized_lookup_mode = _normalize_lookup_mode(lookup_mode, strict=True)
    candidates = _generate_candidate_order_ids(normalized_order_id)
    db_available = True
    allow_db_lookup = normalized_lookup_mode in {"hybrid", "db_only"}
    allow_browser_lookup = bool(use_browser) and normalized_lookup_mode in {"hybrid", "browser_only"}

    if allow_db_lookup and kqdb is None:
        try:
            from .db import get_kqdb

            kqdb = get_kqdb()
        except Exception as exc:
            db_available = False
            logger.warning(f"订单数据库不可用，改走网页查单：order_id={normalized_order_id} error={exc}")

    if allow_db_lookup and db_available and kqdb is not None:
        try:
            for candidate in candidates:
                rows = kqdb.exec2dict(
                    "SELECT * FROM weipay_matview WHERE flow_order=%s OR voucher_id=%s",
                    [candidate, candidate],
                ).fetchall()
                if rows:
                    return _format_order_record(rows[0])
        except Exception as exc:
            logger.warning(f"订单数据库查询失败，改走网页查单：order_id={normalized_order_id} error={exc}")

    if not allow_browser_lookup:
        return {}

    weipay = _ensure_weipay(weipay, weipay_login_users=weipay_login_users)
    last_error = ""
    for candidate in candidates:
        row = weipay.search_refund(candidate)
        if "error" in row:
            last_error = str(row["error"])
            continue

        row["datetime"] = row.get("交易时间")
        row["flow_order"] = row.get("支付单号")
        row["voucher_id"] = row.get("商户订单号")
        row["money"] = row.get("订单金额")
        row["refund"] = row.get("已返款")
        return _format_order_record(row)

    return {"error": last_error} if last_error else {}


def _apply_single_refund(row: dict[str, Any], *, weipay) -> tuple[dict[str, Any], bool]:
    row = dict(row)
    if row["执行退款"]:
        return row, False

    order_amount = _coerce_number(row.get("订单金额"))
    refunded_amount = _coerce_number(row.get("已返款"))
    remaining_amount = max(order_amount - refunded_amount, 0.0)

    explicit_amount = _coerce_number(row.get("退款额度"))
    refund_amount = remaining_amount if not explicit_amount else min(remaining_amount, explicit_amount)
    row["退款额度"] = refund_amount

    if not refund_amount:
        row["执行退款"] = "已退还全部促学金"
        return row, False

    voucher_id = _normalize_order_id(row.get("商户订单号")) or _normalize_order_id(row.get("微信支付订单号"))
    if not voucher_id:
        raise OrderAutomationError("缺少可用于执行退款的订单号")

    weipay.request_single_refund(voucher_id, refund_amount, row.get("退款原因", ""))
    row["执行退款"] = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + " 已退款"
    row["已返款"] = refunded_amount + refund_amount
    return row, True


def _fill_row_from_lookup_result(row: dict[str, Any], order_info: dict[str, Any]) -> dict[str, Any]:
    row["微信支付订单号"] = str(order_info.get("微信支付订单号") or row.get("微信支付订单号") or "")
    row["商户订单号"] = str(order_info.get("商户订单号") or row.get("商户订单号") or "")
    row["订单金额"] = _coerce_number(order_info.get("订单金额"), 0.0)
    row["已返款"] = _coerce_number(order_info.get("已返款"), 0.0)
    return row


def process_order_rows(
    rows: Iterable[dict[str, Any]],
    *,
    need_refund: bool = False,
    weipay=None,
    weipay_login_users: Sequence[str] | None = None,
    kqdb=None,
    lookup_mode: Any = DEFAULT_ORDER_LOOKUP_MODE,
) -> dict[str, Any]:
    """Process order rows without binding to WPS; suitable for codeyun or legacy wrappers."""

    normalized_lookup_mode = _normalize_lookup_mode(lookup_mode, strict=True)
    normalized_rows = [_normalize_order_row(row) for row in rows]
    result_rows: list[dict[str, Any]] = []
    order_index = 0
    refunded_count = 0
    error_count = 0
    skipped_blank_count = 0
    weipay_instance = weipay

    for source_row in normalized_rows:
        if _is_blank_order_row(source_row):
            skipped_blank_count += 1
            continue

        row = dict(source_row)
        if row["已返款"] != "" and "已退款" in str(row["执行退款"]):
            order_index += 1
            row["编号"] = str(order_index)
            result_rows.append(row)
            continue

        lookup_order_id = _normalize_order_id(row["微信支付订单号"] or row["商户订单号"])
        if normalized_lookup_mode == "hybrid":
            order_info = lookup_order(
                lookup_order_id,
                kqdb=kqdb,
                lookup_mode="db_only",
                use_browser=False,
            )
            if not order_info:
                weipay_instance = _ensure_weipay(weipay_instance, weipay_login_users=weipay_login_users)
                order_info = lookup_order(
                    lookup_order_id,
                    kqdb=kqdb,
                    lookup_mode="browser_only",
                    use_browser=True,
                    weipay=weipay_instance,
                    weipay_login_users=weipay_login_users,
                )
        else:
            if normalized_lookup_mode == "browser_only":
                weipay_instance = _ensure_weipay(weipay_instance, weipay_login_users=weipay_login_users)
            order_info = lookup_order(
                lookup_order_id,
                kqdb=kqdb,
                lookup_mode=normalized_lookup_mode,
                weipay=weipay_instance,
                weipay_login_users=weipay_login_users,
            )

        order_index += 1
        row["编号"] = str(order_index)
        if "error" in order_info or not order_info:
            row["订单金额"] = str(order_info.get("error") or "订单不存在")
            row["已返款"] = ""
            error_count += 1
            result_rows.append(row)
            continue

        row = _fill_row_from_lookup_result(row, order_info)

        if need_refund:
            weipay_instance = _ensure_weipay(weipay_instance, weipay_login_users=weipay_login_users)
            row, refunded = _apply_single_refund(row, weipay=weipay_instance)
            if refunded:
                refunded_count += 1

        result_rows.append(row)

    return {
        "action": "refund" if need_refund else "inspect",
        "rows": result_rows,
        "summary": {
            "input_count": len(normalized_rows),
            "processed_count": len(result_rows),
            "refunded_count": refunded_count,
            "error_count": error_count,
            "skipped_blank_count": skipped_blank_count,
        },
    }


def execute_order_action(
    *,
    action: str,
    rows: Iterable[dict[str, Any]],
    weipay=None,
    weipay_login_users: Sequence[str] | None = None,
    kqdb=None,
    lookup_mode: Any = DEFAULT_ORDER_LOOKUP_MODE,
) -> dict[str, Any]:
    normalized_action = str(action or "").strip().lower()
    if normalized_action == "inspect":
        return process_order_rows(
            rows,
            need_refund=False,
            weipay=weipay,
            weipay_login_users=weipay_login_users,
            kqdb=kqdb,
            lookup_mode=lookup_mode,
        )
    if normalized_action == "refund":
        return process_order_rows(
            rows,
            need_refund=True,
            weipay=weipay,
            weipay_login_users=weipay_login_users,
            kqdb=kqdb,
            lookup_mode=lookup_mode,
        )
    raise OrderAutomationError(f"不支持的订单动作：{action}")


def sync_kqbook_order_sheet(
    *,
    need_refund: bool = False,
    kqbook=None,
    file_id: str | None = None,
    script_id: str | None = None,
    weipay=None,
    weipay_login_users: Sequence[str] | None = None,
    lookup_mode: Any = DEFAULT_ORDER_LOOKUP_MODE,
) -> dict[str, Any]:
    """Legacy WPS adapter: read from `订单操作`, process, then write back and release lock."""

    if kqbook is None:
        from .db import KqBook

        if file_id:
            kqbook = KqBook(file_id=file_id, script_id=script_id)
        else:
            kqbook = KqBook()

    try:
        df = kqbook.sql_select("订单操作", ORDER_SHEET_COLUMNS, 4)
        result = execute_order_action(
            action="refund" if need_refund else "inspect",
            rows=df.to_dict(orient="records"),
            weipay=weipay,
            weipay_login_users=weipay_login_users,
            lookup_mode=lookup_mode,
        )
        arr = [[row.get(col, "") for col in ORDER_SHEET_COLUMNS] for row in result["rows"]]
        kqbook.write_arr(arr, "订单操作!A4", 50)
        return result
    finally:
        try:
            kqbook.run_func("releaseMutexLock", "订单操作", "程序状态：")
        except Exception:
            pass
