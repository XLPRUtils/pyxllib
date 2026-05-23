#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
游戏资源文件的轻量解析工具。

本模块只放通用格式能力，不绑定具体游戏：
- Unity AssetBundle：识别 UnityFS/UnityWeb/UnityRaw，剥离前置封装，读取对象摘要，导出 Texture2D。
- Wwise SoundBank：解析 BNK chunk，读取 DIDX 索引，提取内嵌 WEM。

Unity 解析依赖 UnityPy，函数内部按需导入，避免让 pyxllib 基础安装强依赖游戏资源工具链。
"""

import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


UNITY_BUNDLE_MAGICS = (b"UnityFS", b"UnityWeb", b"UnityRaw")
WWISE_BANK_MAGIC = b"BKHD"
_SAFE_FILENAME_RE = re.compile(r"[^0-9A-Za-z_.-]+")


@dataclass
class UnityObjectInfo:
    type_name: str
    path_id: int
    name: str = ""
    read_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnityBundleSummary:
    path: str
    size: int
    magic: str
    offset: int
    object_counts: Dict[str, int]
    objects: List[UnityObjectInfo]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["objects"] = [item.to_dict() for item in self.objects]
        return data


@dataclass
class UnityTextureExport:
    source_path: str
    output_path: str
    path_id: int
    name: str
    width: int
    height: int
    texture_format: Any

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnityTextAssetExport:
    source_path: str
    output_path: str
    path_id: int
    name: str
    byte_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WwiseChunkInfo:
    fourcc: str
    offset: int
    size: int
    data_offset: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WwiseWemEntry:
    wem_id: int
    offset: int
    size: int
    output_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _read_source_bytes(source: Any) -> Tuple[bytes, str]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source), ""
    path = Path(source)
    return path.read_bytes(), str(path)


def _safe_filename(value: str, fallback: str = "asset", max_length: int = 140) -> str:
    name = _SAFE_FILENAME_RE.sub("_", str(value or "").strip()).strip("._")
    if not name:
        name = fallback
    return name[:max_length]


def find_first_magic(data: bytes, magics: Iterable[bytes]) -> Tuple[Optional[bytes], int]:
    """在 data 中查找最先出现的 magic。"""
    found_magic = None
    found_offset = -1
    for magic in magics:
        offset = data.find(magic)
        if offset >= 0 and (found_offset < 0 or offset < found_offset):
            found_magic = magic
            found_offset = offset
    return found_magic, found_offset


def locate_unity_bundle_offset(source: Any, scan_bytes: int = 1024 * 1024) -> Tuple[Optional[bytes], int]:
    """返回 Unity AssetBundle magic 和偏移。

    一些手游热更新包会在 UnityFS 前面加自定义头或填充字节，直接交给 UnityPy 会失败。
    这个函数只扫描前 scan_bytes 字节，适合快速识别候选文件。
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source[:scan_bytes])
    else:
        with Path(source).open("rb") as f:
            data = f.read(scan_bytes)
    return find_first_magic(data, UNITY_BUNDLE_MAGICS)


def strip_unity_bundle_prefix(data: bytes) -> Tuple[bytes, Optional[bytes], int]:
    """剥离 Unity magic 前面的封装头，返回可交给 UnityPy 的数据。"""
    magic, offset = find_first_magic(data, UNITY_BUNDLE_MAGICS)
    if offset > 0:
        return data[offset:], magic, offset
    return data, magic, offset


def is_probable_unity_bundle(source: Any, scan_bytes: int = 1024 * 1024) -> bool:
    return locate_unity_bundle_offset(source, scan_bytes=scan_bytes)[1] >= 0


def _import_unitypy():
    try:
        import UnityPy  # noqa: N812
    except ImportError as exc:
        raise RuntimeError("读取 Unity 资源需要安装 UnityPy，例如：uv add --project backend UnityPy") from exc
    return UnityPy


def _patch_unitypy_version_parser() -> None:
    """兼容少数资源包里带换行/附加字段的 Unity 版本字符串。"""
    try:
        from UnityPy.helpers.UnityVersion import UnityVersion
    except ImportError:
        return
    if getattr(UnityVersion, "_pyxllib_version_parser_patched", False):
        return
    original_from_str = UnityVersion.from_str

    def cleaned_from_str(version):
        text = str(version or "").strip().split()[0]
        return original_from_str(text)

    UnityVersion.from_str = staticmethod(cleaned_from_str)
    UnityVersion._pyxllib_version_parser_patched = True


def load_unity_environment(source: Any):
    """加载 UnityPy Environment，自动剥离 UnityFS 前置封装。"""
    UnityPy = _import_unitypy()
    _patch_unitypy_version_parser()
    data, _path = _read_source_bytes(source)
    bundle_data, _magic, _offset = strip_unity_bundle_prefix(data)
    if _offset < 0:
        raise ValueError("未找到 Unity AssetBundle magic")
    return UnityPy.load(bundle_data)


def summarize_unity_bundle(source: Any, max_objects: int = 100) -> UnityBundleSummary:
    """读取 Unity AssetBundle 对象摘要。"""
    data, path = _read_source_bytes(source)
    size = len(data)
    bundle_data, magic, offset = strip_unity_bundle_prefix(data)
    magic_text = magic.decode("ascii") if magic else ""
    if offset < 0:
        return UnityBundleSummary(path=path, size=size, magic="", offset=-1, object_counts={}, objects=[], error="未找到 Unity AssetBundle magic")

    try:
        UnityPy = _import_unitypy()
        _patch_unitypy_version_parser()
        env = UnityPy.load(bundle_data)
        object_counts = {}
        objects = []
        for obj in env.objects:
            type_name = obj.type.name
            object_counts[type_name] = object_counts.get(type_name, 0) + 1
            if len(objects) >= max_objects:
                continue
            name = ""
            read_error = ""
            try:
                obj_data = obj.read()
                name = getattr(obj_data, "name", "") or getattr(obj_data, "m_Name", "") or ""
            except Exception as exc:  # pragma: no cover - depends on concrete Unity class support.
                read_error = "{}: {}".format(type(exc).__name__, exc)
            objects.append(UnityObjectInfo(type_name=type_name, path_id=int(obj.path_id), name=str(name), read_error=read_error))
        return UnityBundleSummary(path=path, size=size, magic=magic_text, offset=offset, object_counts=object_counts, objects=objects)
    except Exception as exc:
        return UnityBundleSummary(
            path=path,
            size=size,
            magic=magic_text,
            offset=offset,
            object_counts={},
            objects=[],
            error="{}: {}".format(type(exc).__name__, exc),
        )


def export_unity_textures(
    source: Any,
    output_dir: Any,
    max_textures: Optional[int] = None,
    name_prefix: str = "",
) -> List[UnityTextureExport]:
    """导出 Unity Texture2D 为 PNG。

    返回实际导出的文件信息。无法解码的贴图会抛出底层异常，业务层可决定是否捕获。
    """
    source_path = str(source) if not isinstance(source, (bytes, bytearray, memoryview)) else ""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_unity_environment(source)

    exports = []
    for obj in env.objects:
        if obj.type.name != "Texture2D":
            continue
        if max_textures is not None and len(exports) >= max_textures:
            break
        obj_data = obj.read()
        raw_name = getattr(obj_data, "name", "") or getattr(obj_data, "m_Name", "") or ""
        safe_name = _safe_filename(raw_name, fallback="texture_{}".format(obj.path_id))
        prefix = _safe_filename(name_prefix, fallback="", max_length=80) if name_prefix else ""
        stem = "{}__{}__{}".format(prefix, safe_name, obj.path_id) if prefix else "{}__{}".format(safe_name, obj.path_id)
        output_path = out_dir / "{}.png".format(stem)
        image = obj_data.image
        image.save(output_path)
        exports.append(
            UnityTextureExport(
                source_path=source_path,
                output_path=str(output_path),
                path_id=int(obj.path_id),
                name=str(raw_name),
                width=int(image.size[0]),
                height=int(image.size[1]),
                texture_format=getattr(obj_data, "m_TextureFormat", None),
            )
        )
    return exports


def _read_unity_text_asset_bytes(obj_data: Any) -> bytes:
    raw = getattr(obj_data, "script", None)
    if raw is None:
        raw = getattr(obj_data, "m_Script", None)
    if raw is None:
        return b""
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, str):
        return raw.encode("utf-8")
    try:
        return bytes(raw)
    except TypeError:
        return str(raw).encode("utf-8")


def _unique_output_path(output_dir: Path, filename: str, path_id: int) -> Path:
    path = output_dir / filename
    if not path.exists():
        return path
    stem = path.stem or "asset"
    suffix = path.suffix
    return output_dir / "{}__{}{}".format(stem, path_id, suffix)


def export_unity_text_assets(
    source: Any,
    output_dir: Any,
    max_assets: Optional[int] = None,
    name_prefix: str = "",
) -> List[UnityTextAssetExport]:
    """导出 Unity TextAsset 的原始脚本字节。"""
    source_path = str(source) if not isinstance(source, (bytes, bytearray, memoryview)) else ""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = load_unity_environment(source)

    exports = []
    for obj in env.objects:
        if obj.type.name != "TextAsset":
            continue
        if max_assets is not None and len(exports) >= max_assets:
            break
        obj_data = obj.read()
        raw_name = getattr(obj_data, "name", "") or getattr(obj_data, "m_Name", "") or ""
        safe_name = _safe_filename(raw_name, fallback="text_asset_{}".format(obj.path_id))
        prefix = _safe_filename(name_prefix, fallback="", max_length=80) if name_prefix else ""
        filename = "{}__{}".format(prefix, safe_name) if prefix else safe_name
        output_path = _unique_output_path(out_dir, filename, int(obj.path_id))
        payload = _read_unity_text_asset_bytes(obj_data)
        output_path.write_bytes(payload)
        exports.append(
            UnityTextAssetExport(
                source_path=source_path,
                output_path=str(output_path),
                path_id=int(obj.path_id),
                name=str(raw_name),
                byte_size=len(payload),
            )
        )
    return exports


def parse_wwise_bnk_chunks(source: Any) -> List[WwiseChunkInfo]:
    """解析 Wwise BNK chunk 表。"""
    data, _path = _read_source_bytes(source)
    chunks = []
    offset = 0
    while offset + 8 <= len(data):
        fourcc_raw = data[offset:offset + 4]
        try:
            fourcc = fourcc_raw.decode("ascii")
        except UnicodeDecodeError:
            break
        size = struct.unpack_from("<I", data, offset + 4)[0]
        data_offset = offset + 8
        end = data_offset + size
        if end > len(data):
            break
        chunks.append(WwiseChunkInfo(fourcc=fourcc, offset=offset, size=size, data_offset=data_offset))
        offset = end
    return chunks


def parse_wwise_didx_entries(source: Any) -> List[WwiseWemEntry]:
    """读取 Wwise BNK 的 DIDX 媒体索引。"""
    data, _path = _read_source_bytes(source)
    chunks = parse_wwise_bnk_chunks(data)
    didx = next((chunk for chunk in chunks if chunk.fourcc == "DIDX"), None)
    if not didx:
        return []

    entries = []
    cursor = didx.data_offset
    end = didx.data_offset + didx.size
    while cursor + 12 <= end:
        wem_id, offset, size = struct.unpack_from("<III", data, cursor)
        entries.append(WwiseWemEntry(wem_id=int(wem_id), offset=int(offset), size=int(size)))
        cursor += 12
    return entries


def extract_wwise_wem_entries(source: Any, output_dir: Any, max_entries: Optional[int] = None) -> List[WwiseWemEntry]:
    """从 Wwise BNK 的 DATA chunk 中提取 WEM 文件。"""
    data, _path = _read_source_bytes(source)
    chunks = parse_wwise_bnk_chunks(data)
    data_chunk = next((chunk for chunk in chunks if chunk.fourcc == "DATA"), None)
    if not data_chunk:
        return []

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = parse_wwise_didx_entries(data)
    extracted = []
    for entry in entries:
        if max_entries is not None and len(extracted) >= max_entries:
            break
        start = data_chunk.data_offset + entry.offset
        end = start + entry.size
        if start < data_chunk.data_offset or end > data_chunk.data_offset + data_chunk.size:
            continue
        output_path = out_dir / "{}.wem".format(entry.wem_id)
        output_path.write_bytes(data[start:end])
        extracted_entry = WwiseWemEntry(wem_id=entry.wem_id, offset=entry.offset, size=entry.size, output_path=str(output_path))
        extracted.append(extracted_entry)
    return extracted
