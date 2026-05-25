#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers for length-prefixed binary packet streams and Lua-generated schemas."""

from __future__ import annotations

import re
import struct
import subprocess
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class PacketDecodeError(ValueError):
    pass


class VarintBinaryReader:
    """Reader for the Lusuo-style compressed binary primitives seen in Unity Lua games."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def left(self) -> int:
        return len(self.data) - self.pos

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise PacketDecodeError(f"need {n} bytes at {self.pos}, left={self.left()}")
        value = self.data[self.pos : self.pos + n]
        self.pos += n
        return value

    def read_byte(self) -> int:
        return self.take(1)[0]

    def read_int(self) -> int:
        return self.read_zigzag_varint()

    def read_long(self) -> int:
        return self.read_zigzag_varint()

    def read_zigzag_varint(self) -> int:
        value = 0
        shift = 0
        for _ in range(10):
            byte = self.read_byte()
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return (value >> 1) ^ -(value & 1)
            shift += 7
        raise PacketDecodeError("varint too long")

    def read_bool(self) -> bool:
        return self.read_byte() != 0

    def read_float(self) -> float:
        return struct.unpack(">f", self.take(4))[0]

    def read_double(self) -> float:
        return struct.unpack(">d", self.take(8))[0]

    def read_string(self) -> Optional[str]:
        length = self.read_int()
        if length < 0:
            return None
        return self.take(length).decode("utf-8", errors="replace")


@dataclass
class PacketFrame:
    offset: int
    length: int
    body: bytes


def iter_u32be_length_prefixed_frames(data: bytes) -> Iterator[PacketFrame]:
    pos = 0
    while pos + 4 <= len(data):
        offset = pos
        length = int.from_bytes(data[pos : pos + 4], "big")
        pos += 4
        if pos + length > len(data):
            raise PacketDecodeError(f"incomplete frame at {offset}: length={length}, left={len(data) - pos}")
        body = data[pos : pos + length]
        pos += length
        yield PacketFrame(offset, length, body)


def maybe_zlib_decompress(data: bytes) -> Tuple[bytes, bool]:
    if len(data) >= 2 and data[0] == 0x78:
        try:
            return zlib.decompress(data), True
        except zlib.error:
            return data, False
    return data, False


def extract_tcp_stream_payloads_with_tshark(
    pcap: Union[str, Path],
    stream: int,
    *,
    server_host: str,
    tshark: Union[str, Path] = r"C:\Program Files\Wireshark\tshark.exe",
) -> Tuple[bytes, bytes]:
    """Return concatenated client/server TCP payload bytes for a tshark stream."""

    cmd = [
        str(tshark),
        "-r",
        str(pcap),
        "-Y",
        f"tcp.stream == {int(stream)} && tcp.len > 0",
        "-T",
        "fields",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "tcp.payload",
    ]
    text = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    client = bytearray()
    server = bytearray()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or not parts[2]:
            continue
        payload = bytes.fromhex(parts[2].replace(":", ""))
        if parts[1] == server_host:
            client.extend(payload)
        elif parts[0] == server_host:
            server.extend(payload)
    return bytes(client), bytes(server)


def list_tcp_streams_with_tshark(
    pcap: Union[str, Path],
    *,
    host: str = "",
    tshark: Union[str, Path] = r"C:\Program Files\Wireshark\tshark.exe",
) -> List[Dict[str, Any]]:
    """Return TCP stream ids and rough payload byte counts from a capture."""

    display_filter = "tcp.len > 0"
    if host:
        display_filter = f"ip.addr == {host} && {display_filter}"
    cmd = [
        str(tshark),
        "-r",
        str(pcap),
        "-Y",
        display_filter,
        "-T",
        "fields",
        "-e",
        "tcp.stream",
        "-e",
        "tcp.len",
    ]
    text = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    streams: Dict[int, Dict[str, Any]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        try:
            stream = int(parts[0])
            length = int(parts[1] or 0) if len(parts) > 1 else 0
        except ValueError:
            continue
        item = streams.setdefault(stream, {"stream": stream, "packets": 0, "payload_bytes": 0})
        item["packets"] += 1
        item["payload_bytes"] += length
    return sorted(streams.values(), key=lambda item: (item["payload_bytes"], item["packets"]), reverse=True)


@dataclass
class LuaPacketClassInfo:
    name: str
    path: Path
    packet_id: Optional[int]
    parent: Optional[str]
    ops: List[Tuple[str, str, Optional[str]]]


class LuaPacketSchemaIndex:
    """Index generated Lua packet/VO classes and decode fields in their reading order."""

    def __init__(self, text_assets: Union[str, Path], vo_url: Optional[Union[str, Path]] = None):
        self.text_assets = Path(text_assets)
        self.vo_url = Path(vo_url) if vo_url else self.text_assets / "VO_URL.lua"
        self.protocol_names = self._load_protocol_names()
        self.by_name, self.by_id = self._load_classes()

    def _load_protocol_names(self) -> Dict[int, str]:
        if not self.vo_url.is_file():
            return {}
        text = self.vo_url.read_text(encoding="utf-8", errors="ignore")
        names: Dict[int, str] = {}
        for line in text.splitlines():
            match = re.search(r"\[['\"]?(\d+)['\"]?\]\s*=\s*setmetatable\(\{['\"]\d+['\"],['\"]([^'\"]+)['\"]", line)
            if match:
                names[int(match.group(1))] = match.group(2).split(".")[-1]
        return names

    def _load_classes(self) -> Tuple[Dict[str, LuaPacketClassInfo], Dict[int, LuaPacketClassInfo]]:
        by_name: Dict[str, LuaPacketClassInfo] = {}
        by_id: Dict[int, LuaPacketClassInfo] = {}
        for path in sorted(self.text_assets.glob("*.lua")):
            if "__" in path.stem:
                continue
            info = self.parse_class_file(path)
            by_name[info.name] = info
            if info.packet_id is not None:
                by_id[info.packet_id] = info
        return by_name, by_id

    @staticmethod
    def parse_class_file(path: Path) -> LuaPacketClassInfo:
        text = path.read_text(encoding="utf-8", errors="ignore")
        name = path.stem.split("__", 1)[0]
        parent = None
        class_match = re.search(r"_M\s*=\s*class\(([^,\)]+)", text)
        if class_match:
            parent = class_match.group(1).split(".")[-1].strip()
        packet_id = None
        id_match = re.search(r"function _M\.getId\(self\)\s*return\s+(-?\d+)", text, re.S)
        if id_match:
            packet_id = int(id_match.group(1))

        aliases = {
            alias: req.split(".")[-1]
            for alias, req in re.findall(r'local\s+(\w+)\s*=\s*require"([^"]+)"', text)
        }
        write_methods = {
            field: method
            for method, field in re.findall(r"self:write([A-Za-z0-9_]+)\(\s*self\.([A-Za-z0-9_]+)", text)
        }
        ops: List[Tuple[str, str, Optional[str]]] = []
        reading = re.search(r"function _M\.reading\(self\)(.*?)function _M\.writing", text, re.S)
        if reading:
            for raw_line in reading.group(1).splitlines():
                line = raw_line.strip()
                if not line or line.startswith("local "):
                    continue
                if "_M._super_.reading" in line:
                    ops.append(("super", "", None))
                    continue
                primitive = re.search(r"self\.(\w+)\s*=\s*self:read(Int|Long|String|Float|Double|Bool)\(\)", line)
                if primitive:
                    ops.append(("primitive", primitive.group(1), primitive.group(2)))
                    continue
                bean = re.search(r"self\.(\w+)\s*=.*self:readBean\(typeof\((\w+)\)\)", line)
                if bean:
                    ops.append(("bean", bean.group(1), aliases.get(bean.group(2), bean.group(2))))
                    continue
                list_match = re.search(r"self:readMessageList2List\(self\.(\w+)\)", line)
                if list_match:
                    field = list_match.group(1)
                    ops.append(("list", field, write_methods.get(field)))
                    continue
                map_match = re.search(r"self:readMessageMap2Dic\(self\.(\w+)\)", line)
                if map_match:
                    ops.append(("map", map_match.group(1), None))
        return LuaPacketClassInfo(name=name, path=path, packet_id=packet_id, parent=parent, ops=ops)

    def decode_packet_payload(self, packet_id: int, payload: bytes) -> Dict[str, Any]:
        data, compressed = maybe_zlib_decompress(payload)
        info = self.by_id.get(packet_id)
        if info is None:
            return {"zlib": compressed, "payload_hex": data[:160].hex()}
        reader = VarintBinaryReader(data)
        result: Dict[str, Any] = {"zlib": compressed}
        try:
            result["parsed"] = self._parse_class(reader, info)
            result["parsed_bytes"] = reader.pos
            result["remain"] = reader.left()
        except Exception as exc:
            result["parse_error"] = str(exc)
            result["parsed_bytes"] = reader.pos
            result["remain_hex"] = data[reader.pos : reader.pos + 160].hex()
        return result

    def _parse_class(self, reader: VarintBinaryReader, info: LuaPacketClassInfo, depth: int = 0) -> Dict[str, Any]:
        if depth > 16:
            raise PacketDecodeError("max bean depth reached")
        obj: Dict[str, Any] = {"_class": info.name}
        for kind, field, arg in info.ops:
            if kind == "primitive":
                obj[field] = self._read_primitive(reader, arg or "")
            elif kind == "bean":
                obj[field] = self._read_bean(reader, expected=arg, depth=depth + 1)
            elif kind == "list":
                obj[field] = self._read_list(reader, write_method=arg, depth=depth + 1)
            elif kind == "map":
                obj[field] = self._read_map(reader, depth=depth + 1)
            elif kind == "super" and info.parent in self.by_name:
                obj["_super"] = self._parse_class(reader, self.by_name[info.parent], depth=depth + 1)
        return obj

    def _read_primitive(self, reader: VarintBinaryReader, method: str) -> Any:
        if method == "Int":
            return reader.read_int()
        if method == "Long":
            return reader.read_long()
        if method == "String":
            return reader.read_string()
        if method == "Float":
            return reader.read_float()
        if method == "Double":
            return reader.read_double()
        if method == "Bool":
            return reader.read_bool()
        raise PacketDecodeError(f"unsupported primitive: {method}")

    def _read_bean(self, reader: VarintBinaryReader, expected: Optional[str] = None, depth: int = 0) -> Any:
        bean_id = reader.read_int()
        if bean_id in (-1, 0):
            return None
        info = self.by_id.get(bean_id)
        if info is None:
            return {"_bean_id": bean_id, "_unparsed_at": reader.pos}
        obj = self._parse_class(reader, info, depth=depth)
        obj["_bean_id"] = bean_id
        if expected and info.name != expected:
            obj["_expected"] = expected
        return obj

    def _read_list(self, reader: VarintBinaryReader, write_method: Optional[str] = None, depth: int = 0) -> Any:
        count = reader.read_int()
        if count < -1:
            count = reader.read_int()
        if count <= 0:
            return []
        flag = reader.read_byte()
        values: List[Any] = []
        if flag == 1:
            type_id = reader.read_int()
            if type_id < 0 and type_id > -44:
                for _ in range(count):
                    values.append(self._read_base_by_type(reader, type_id, depth=depth))
                return {"_count": count, "_type_id": type_id, "items": values}
            info = self.by_id.get(type_id)
            for _ in range(count):
                if info is None:
                    values.append({"_type_id": type_id, "_unparsed_at": reader.pos})
                    break
                values.append(self._parse_class(reader, info, depth=depth))
            return {"_count": count, "_type_id": type_id, "_type": info.name if info else None, "items": values}
        for _ in range(count):
            type_id = reader.read_int()
            if type_id < 0 and type_id > -44:
                values.append(self._read_base_by_type(reader, type_id, depth=depth))
                continue
            info = self.by_id.get(type_id)
            if info is None:
                values.append({"_type_id": type_id, "_unparsed_at": reader.pos})
                break
            item = self._parse_class(reader, info, depth=depth)
            item["_bean_id"] = type_id
            values.append(item)
        return {"_count": count, "_flag": flag, "items": values}

    def _read_primitive_list(self, reader: VarintBinaryReader, method: str) -> Any:
        count = reader.read_int()
        if count <= 0:
            return []
        return [self._read_primitive(reader, method) for _ in range(min(count, 5000))]

    def _read_base_by_type(self, reader: VarintBinaryReader, type_id: int, depth: int = 0) -> Any:
        if type_id in {-2, -10}:
            return reader.read_bool()
        if type_id in {-3, -11}:
            return reader.read_byte()
        if type_id in {-5, -13, -6, -14}:
            return reader.read_int()
        if type_id in {-7, -15}:
            return reader.read_long()
        if type_id in {-8, -16}:
            return reader.read_float()
        if type_id in {-9, -17}:
            return reader.read_double()
        if type_id == -18:
            return reader.read_string()
        if -30 <= type_id <= -20 or type_id in {-42, -43}:
            return self._read_list(reader, depth=depth + 1)
        if -40 <= type_id <= -31:
            return self._read_map(reader, depth=depth + 1)
        return {"_base_type": type_id, "_unparsed_at": reader.pos}

    def _read_map(self, reader: VarintBinaryReader, depth: int = 0) -> Any:
        count = reader.read_int()
        if count <= 0:
            return []
        flags = reader.read_byte()
        same_key_type = bool(flags & 1)
        same_value_type = bool(flags & 2)
        key_type = reader.read_int() if same_key_type else None
        value_type = reader.read_int() if same_value_type else None
        items: List[Dict[str, Any]] = []
        for _ in range(min(count, 5000)):
            current_key_type = key_type if same_key_type else reader.read_int()
            key = self._read_base_by_type(reader, int(current_key_type), depth=depth + 1)
            current_value_type = value_type if same_value_type else reader.read_int()
            value = self._read_base_by_type(reader, int(current_value_type), depth=depth + 1)
            items.append({"key": key, "value": value})
        return {"_count": count, "_flags": flags, "items": items}


def decode_lusuo_frames(data: bytes, schema: LuaPacketSchemaIndex) -> List[Dict[str, Any]]:
    frames: List[Dict[str, Any]] = []
    for frame in iter_u32be_length_prefixed_frames(data):
        reader = VarintBinaryReader(frame.body)
        sn = reader.read_int()
        packet_id = reader.read_int()
        payload = frame.body[reader.pos :]
        item: Dict[str, Any] = {
            "offset": frame.offset,
            "frame_len": frame.length,
            "sn": sn,
            "pro_id": packet_id,
            "name": schema.protocol_names.get(packet_id),
            "payload_len": len(payload),
        }
        item.update(schema.decode_packet_payload(packet_id, payload))
        frames.append(item)
    return frames


def summarize_decoded_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts = Counter((int(item["pro_id"]), item.get("name") or "") for item in frames)
    return [
        {"pro_id": pro_id, "name": name, "count": count}
        for (pro_id, name), count in counts.most_common()
    ]
