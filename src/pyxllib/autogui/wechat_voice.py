"""Account-scoped voice assets from the WeChat snapshot provider.

VoiceInfo is separate from MessageResourceInfo, so the image/file resource
index cannot establish that a voice attachment is absent. Resolve the chat,
local id and server id together before exporting any bytes. No GUI or ASR
service is called by this reader.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import wave
import xml.etree.ElementTree as ET


def _protobuf_fields(data: bytes) -> dict:
    """Bounded protobuf wire reader; reject truncated/unknown wire formats."""
    offset = 0
    fields = {}
    def varint():
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(data):
                raise ValueError('Truncated protobuf')
            byte = data[offset]
            offset += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                return value
        raise ValueError('Oversized varint')
    if len(data) > 65536:
        raise ValueError('Oversized packed voice metadata')
    while offset < len(data):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError('Invalid protobuf tag')
        if wire == 0:
            value = varint()
        elif wire in (1, 2, 5):
            size = varint() if wire == 2 else (8 if wire == 1 else 4)
            if offset + size > len(data):
                raise ValueError('Truncated protobuf payload')
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError('Unsupported protobuf wire type')
        fields[field] = value
    return fields


def official_voice_text(packed: bytes) -> str | None:
    """WeChat 4.1 voice result: field 5 contains status=2 and UTF-8 text.

    Verified against the official GUI and two independent received messages.
    Other metadata and pending/failed statuses must never become owner input.
    """
    try:
        result = _protobuf_fields(_protobuf_fields(packed).get(5, b''))
        if result.get(1) != 2 or not isinstance(result.get(2), bytes):
            return None
        text = result[2].decode('utf-8').strip()
        return text if text and all(ord(c) >= 32 or c in '\n\t' for c in text) else None
    except (ValueError, TypeError, UnicodeError):
        return None


def read_voice_message(storage, chat_id: str, local_id: int) -> dict:
    from pyxllib.autogui.wechat_db import _connect_readonly, _table_exists, _decode_text_value, message_table_name
    username = storage._resolve_chat_username(chat_id)
    local_id = int(local_id)
    conn = storage._message_conn('message')
    try:
        table = message_table_name(username)
        if not _table_exists(conn, table):
            raise ValueError('Voice message does not exist')
        columns = {r[1] for r in conn.execute(f'PRAGMA table_info("{table}")')}
        packed = 'packed_info_data' if 'packed_info_data' in columns else 'NULL'
        row = conn.execute(f'SELECT local_type,server_id,message_content,{packed} AS packed FROM "{table}" WHERE local_id=?', (local_id,)).fetchone()
    finally:
        conn.close()
    if row is None or (int(row['local_type']) & 65535) != 34:
        raise ValueError('Target is not a voice message')
    result = {'chat_id': username, 'local_id': local_id, 'server_id': str(row['server_id']),
              'text': None, 'source': None, 'audio': None}
    result['text'] = official_voice_text(bytes(row['packed'] or b''))
    if result['text']:
        result['source'] = 'wechat-official'
    try:
        root = ET.fromstring(_decode_text_value(row['message_content']))
        result['duration_seconds'] = int(root.find('voicemsg').get('voicelength', '0')) / 1000
    except (ET.ParseError, AttributeError, ValueError):
        result['duration_seconds'] = None
    if not storage.paths.media.exists():
        result['error'] = 'Voice database is not available'
        return result
    conn = _connect_readonly(storage.paths.media)
    try:
        if not _table_exists(conn, 'VoiceInfo'):
            result['error'] = 'VoiceInfo is not available'
            return result
        rows = conn.execute('SELECT voice_data FROM VoiceInfo v JOIN Name2Id n ON n.rowid=v.chat_name_id '
                            'WHERE n.user_name=? AND v.local_id=? AND v.svr_id=? ORDER BY v.data_index',
                            (username, local_id, row['server_id'])).fetchall()
    finally:
        conn.close()
    data = b''.join(bytes(r['voice_data'] or b'') for r in rows)
    if not data:
        result['error'] = 'Voice bytes have not been downloaded'
        return result
    if not (data.startswith(b'#!SILK_V3') or data.startswith(b'\x02#!SILK_V3')):
        result['error'] = 'Unknown voice encoding; refusing to decode'
        return result
    folder = storage.root.parent / 'exported_resources' / 'voice' / sha256(username.encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{local_id}-{sha256(data).hexdigest()[:16]}'
    raw = folder / f'{stem}.silk'
    if not raw.exists():
        raw.write_bytes(data)
    audio = {'raw_path': str(raw), 'size': len(data), 'format': 'silk', 'readable': False}
    result['audio'] = audio
    decoded = folder / f'{stem}.wav'
    partial = folder / f'{stem}.wav.part'
    try:
        if not decoded.exists():
            import pysilk
            pcm = BytesIO()
            pysilk.decode(BytesIO(data), pcm, 16000)
            with wave.open(str(partial), 'wb') as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(16000)
                out.writeframes(pcm.getvalue())
            with wave.open(str(partial), 'rb') as wav:
                if wav.getnframes() <= 0:
                    raise ValueError('Empty decoded voice')
            partial.replace(decoded)
        with wave.open(str(decoded), 'rb') as wav:
            seconds = wav.getnframes() / wav.getframerate()
            if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or seconds <= 0:
                raise ValueError('Invalid decoded voice')
        audio.update(stored_path=str(decoded), format='wav', readable=True, duration_seconds=seconds)
    except Exception as exc:
        partial.unlink(missing_ok=True)
        audio['read_error'] = str(exc)
    return result
