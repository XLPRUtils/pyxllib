"""Read one exact emoji attachment; verify CDN bytes before exposing a file."""
from pathlib import Path
import hashlib
import os
import re
import uuid
from urllib.parse import urlparse
import xml.etree.ElementTree as ET


def read_emoji_message(storage, chat_id: str, local_id: int) -> dict:
    """Export a full original emoji, bound to message identity, size and MD5.

    Call only after interpreting the owner's attachment request. CDN URLs come
    from the exact archived message; no guessed IDs or encrypted-file fallback.
    """
    rows = storage.list_messages(chat_id, local_id=int(local_id), limit=1, include_resources=False)['items']
    if len(rows) != 1 or rows[0]['local_id'] != int(local_id):
        return {}
    row = rows[0]
    if row.get('local_type_normalized', row.get('local_type')) != 47:
        return {}
    text = row.get('message_content') or ''
    start = text.find('<msg')
    try:
        node = ET.fromstring(text[start:]).find('emoji')
    except ET.ParseError:
        return {}
    if node is None:
        return {}
    digest = node.get('md5', '').lower()
    expected = int(node.get('len', '0'))
    if not re.fullmatch('[a-f0-9]{32}', digest) or not 0 < expected <= 50 * 1024 * 1024:
        return {}
    target = storage.root.parent / 'exported_media' / 'emoji' / (digest + '.gif')
    data = target.read_bytes() if target.is_file() else b''
    if len(data) != expected or hashlib.md5(data).hexdigest() != digest:
        import requests
        url = node.get('cdnurl', '')
        host = urlparse(url).hostname or ''
        if urlparse(url).scheme not in ('http', 'https') or not host.endswith('.qq.com'):
            return {}
        with requests.get(url, stream=True, timeout=(10, 30), allow_redirects=False) as response:
            response.raise_for_status()
            chunks, size = [], 0
            for chunk in response.iter_content(65536):
                size += len(chunk)
                if size > expected:
                    raise ValueError('动态表情下载超出原始大小')
                chunks.append(chunk)
        data = b''.join(chunks)
        if len(data) != expected or hashlib.md5(data).hexdigest() != digest:
            raise ValueError('动态表情大小或 MD5 校验未通过')
        from PIL import Image
        from io import BytesIO
        with Image.open(BytesIO(data)) as image:
            image.verify()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        temporary.write_bytes(data)
        os.replace(temporary, target)
    from PIL import Image
    with Image.open(target) as image:
        width, height = image.size
        frames, format_name = getattr(image, 'n_frames', 1), image.format
    return {'resource_count': 1, 'items': [{'kind': 'emoji', 'type': 47, 'data_index': '0',
        'export': {'kind': 'image', 'variant': 'high', 'readable': True, 'stored_path': str(target),
                   'size': expected, 'decoded_size': expected, 'md5': digest, 'width': width,
                   'height': height, 'frames': frames, 'format': format_name}}]}
