import ctypes
import hashlib
import hmac
import struct
from types import SimpleNamespace

from pyxllib.autogui.wechat_key_scan import CONFIG_NAME, CONFIG_MASK, scan_account_keys, verify_database_key


def test_scan_accepts_only_key_verified_against_account_database(monkeypatch, tmp_path):
    key = bytes(range(32))
    salt = bytes(range(80, 96))
    page = salt + b'\x9a' * 4016
    mac_key = hashlib.pbkdf2_hmac('sha512', key, bytes(x ^ 0x3a for x in salt), 2, 32)
    page += hmac.new(mac_key, page[16:] + struct.pack('<I', 1), hashlib.sha512).digest()
    (tmp_path / 'contact.db').write_bytes(page)
    memory = bytearray(2048)
    base = 0x10000
    memory[128:128 + len(CONFIG_NAME)] = CONFIG_NAME
    struct.pack_into('<QQ', memory, 256, base + 128, len(CONFIG_NAME))
    struct.pack_into('<Q', memory, 280, base + 512)
    blob = ("x'" + key.hex() + "'").encode()
    encoded = bytes(value ^ CONFIG_MASK[index % len(CONFIG_MASK)] for index, value in enumerate(blob))
    struct.pack_into('<QQ', memory, 512 + 0x88 + 8, base + 1024, len(encoded))
    memory[1024:1024 + len(encoded)] = encoded

    class Function:
        def __init__(self, fn):
            self.fn = fn
        def __call__(self, *args):
            return self.fn(*args)

    def query(handle, address, out, size):
        if address > base:
            return 0
        item = out._obj
        item.base = base
        item.size = len(memory)
        item.state = 0x1000
        item.protect = 4
        item.type = 0x20000
        return size

    def read(handle, address, buffer, size, count):
        offset = address - base
        if not 0 <= offset < len(memory):
            return 0
        data = bytes(memory[offset:offset + size])
        ctypes.memmove(buffer, data, len(data))
        count._obj.value = len(data)
        return 1

    kernel = SimpleNamespace(OpenProcess=Function(lambda *args: 1), VirtualQueryEx=Function(query),
                             ReadProcessMemory=Function(read), CloseHandle=Function(lambda _: 1))
    monkeypatch.setattr(ctypes, 'WinDLL', lambda *args, **kwargs: kernel)
    assert scan_account_keys(123, tmp_path) == {'contact.db': {'key_hex': key.hex(), 'mode': 'raw-derived-key'}}
    assert not verify_database_key(bytes(reversed(key)), page)
    assert not verify_database_key(key, page[:-1] + bytes([page[-1] ^ 1]))
