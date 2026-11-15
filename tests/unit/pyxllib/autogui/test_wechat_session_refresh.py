import pytest
from pyxllib.autogui.wechat_db import WeChatDbStorage, WeChatDbError


def storage(monkeypatch, tmp_path, published):
    obj = WeChatDbStorage(tmp_path / 'snapshot')
    monkeypatch.setattr(obj, '_wechat_account_root', lambda: tmp_path)
    monkeypatch.setattr(obj, '_copy_live_db_storage', lambda root, **kwargs: {'error_count': 0})
    def decrypt(root, *, databases):
        assert databases == {'session/session.db'}
        return published
    monkeypatch.setattr(obj, '_decrypt_snapshot_dbs', decrypt)
    monkeypatch.setattr(obj, '_session_map', lambda: {'u': {'unread_count': 3}})
    return obj


def test_refresh_excludes_message_shards(monkeypatch, tmp_path):
    obj = storage(monkeypatch, tmp_path, {'decrypted': 1})
    assert obj.session_unread(refresh=True) == {'u': 3}


@pytest.mark.parametrize('published', [{'failed_count': 1}, {'skipped': 1}, {}])
def test_refresh_cannot_accept_stale_or_missing_session(monkeypatch, tmp_path, published):
    obj = storage(monkeypatch, tmp_path, published)
    monkeypatch.setattr(obj, '_session_map', lambda: pytest.fail('stale read'))
    with pytest.raises(WeChatDbError, match='发布失败'):
        obj.session_unread(refresh=True)


def test_session_copy_does_not_touch_or_forget_message_shard(tmp_path):
    live = tmp_path / 'live'
    session = live / 'db_storage/session/session.db'
    message = live / 'db_storage/message/message_0.db'
    for path in (session, message):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'snapshot')
    obj = WeChatDbStorage(tmp_path / 'archive/db_storage')
    obj._save_sync_state({'live_db_files': {'message/message_0.db': {'sentinel': True}}})
    result = obj._copy_live_db_storage(live, databases={'session/session.db'})
    assert result['copied'] == 1 and result['error_count'] == 0
    target = obj._raw_snapshot_db_storage(live)
    assert (target / 'session/session.db').exists()
    assert not (target / 'message/message_0.db').exists()
    assert obj._load_sync_state()['live_db_files']['message/message_0.db'] == {'sentinel': True}


def test_session_identities_include_business_accounts_without_message_history(monkeypatch, tmp_path):
    obj = WeChatDbStorage(tmp_path / 'archive')
    monkeypatch.setattr(obj, '_contact_map', lambda: {'gh_service': {'nick_name': '懒人信息转发服务'}})
    monkeypatch.setattr(obj, '_session_map', lambda: {'gh_service': {'unread_count': 2, 'type': 0}})
    monkeypatch.setattr(obj, '_message_conn', lambda *args: pytest.fail('must not require ordinary chat history'))
    assert obj.list_sessions(unread_only=True)[0]['name'] == '懒人信息转发服务'
