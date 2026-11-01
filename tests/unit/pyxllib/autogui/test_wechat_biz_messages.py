import sqlite3

import zstandard

from pyxllib.autogui.wechat_db import WeChatDbStorage, message_table_name


def _snapshot(root):
    (root/'message').mkdir(parents=True)
    (root/'contact').mkdir()
    conn = sqlite3.connect(root/'message'/'biz_message_0.db')
    conn.execute('CREATE TABLE Name2Id (user_name TEXT, is_session INT)')
    conn.execute("INSERT INTO Name2Id VALUES ('gh_a', 1), ('gh_b', 1), ('self', NULL)")
    for name in ('gh_a', 'gh_b'):
        conn.execute(f'CREATE TABLE "{message_table_name(name)}" '
                     '(local_id INT, server_id INT, local_type INT, create_time INT, message_content)')
    packed = zstandard.ZstdCompressor().compress('<msg>推送</msg>'.encode())
    conn.execute(f'INSERT INTO "{message_table_name("gh_a")}" VALUES (1, 7, ?, 50, ?), (2, 8, 1, 70, ?)',
                 ((5 << 32) | 49, packed, b'hi'))
    conn.execute(f'INSERT INTO "{message_table_name("gh_b")}" VALUES (1, 9, 49, 60, ?)', (b'<msg>b</msg>',))
    conn.commit()
    conn.close()
    conn = sqlite3.connect(root/'contact'/'contact.db')
    conn.execute('CREATE TABLE contact (username TEXT, remark TEXT, nick_name TEXT, alias TEXT, '
                 'local_type INT, flag INT, chat_room_type INT)')
    conn.execute("INSERT INTO contact VALUES ('gh_a', '', '号A', '', 0, 0, 0)")
    conn.commit()
    conn.close()


def test_biz_sessions_and_messages(tmp_path):
    _snapshot(tmp_path)
    storage = WeChatDbStorage(tmp_path)
    assert [(s['username'], s['name'], s['message_count'], s['last_time']) for s in storage.list_biz_sessions()] == [
        ('gh_a', '号A', 2, 70), ('gh_b', 'gh_b', 1, 60)]
    rows = storage.list_biz_messages(message_type=49)
    assert [(r['username'], r['create_time'], r['message_text']) for r in rows] == [
        ('gh_b', 60, '<msg>b</msg>'), ('gh_a', 50, '<msg>推送</msg>')]
    assert [r['local_id'] for r in storage.list_biz_messages('gh_a', since=50)] == [2]
    assert WeChatDbStorage(tmp_path/'missing').list_biz_messages() == []
