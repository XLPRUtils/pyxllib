from pyxllib.autogui.wechat_db import _parse_appmsg


def test_parse_appmsg_expands_forwarded_text_and_image_items():
    record = (
        "&lt;recordinfo&gt;&lt;datalist count=\"2\"&gt;"
        "&lt;dataitem datatype=\"2\" dataid=\"image-1\"&gt;"
        "&lt;datadesc&gt;[图片]&lt;/datadesc&gt;&lt;sourcename&gt;elsa&lt;/sourcename&gt;"
        "&lt;sourcetime&gt;2026-08-14 12:05:12&lt;/sourcetime&gt;"
        "&lt;datasize&gt;635917&lt;/datasize&gt;&lt;fullmd5&gt;abc&lt;/fullmd5&gt;"
        "&lt;cdndataurl&gt;data-url&lt;/cdndataurl&gt;&lt;cdndatakey&gt;data-key&lt;/cdndatakey&gt;"
        "&lt;/dataitem&gt;"
        "&lt;dataitem datatype=\"1\" dataid=\"text-1\"&gt;"
        "&lt;datadesc&gt;三位师兄考勤没统计到&lt;/datadesc&gt;"
        "&lt;sourcename&gt;elsa&lt;/sourcename&gt;"
        "&lt;/dataitem&gt;&lt;/datalist&gt;&lt;/recordinfo&gt;"
    )
    payload = _parse_appmsg(
        f"<msg><appmsg><type>19</type><title>聊天记录</title><recorditem>{record}</recorditem></appmsg></msg>"
    )

    assert payload is not None
    assert payload["app_type"] == 19
    assert payload["forwarded_items"] == [
        {
            "data_index": 0,
            "data_id": "image-1",
            "datatype": 2,
            "speaker": "elsa",
            "source_time": "2026-08-14 12:05:12",
            "text": "[图片]",
            "data_size": 635917,
            "full_md5": "abc",
            "cdn_data_url": "data-url",
            "cdn_data_key": "data-key",
        },
        {
            "data_index": 1,
            "data_id": "text-1",
            "datatype": 1,
            "speaker": "elsa",
            "text": "三位师兄考勤没统计到",
        },
    ]
