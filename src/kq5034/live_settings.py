"""Current Xiaoe live settings reader; observes configuration without saving edits."""
from datetime import datetime, timedelta
import re
import time
from urllib.parse import quote


def parse_live_playback_settings(snapshot):
    text = snapshot['text']
    match = re.search(r'([^\n]+)\n直播时间[：:]\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*至\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', text)
    if not match:
        raise ValueError('直播标题或起止时间未加载')
    name, start_text, finish_text = match.groups()
    start, finish = datetime.fromisoformat(start_text), datetime.fromisoformat(finish_text)
    expiry = [x for x in snapshot['selected'] if any(t in x['text'] for t in ('永久有效','直播结束后','指定时段有效'))]
    if len(expiry) != 1:
        raise ValueError('无法确定回放有效期选项')
    mode = expiry[0]
    if '永久有效' in mode['text']:
        end = ''
    elif '直播结束后' in mode['text']:
        if not mode.get('inputs'):
            raise ValueError('回放天数未加载')
        days = int(mode['inputs'][0])
        if days <= 0:
            raise ValueError('回放天数无效')
        end = (finish + timedelta(days=days)).isoformat(sep=' ')
    else:
        values = [x for x in snapshot['inputs'] if re.fullmatch(r'\d{4}-\d{2}-\d{2}( \d{2}:\d{2}:\d{2})?', x)]
        if not values:
            raise ValueError('指定回放截止时间未加载')
        end = max(values)
        if len(end) == 10:
            end += ' 23:59:59'
    durations = re.findall(r'(?:^|\n)\s*(\d{1,3}:\d{2}(?::\d{2})?)\s*(?:\n|$)', snapshot.get('table_text',''))
    if len(durations) != 1:
        raise ValueError(f'回放视频时长不唯一：{durations}')
    parts = list(map(int, durations[0].split(':')))
    duration = sum(n * 60 ** i for i,n in enumerate(reversed(parts)))
    if finish <= start or duration <= 0:
        raise ValueError('直播时间或视频时长无效')
    return {'lesson_name': name, 'start_date': start_text, 'end_date': end,
            'next_update': finish_text, 'video_duration': duration}


def read_live_playback_settings(browser, resource_id, *, timeout=30):
    """Read one existing live resource; close the owned tab even on timeout."""
    tab = browser.new_tab()
    try:
        tab.get('https://admin.xiaoe-tech.com/t/live_management#/baseSetting?id='
                + quote(str(resource_id),safe='') + '&tabName=playbackSetting')
        deadline = time.monotonic() + timeout
        error = None
        while time.monotonic() < deadline:
            snapshot = tab.run_js(r"""
                return {text: document.body.innerText,
                    table_text: Array.from(document.querySelectorAll('table')).map(x=>x.innerText).join('\n'),
                    selected: Array.from(document.querySelectorAll('input[type=radio]')).filter(x=>x.checked)
                      .map(x=>({text:x.closest('label')?.innerText || '',
                        inputs:Array.from(x.closest('label')?.querySelectorAll('input[type=text]') || []).map(y=>y.value)})),
                    inputs:Array.from(document.querySelectorAll('input')).map(x=>x.value)};
            """)
            try:
                return parse_live_playback_settings(snapshot)
            except ValueError as exc:
                error = exc
                time.sleep(.25)
        raise RuntimeError(f'直播资源 {resource_id} 配置读取超时：{error}; url={tab.url}')
    finally:
        tab.close()
