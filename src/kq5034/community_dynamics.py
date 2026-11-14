"""Read ordinary community posts through the attendance browser, without exports."""
from datetime import datetime
import json
import time
from urllib.parse import parse_qs, urlencode, urlsplit


def community_id(url):
    parts = urlsplit(str(url))
    if parts.scheme != 'https' or parts.netloc != 'admin.xiaoe-tech.com':
        raise ValueError('动态来源必须是小鹅通后台 URL')
    route = urlsplit(parts.fragment)
    if route.path != '/community_manage/content_settings/feed_list':
        raise ValueError('动态来源必须是圈子动态列表')
    values = parse_qs(route.query).get('communityId', [])
    if len(values) != 1 or not values[0]:
        raise ValueError('动态来源缺少唯一 communityId')
    return values[0]


def query_community_dynamics(web, url, *, start_at, end_at, max_pages=200):
    """Return a complete, deduplicated [start_at, end_at) snapshot or raise.

    Date filtering on the platform is inclusive; enforce exact bounds locally.
    Each post retains its stable ID, user ID and original content. No inferred
    study/practice classification is performed here.
    """
    cid = community_id(url)
    start = datetime.fromisoformat(str(start_at))
    end = datetime.fromisoformat(str(end_at))
    if end <= start or not 1 <= max_pages <= 200:
        raise ValueError('动态查询时间范围或分页上限无效')
    rows, seen = [], set()
    deadline = time.monotonic() + 120
    with web.临时工作标签页(url=url, page_load_timeout_seconds=20) as tab:
        for page in range(1, max_pages + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError('动态分页超过120秒，未返回不完整结果')
            params = dict(community_id=cid, page_index=page, page_size=100,
                          feeds_list_type=-1, search_content='', nick_name='', roles='',
                          start_date=start.date().isoformat(), end_date=end.date().isoformat())
            target = '/small_community/b_feeds_list?' + urlencode(params)
            response = tab.run_js('''const x=new XMLHttpRequest();
                x.open('GET',arguments[0],false);x.withCredentials=true;x.send(null);
                return {status:x.status,text:x.responseText};''', target)
            if not isinstance(response, dict) or response.get('status') != 200:
                raise RuntimeError('动态接口 HTTP 请求失败')
            payload = json.loads(response['text'])
            data = payload.get('data')
            if payload.get('code') != 0 or not isinstance(data, dict) or not isinstance(data.get('list'), list):
                raise RuntimeError('动态接口返回失败或结构异常')
            if 'total_count' not in data or int(data['total_count']) < 0:
                raise RuntimeError('动态接口缺少有效总数')
            batch = data['list']
            added = 0
            for row in batch:
                identifier = str(row.get('id') or '')
                if not identifier or not row.get('user_id') or not row.get('created_at'):
                    raise RuntimeError('动态缺少 ID、用户或发布时间')
                if identifier in seen:
                    continue
                seen.add(identifier); added += 1
                published = datetime.fromisoformat(str(row['created_at']))
                if start <= published < end:
                    rows.append({**row, 'community_id': cid})
            total = int(data.get('total_count') or 0)
            if len(seen) >= total:
                return rows
            if not batch or not added:
                raise RuntimeError('动态分页提前结束或重复，拒绝不完整快照')
        raise RuntimeError('动态分页达到上限，拒绝不完整快照')
