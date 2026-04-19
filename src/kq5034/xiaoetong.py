"""小鹅通相关实现。"""

from .common import *  # noqa: F403

class XiaoetongApi:
    """ 小鹅通api """

    def __init__(self):
        self.token = None

    def login(self, app_id=None, client_id=None, secret_key=None):
        """ 登录，获取token
        """
        app_id = app_id or os.getenv('XIAOETONG_APP_ID')
        client_id = client_id or os.getenv('XIAOETONG_CLIENT_ID')
        secret_key = secret_key or os.getenv('XIAOETONG_SECRET_KEY')

        # 启用缓存(是否可以不启用？我先关了试试，如果会报错，可以再打开)
        # requests_cache.install_cache('access_token_cache', expire_after=None)  # 设置缓存过期时间xx（单位：秒）
        # 接口地址
        url = "https://api.xiaoe-tech.com/token"
        params = {
            "app_id": app_id,
            "client_id": client_id,
            "secret_key": secret_key,
            "grant_type": "client_credential"
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            result = response.json()
            if result['code'] == 0:
                # access_token 是小鹅通开放api的全局唯一接口调用凭据，店铺调用各接口时都需使用 access_token ,开发者需要进行妥善保管；
                self.token = result['data']['access_token']
            else:
                raise Exception("Error getting access token: {}".format(result['msg']))
        else:
            raise Exception("HTTP request failed with status code {}".format(response.status_code))

    def get_new_user_list(self, maxn=-1, exists_users=None):
        """ 获得(店铺1)新增的用户数据

        :param maxn: 最多获取多少条数据, -1表示全部获取
        :param exists_users: 已经存在的用户id，遇到这里面的任意一个用户时，都会停止获取
        """
        exists_users = exists_users or set()

        url = "https://api.xiaoe-tech.com/xe.user.batch.get/2.0.0"
        post_data = {
            "access_token": self.token,
            "page_size": 50,  # 每页条数，最大50条
            "need_column": [
                "latest_visited_at"
            ]
        }

        data2, lst_inc = [], []
        while True:
            if data2:
                post_data["es_skip"] = data2[-1]['es_skip']
            response = requests.post(url, json=post_data)
            items = response.json()['data']['list']
            if not items:
                break
            for item in items:
                if item['user_id'] not in exists_users:  # 实现"增量"更新
                    # 转成跟数据库里对称的格式
                    date = datetime.datetime.fromtimestamp(item['latest_visited_at'] / 1000)
                    item['latest_visited_at'] = date.strftime("%Y-%m-%d %H:%M:%S")
                    lst = [('from_channel', 'from'), ('user_id2', 'user_id'), ('consume_money', 'pay_sum'),
                           ('last_visited_at', 'latest_visited_at'), ('consume_times', 'punch_count')]
                    for x, y in lst:
                        item[x] = item.pop(y)
                    item['shop_id'] = 1
                    data2.append(item)
                    if len(data2) == maxn:
                        return data2[::-1]
                else:
                    return data2[::-1]
        return data2[::-1]

    def get_user_detail(self, user_id):
        """ 获得指定用户的详细数据 """
        url = 'https://api.xiaoe-tech.com/xe.user.info.get/1.0.0'
        data = {
            "access_token": self.token,
            "user_id": user_id,
            "data": {
                "field_list": [
                    "sdk_user_id",  # sdk用户id
                    "wx_email",  # 微信邮箱
                    "name",  # 真实姓名
                    "gender",  # 性别【0-无 1-男 2-女
                    "city",  # 城市
                    "province",  # 省份
                    "country",  # 国家
                    "birth",  # 生日
                    "address",  # 地址
                    "company",  # 公司
                    "is_seal",  # 用户状态【-1-已注销，0-正常，1-已封号，2-待注销状态，3-待激活
                    "job",  # 职位
                    "wx_account",  # 微信号
                ]
            }
        }
        response = requests.post(url, json=data)
        item = response.json()["data"]
        item['gender'] = {1: '男', 2: '女'}.get(item['gender'], None)
        item['is_seal'] = {-2: '待注销状态', -1: '已注销', 0: '正常',
                           1: '已封号', 3: '待激活'}.get(item['is_seal'], None)
        # 删除None类型的值
        for k in list(item.keys()):
            if item[k] is None:
                del item[k]
        # 删除一些字段
        lst2 = ['user_id']
        for x in lst2:
            if x in item:
                del item[x]
        return item

    def get_new_user_list_with_detail(self, maxn=-1, exists_users=None):
        users = self.get_new_user_list(maxn, exists_users)
        for user in users:
            user.update(self.get_user_detail(user['user_id2']))
        return users

    def get_alive_user_list(self, resource_id, page_size=100):
        """ 获取直播间用户
        """
        # 1 获取总页数
        url = "https://api.xiaoe-tech.com/xe.alive.user.list/1.0.0"  # 接口地址【路径：API列表 -> 直播管理 -> 获取直播间用户列表】
        data_1 = {
            "access_token": self.token,
            "resource_id": resource_id,
            "page": 1,
            "page_size": page_size
        }
        response_1 = requests.post(url, data=data_1)
        result_1 = response_1.json()
        page = math.ceil(result_1['data']['total'] / page_size)  # 页数

        # 2 获取直播间用户数据
        lst = result_1['data']['list']
        for i in range(1, page):  # 为什么从1开始，因为第一页的数据上面已经获取到了，这里没必要从新获取一次
            data = {
                "access_token": self.token,
                "resource_id": resource_id,
                "page": i + 1,
                "page_size": page_size
            }
            response = requests.post(url, data=data)
            result = response.json()
            data_1 = result['data']['list']
            lst += data_1
            # lst.extend(data_1)
        return lst

    def get_elock_actor(self, activity_id, page_size=100):
        """ 获取打卡参与用户
        """
        # 获取总页数
        url = "https://api.xiaoe-tech.com/xe.elock.actor/1.0.0"  # 接口地址【路径：API列表 -> 打卡管理 -> 获取打卡参与用户】
        data_1 = {
            "access_token": self.token,
            "activity_id": activity_id,
            "page_index": 1,
            "page_size": page_size
        }
        response_1 = requests.post(url, data=data_1)
        result_1 = response_1.json()
        page = math.ceil(result_1['data']['count'] / page_size)  # 页数
        # 获取打卡用户数据
        lst = result_1['data']['list']
        for i in range(1, page):  # 为什么从1开始，因为第一页的数据上面已经获取到了，这里没必要从新获取一次
            data = {
                "access_token": self.token,
                "activity_id": activity_id,
                "page_index": i + 1,
                "page_size": page_size
            }
            response = requests.post(url, data=data)
            result = response.json()
            data_1 = result['data']['list']
            lst += data_1
            # lst.extend(data_1)
        return lst


class XiaoetongWeb(DpWebBase):
    """ 网页版的小鹅通爬虫 """

    _CACHE_MISS = object()
    _EMPTY_EXPORT = object()
    _runtime_export_cache = {}

    def __init__(self, name=None, passwd=None):
        super().__init__('https://admin.xiaoe-tech.com')
        self.exist_files = set()  # 已下载过的文件名

        self.name = name or os.getenv('XIAOETONG_USERNAME')
        self.passwd = passwd or os.getenv('XIAOETONG_PASSWORD')

        self.cur_shop_id = None

    @classmethod
    def clear_runtime_export_cache(cls):
        cls._runtime_export_cache.clear()

    def _make_runtime_cache_key(self, category, identifier, *args, shop_id=None):
        shop_id = shop_id if shop_id is not None else (self.cur_shop_id or 0)
        return (category, shop_id, identifier, *args)

    def _restore_runtime_cached_file(self, key, *, label=''):
        cache = type(self)._runtime_export_cache
        if key not in cache:
            return self._CACHE_MISS

        entry = cache[key]
        if entry is self._EMPTY_EXPORT:
            if label:
                logger.info(f'命中导出缓存（空结果）：{label}')
            return None

        if label:
            logger.info(f'命中导出缓存：{label}')

        with tempfile.NamedTemporaryFile(suffix=entry['suffix'], delete=False) as tmp:
            tmp.write(entry['content'])
            return XlPath(tmp.name)

    def _store_runtime_cached_file(self, key, file):
        cache = type(self)._runtime_export_cache
        if file is None:
            cache[key] = self._EMPTY_EXPORT
            return None

        file = XlPath(file)
        if not file.exists():
            return file

        cache[key] = {
            'suffix': file.suffix,
            'content': file.read_bytes(),
        }
        return file

    @staticmethod
    def _标准化店铺(shop):
        if isinstance(shop, int):
            shop = ['5034山中薪', '宗门学府'][shop - 1]
        shop_id = 1 if shop == '5034山中薪' else 2 if shop == '宗门学府' else None
        if shop_id is None:
            raise ValueError(f'未知店铺名：{shop}')
        return shop, shop_id

    def _当前店铺名(self, timeout=0.8):
        tab = self.tab
        for name in ('5034山中薪', '宗门学府'):
            if tab(f'tag:span@@class:global-shop-name@@text()={name}', timeout=timeout):
                return name
            if tab(f'tag:div@@class=shop-name@@text()={name}', timeout=0):
                return name
        return ''

    def _重连当前标签页(self):
        try:
            _ = self.tab.url
            return self.tab
        except DrissionPage.errors.PageDisconnectedError:
            logger.warning('小鹅通当前标签页连接已断开，尝试重连到可用页签')

        with contextlib.suppress(Exception):
            tab = get_latest_not_dev_tab(self.browser)
            if tab:
                _ = tab.url
                self.tab = tab
                return tab

        tab = self.browser.new_tab()
        self.tab = tab
        return tab

    def _在选店页点击店铺(self, shop):
        tab = self.tab
        for _ in range(10):
            clicked = tab.run_js(f"""
const target = {shop!r};
const rows = [...document.querySelectorAll('.shop-list > .shop-list-item')]
  .filter(row => row.getBoundingClientRect().width > 0 && row.getBoundingClientRect().height > 0);
const row = rows.find(row => (row.innerText || '').includes(target));
if (!row) return false;
row.click();
return true;
""")
            if clicked:
                return True
            tab.wait(0.5)
        return False

    def switch_shop(self, shop='5034山中薪'):
        """ 返回一个小鹅通指定店铺的新tab页面 """
        tab = self._重连当前标签页()
        shop, shop_id = self._标准化店铺(shop)

        if self.cur_shop_id == shop_id and self._当前店铺名() == shop:
            return tab

        # 1 检查是否已在目标店铺
        max_attempts = 8
        for attempt in range(1, max_attempts + 1):
            try:
                current_shop = self._当前店铺名()
            except DrissionPage.errors.PageDisconnectedError:
                tab = self._重连当前标签页()
                current_shop = ''
            if current_shop == shop:
                self.cur_shop_id = shop_id
                break

            if tab.url.startswith('https://admin.xiaoe-tech.com/t/login#'):  # 跳转到了登录页
                tab.get('https://admin.xiaoe-tech.com/t/login#/acount')
                tab('t:input@@placeholder=请输入手机号').input(self.name, clear=True)
                tab('t:input@@placeholder=请输入密码').input(self.passwd, clear=True)
                tab('t:label@@for=agree').click()
                tab('t:span@@text()=登录').click()
                tab.wait(2)
                # 下述验证码算法不一定精确，但是有能力多次迭代尝试可能能试出来。一直用一台电脑，希望尽量不触发重登录是最好的。
                iframe = tab.get_frame('#tcaptcha_iframe_dy')
                img_file1 = iframe('#slideBg').get_screenshot(Path(tempfile.gettempdir()) / f'{time.time()}.png')
                scl = SliderCaptchaLocator(img_file1)
                scl.radius = 27
                pos = scl.find_captcha_position()
                logger.info(pos)
                # scl.debug()
                iframe('.tc-fg-item tc-slider-normal').drag(pos)
                tab.wait(2)
                continue

            if tab.url != 'https://admin.xiaoe-tech.com/t/account/muti_index#/chooseShop':
                try:
                    tab.get('https://admin.xiaoe-tech.com/t/account/muti_index#/chooseShop')
                except DrissionPage.errors.PageDisconnectedError:
                    tab = self._重连当前标签页()
                    tab.get('https://admin.xiaoe-tech.com/t/account/muti_index#/chooseShop')
            tab.wait.doc_loaded()
            if tab.url == 'https://admin.xiaoe-tech.com/t/account/muti_index#/chooseShop':  # 跳转到了店铺页
                if not self._在选店页点击店铺(shop):
                    raise RuntimeError(f'选店页未找到目标店铺：{shop}')
                tab.wait(2)
                current_shop = self._当前店铺名()
                if current_shop == shop:
                    self.cur_shop_id = shop_id
                    break
                logger.warning(f'切换店铺重试 {attempt}/{max_attempts}: target={shop} url={tab.url} current_shop={current_shop or "未知"}')
        else:
            current_shop = self._当前店铺名()
            raise RuntimeError(f'切换店铺失败：target={shop} url={tab.url} current_shop={current_shop or "未知"}')

        # 关闭下载的提示框
        # tab.run_js('document.querySelector(".notify-wrap")?.remove()')
        for t in tab.eles('t:i@@class=sense-icon-close'):
            t.click(by_js=True)

        self.cur_shop_id = shop_id

        return self.tab

    @contextlib.contextmanager
    def 临时工作标签页(self, *, url=None, wait_seconds=0):
        """ 为一次性页面操作创建独立标签页，结束后自动关闭，避免堆积重复页面 """
        original_tab = self.tab
        tab = self.browser.new_tab()
        self.tab = tab
        try:
            if url:
                tab.get(url)
            if wait_seconds:
                tab.wait(wait_seconds)
            yield tab
        finally:
            try:
                if getattr(tab, 'tab_id', None):
                    tab._run_cdp('Target.closeTarget', targetId=tab.tab_id)
                else:
                    tab.close()
            except Exception:
                with contextlib.suppress(Exception):
                    tab.close()
            self.tab = original_tab

    def __1_导出各种表格数据(self):
        pass

    def download_last_file(self):
        """ 去数据中心等着下载最新的生成的数据文件
        """
        tab = self.tab

        # 1 等待按钮变可下载
        tr = ele = None
        max_attempts = 30
        for attempt in range(1, max_attempts + 1):
            tab.get('https://admin.xiaoe-tech.com/t/basic-platform/downloadCenter#/')
            tab.wait(5)
            rows = []
            for tbody in tab.eles('tag:tbody'):
                rows.extend(tbody.eles('tag:tr'))

            for row in rows:
                btn = row('tag:button@@text():下载', timeout=0.5)
                if btn:
                    tr, ele = row, btn
                    break

            if ele:
                break

            logger.warning(f'下载中心暂未出现可下载任务，重试 {attempt}/{max_attempts}，url={tab.url}')
            tab.refresh()
            tab.wait(2)
        else:
            raise RuntimeError(f'下载中心等待超时，未找到可下载任务：url={tab.url}')

        # 2 判断是不是之前已下载过的文件，如果是就是上游出现问题了。
        #   一般是比如第18课有数据，第19课没数据，没有导出，但硬下载第19课数据，则实际是下载了第18课数据来填充第19课数据。
        #   可以通过在下载这里缓存已下载过文件名进行检查
        first_td = tr('tag:td', timeout=2)
        file1 = first_td.text if first_td else tr.text.split('\n', 1)[0]  # file1是网页上标记的名字，file2是下载后的路径和名字，两者不一定完全一致
        if file1 in self.exist_files:
            return None
        else:
            self.exist_files.add(file1)

        # 3 下载，并找到下载的文件名
        tab.wait(5)  # 等待一会，安全稳定些~ 防止下载到空文件夹等异常

        # + 修改：使用指定的下载目录，避免系统临时目录清理导致的问题
        from pathlib import Path
        download_dir = Path.home() / 'Downloads' / '_xlproject_temp_downloads'
        download_dir.mkdir(parents=True, exist_ok=True)

        file2 = XlPath(ele.click.to_download(str(download_dir), by_js=True).wait(show=False))
        return file2

    def export_user_list(self, search_name=None, download=True):
        """ 导出全部用户清单 """
        tab = self.tab
        tab.get('https://admin.xiaoe-tech.com/t/user_manage/index#/user_list/list')
        if search_name is None:
            tab('tag:span@@text()=重置', timeout=30).click()
        else:
            tab('tag:input@@placeholder=请输入昵称/备注名搜索').input(str(search_name), clear=True)  # 小批量数据测试用
            tab('tag:span@@text()=筛选').click()
        tab('tag:span@@text():导出列表').click()
        tab.wait(5)
        # 把能勾选的，没勾选的，全勾上
        for label in tab.eles('tag:label@@class=el-checkbox'):
            label('tag:span').click()
        # 241209周一20:03，不知道为啥，考勤六步这里"导出"老是经常容易找不到，所以就从5秒再多加一些时间
        tab.wait(10)
        # 这一步不知道为什么，每次检索速度会有点慢
        tab('tag:button@@text()=导出', timeout=20).click()

        if download:
            tab.wait(3)
            return self.download_last_file()

    def export_clockin_data(self, url, download=True, start_date=None, end_date=None):
        """ 导出指定的打卡数据文件 """
        cache_key = None
        if 'community_admin' in url:  # 禅宗打卡
            if start_date is None:  # 开始时间可以设置为一年前
                start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:  # 结束时间可以设置为今天
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
            cache_key = self._make_runtime_cache_key('clockin', url, start_date, end_date)
        elif 'diaryList' in url:  # 日历打卡
            cache_key = self._make_runtime_cache_key('clockin', url, start_date or '', end_date or '')

        if download and cache_key is not None:
            cached_file = self._restore_runtime_cached_file(cache_key, label=url)
            if cached_file is not self._CACHE_MISS:
                return cached_file

        # tab = self.browser.new_tab()
        tab = self.tab.get2(url)

        if 'community_admin' in url:  # 禅宗打卡
            tab.wait(3)

            # 直接操作 DOM 移除提示框组件（移除整个通知容器）
            tab.run_js('document.querySelector(".notify-wrap")?.remove()')
            tab.wait(0.5)

            # tab.run_js('location.reload()')  # 强制刷新页面
            tab.wait.eles_loaded('.tabs-pane', timeout=5)  # 确保父容器加载完成
            # 等待按钮变为可点击状态
            btn = tab('.tabs-pane')('tag:button@@class:ss-button@@text():导出')(
                'tag:span@@text():导出').wait.clickable()
            btn.scroll.to_see()
            # 点击前再次检查状态
            btn.click(timeout=4) if btn.states.is_clickable else print("按钮仍不可点击")

            # page('.tabs-pane')('.ss-button__content')('tag:span@@text():导出').click()
            # tab('.tabs-pane')('tag:button@@class:ss-button@@text():导出')('tag:span@@text():导出').click()
            tab.wait(1)
            tab('tag:input@@placeholder=开始日期@@class=el-range-input').click()
            tab('tag:input@@placeholder=开始日期@@class=el-input__inner').input(start_date, clear=True)
            tab('tag:input@@placeholder=开始时间@@class=el-input__inner').input('00:00', clear=True)
            tab('tag:input@@placeholder=结束日期@@class=el-input__inner').input(end_date, clear=True)
            tab('tag:input@@placeholder=结束时间@@class=el-input__inner').input('23:59', clear=True)
            tab('tag:div@@class=el-picker-panel__footer@@text():确定').click()
            tab.wait(1)
            tab('tag:button@@class=el-button el-picker-panel__link-btn '
                'el-button--default el-button--mini is-plain@@text():确定').click()  # 这是两个不同的"确认"按钮
            tab('tag:button@@class=el-button el-button--primary@@text():导出').click()
        elif 'diaryList' in url:  # 日历打卡（日历打卡多了一个字段"打卡天数"）
            try:
                page_text = tab.run_js('return document.body.innerText') or ''
            except Exception:
                page_text = ''
            if '暂无内容' in page_text or '无数据' in page_text:
                if cache_key is not None:
                    self._store_runtime_cached_file(cache_key, None)
                return None
            tab('tag:button@@text():导出动态').click(by_js=True)
            while True:
                ele = tab('tag:div@@role=dialog@@aria-label=导出数据')
                try:
                    ele('tag:button@@text():确认', timeout=5).click()
                    break
                except Exception as e:
                    logger.warning(format_exception(e, 3))
                    tab.wait(2)
            # 下载中心的任务列表有时不会立刻刷新出本次新建任务，等几秒再取文件能显著减少拿到旧导出的概率。
            tab.wait(10)
        # elif 'joinUser' in url:  # 作业打卡
        #     page('tag:button@@text():导出').click(by_js=True)
        #     while True:
        #         ele = page('tag:div@@class=ss-dialog@@aria-label=导出数据', timeout=2)
        #         try:
        #             ele('tag:button@@text()=确认', timeout=2).click()
        #             break
        #         except Exception as e:
        #             page.wait(2)
        #     page.wait(2)
        else:
            # tab.close()
            raise NotImplementedError

        if download:
            tab.wait(3)
            # tab.close()
            file = self.download_last_file()
            if cache_key is not None:
                file = self._store_runtime_cached_file(cache_key, file)
            return file

    def find_clockin_message(self, url):
        """ 获得打卡页面上的日期等详细数据

        目前先只获得shop2的数据，还没检查、考虑对shop1的兼容
        """
        # tab = self.browser.new_tab()
        tab = self.tab.get2(url)

        msg = {}

        # 找日期数据
        ms = []
        while not ms:
            text = tab('tag:div@@class=description').text
            ms = re.findall(r'\d{4}-\d{2}-\d{2}(?:[\s\d:]+)?', text)
            tab.wait(1)

        if ms:
            msg['start_date'] = ms[0].strip()
            if len(ms) > 1:
                msg['end_date'] = ms[1].strip()

        # 总共打卡天数
        m = re.search(r'共(\d+)天', text)
        if m:
            msg['days'] = int(m.group(1))

        # tab.close()
        return msg

    def click_filtered_items(self, lst):  # todo：临时加的_20240914
        """
        通过传递的页面对象和筛选列表来点击符合条件的元素。

        :param page: ChromiumPage 对象，表示要操作的页面
        :param lst: 筛选条件列表，例如 ["6期7组", "6期6组", "6期5组", "6期4组", "6期3组", "6期2组", "6期1组"]
        """
        tab = self.tab
        # 点击筛选输入框
        tab.actions.click('tag:input@@autocomplete=off@@placeholder=请输入内容')
        ele1 = tab.ele('tag:ul@@class=ss-scrollbar__view s374 ss-select-dropdown__list')
        sons = ele1.children('tag:li')

        # 方式1：通过判断 li 标签中是否包含指定文本
        # for son in sons:
        #     if son.child('tag:div@@text():6期'):  # 判断是否符合条件
        #         span_element = son.child('tag:span')
        #         if span_element:
        #             span_element.scroll.to_see()  # 滚动到可见
        #             span_element.click()  # 点击目标元素
        #         else:
        #             print("No span element found.")  # 如果没有找到 span，输出提示

        # 方式2：遍历 lst 中的筛选条件并点击
        for value in lst:
            ele2 = tab.ele(f'x://div/ul/li/div/span[text()="{value}"]')  # 定位每个需要点击的元素
            if ele2:
                ele2.scroll.to_see()  # 滚动到元素可见位置
                tab.set.scroll.wait_complete(on_off=True)  # 等待滚动完成
                ele2.click()  # 点击目标元素

        tab.actions.click('tag:span@@text():筛选')  # 点击筛选按钮并等待
        # 启动监听，等待分页加载完毕
        tab.listen.start('tag:ul@@unselectable=unselectable@@class=ant-pagination ant-table-pagination')
        tab.wait(3)

    def export_lesson_data(self, row, download=True):
        """ 导出指定课程的数据

        :param row: 一般是从数据库lesson_table获取的一条数据
            也可以只输入lesson_id2的值，即课次url，或者课次的小鹅通id
        """
        # 1 参数校验
        if isinstance(row, dict):
            pass
        elif isinstance(row, str):
            row = {'lesson_id2': row}
        else:
            raise TypeError

        shop_id = row.get('shop_id') if isinstance(row, dict) else None
        cache_key = self._make_runtime_cache_key('lesson', row['lesson_id2'], shop_id=shop_id)
        if download:
            cached_file = self._restore_runtime_cached_file(cache_key, label=row['lesson_id2'])
            if cached_file is not self._CACHE_MISS:
                return cached_file

        # 2 目前有3类情况的课次分类处理
        url = row['lesson_id2']
        work_url = url
        wait_seconds = 3
        if not url.startswith('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page') \
                and not url.startswith('https://admin.xiaoe-tech.com/t/course/camp_pro/course_detail_page'):
            work_url = f'https://admin.xiaoe-tech.com/t/live_management#/userOperation?id={url}&tabName=UserManage'
            wait_seconds = 5
        elif url.startswith('https://admin.xiaoe-tech.com/t/course/camp_pro/course_detail_page'):
            wait_seconds = 5

        with self.临时工作标签页(url=work_url, wait_seconds=wait_seconds) as tab:
            if shop_id:
                target_shop, _ = self._标准化店铺(shop_id)
                for _ in range(2):
                    current_shop = self._当前店铺名(timeout=0.3)
                    if tab.url.startswith('https://admin.xiaoe-tech.com/t/account/muti_index#/chooseShop') \
                            or tab.url.startswith('https://admin.xiaoe-tech.com/t/login#') \
                            or (current_shop and current_shop != target_shop):
                        logger.warning(f'课次页打开后店铺上下文异常，准备重进：lesson={row.get("lesson_name", row["lesson_id2"])} '
                                       f'target_shop={target_shop} url={tab.url} current_shop={current_shop or "未知"}')
                        self.switch_shop(shop_id)
                        tab.get(work_url)
                        tab.wait(wait_seconds)
                        continue
                    break

            if url.startswith('https://admin.xiaoe-tech.com/t/community_admin/miniCommunity#/course_detail_page'):
                # 空表，不用处理
                trs = tab.eles('t:table@@class=ant-table-fixed')[1]('t:tbody').eles('t:tr')
                if not trs:
                    self._store_runtime_cached_file(cache_key, None)
                    return

                # 正常导出
                tab('tag:button@@class=ant-btn@@text():导 出').click(by_js=True)
                tab.wait(3)
                tab('tag:button@@class=ant-btn ant-btn-primary@@text():导 出').click()
            elif url.startswith('https://admin.xiaoe-tech.com/t/course/camp_pro/course_detail_page'):
                # 关掉下载提示
                for ele in tab.eles('t:i@@class=sense-icon-close'):
                    ele.click(by_js=True)

                # 该课完成标记，例如：'(未开始9人；进行中3人；已完成 9人)'
                status_ele = tab('t:div@@class=num-box-item@@text():未开始', timeout=3)
                status = status_ele.text if status_ele else ''
                nums = list(map(int, re.findall(r'\d+', status)))
                if len(nums) >= 3:
                    _, b, c = nums[:3]
                    if b + c == 0:  # 这个课目前还没有人学，可以跳过
                        self._store_runtime_cached_file(cache_key, None)
                        return
                else:
                    logger.warning(f'闯关课状态文本解析失败，继续尝试导出：lesson={row.get("lesson_name", row["lesson_id2"])} '
                                   f'status={status!r} url={tab.url}')

                tab('t:button@@text():导 出').click()
                tab.wait(5)
                tab('t:div@@class=ant-modal-content')('t:button@@text():导 出').click()
            else:
                # 遇到空数据表，不用处理
                trs = tab('t:table@@class:ss-table__body')('t:tbody').eles('t:tr')
                if not trs:
                    self._store_runtime_cached_file(cache_key, None)
                    return

                # 正常导出数据
                tab('tag:button@@text()=导出列表').click()
                tab.wait(5)
                for i in range(10):
                    try:
                        tab('tag:button@@text():导出 ').click(by_js=True)
                        break
                    except Exception as e:
                        wechat_logger.warning(format_exception(e, 3))
                        if i == 9:
                            raise e
                        tab.wait(2)

            if download:
                tab.wait(5)
                file = self.download_last_file()
                if not file:
                    self._store_runtime_cached_file(cache_key, None)
                    return None
                # bug: 这个要考虑后缀.csv等的影响，以及dp自带的下载，逻辑有些不同
                m = re.search(r'\(\d+\)$', file.stem)
                if m:
                    msg = (f'尝试导出课程数据：{url}，但是获得文件：{file.name}，从文件名看，大概率是出问题了，'
                           '比如这堂课数据是空的没有导出文件，而自动继续下载浏览器记录的上一次的课程数据文件，所以有文件名重名问题'
                           '或者小鹅通不稳定，该课次数据未正常导出。为了避免意外错误，本处直接返回None。')
                    wechat_logger.warning(msg)
                    self._store_runtime_cached_file(cache_key, None)
                    return
                file = self._store_runtime_cached_file(cache_key, file)
                return file

    def __2_从网页获得某些信息(self):
        pass

    def search_lesson_links(self, name, live_status=None, maxn=-1):
        """
        :param name: 课程名称
        :param str live_status: 直播状态：全部状态，未开始，直播中，已结束
        :param maxn: 获取的最多条目数
        :return: 使用 yield 逐条返回查询结果
        """
        tab = self.switch_shop()

        # 1 配置要访问的 url
        # 小鹅通直播列表页是 hash 路由，查询参数必须放在 '#/list' 后面。
        params = {
            'page_size': 10,
            'page': 1,
            'search_owner': 0,
        }
        if name:
            params['search_content'] = name

        if isinstance(live_status, str):
            live_status = live_status.strip()
        live_status_value = {
            '未开始': '0',
            '直播中': '1',
            '已结束': '2',
        }.get(live_status)
        if live_status_value is not None:
            params['search_aliveState_type'] = live_status_value

        url = f"https://admin.xiaoe-tech.com/t/live#/list?{urlencode(params)}"
        tab.get(url)

        count = 0
        # 2 遍历所有页
        while True:
            tbody = tab('tag:tbody')
            # 如果没有相应的数据
            if tbody('没有相应的数据'):
                break

            for tr in tbody.eles('tag:tr'):
                tab2 = tr('t:button@@text():管理').click.for_new_tab(by_js=True)

                title = tr('.title title-hover ss-popover__reference').text
                course_id = re.search(r'id=(.+?)(?=$|&)', tab2.url).group(1)
                row = {
                    'lesson_name': title,
                    'lesson_id': course_id,
                }

                # 使用 yield 返回结果
                yield row
                count += 1

                tab2.close()

                if maxn > 0 and count >= maxn:
                    # 已达到最大条目数
                    return

            # 判断是不是最后一页了
            # 下一页的按钮
            next_page_li = tab('t:ul@@class:ss-pagination').eles('t:li')[-2]
            # 如果最后一个 li 是当前页码，表示没有下一页
            if 'ss-pagination-item__disabled' in next_page_li.attr('class'):
                break
            else:
                next_page_li.click()

    def get_leeson_playback_settings(self, lesson_id2):
        """ 设置课次的回放配置情况
        通过浏览网页，查看课程的回放等相关配置，确定每个课所需的更新时间节点

        241217周二21:13，删除了对update_times的配置
            其实无论任何情况，如果有需要，保底每天监控就行了，都能兼容的
            并不是非要用update_times的机制
            或者特殊的课程名，下游任务自己写next_update的更新逻辑
        """
        row = {}

        # 1 访问课次网址

        # 这里有个比较高效的方法，是判断是否有现成的这个链接的tab可以直接服用
        #   不过这样其实也有危险，真的是其他并行在跑的任务，去抢占别人资源就冲突了~
        #   所以现在这样如果配合"search_lesson_links"使用，虽然会出现一个课程出现两个tab，但影响并不大
        tab = self.browser.new_tab()
        tab.get(f'https://admin.xiaoe-tech.com/t/live#/detail?id={lesson_id2}&tab=playbackSettings')

        tab('tag:div@@class=config-title@@text()=回放有效期：')  # 开始监听，出现这个元素在执行下述操作

        tab.listen.start('tag:div@@class=time')
        # tab.listen.start('tag:input@@placeholder=请选择日期和时间')  # 这个可能不存在
        tab.wait(5)

        # 课程标题
        row['lesson_name'] = tab('t:div@@class=title-text').text

        # 2 计算开始时间
        text = tab('直播时间').text
        dts = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', text)
        # 这里会获得两个时间：dts[0]直播开始时间，dts[1]直播结束时间
        assert len(dts) == 2
        row['start_date'] = dts[0]
        d0 = datetime.datetime.strptime(dts[1], '%Y-%m-%d %H:%M:%S')
        update_times = [d0]

        # 3 计算课程结束时间
        ele = tab('t:input@@placeholder=请选择日期和时间', timeout=1)  # todo："请选择截止日期"   变成：请选择日期和时间
        if ele:  # 有回放设置
            while True:
                if ele.value:
                    break
                time.sleep(1)
            # row['end_date'] = ele.value + ' 23:59:59'
            row['end_date'] = ele.value if len(ele.value) == 19 else (
                ele.value + ' 23:59:59' if len(ele.value) == 10 else ValueError("ele.value 格式不正确。"))
            # 在间隔范围内，每天监控一次
            # d1 = datetime.datetime.strptime(row['end_date'], '%Y-%m-%d %H:%M:%S')
            # while True:  # 理论上在课程回放结束前，应该每天在课程结束24小时后更新一次
            #     d0 += datetime.timedelta(days=1)
            #     if d0 > d1:
            #         break
            #     update_times.append(d0)
        else:  # 找不到选项，则表示很可能设置了"永久"而不是限时
            # row['end_date'] = dts[1]
            row['end_date'] = ''  # 永久情况，这个参数可以置空不写

        tab.close()

        # 4 每个要更新的时间节点
        # row['update_times'] = [x.strftime('%Y-%m-%d %H:%M:%S') for x in update_times]
        # row['next_update'] = update_times[0]  # 更新时间的初始配置
        row['next_update'] = dts[1]  # 更新时间的初始配置

        d1 = datetime.datetime.strptime(row['start_date'], '%Y-%m-%d %H:%M:%S')
        d2 = update_times[0]
        row['video_duration'] = (d2 - d1).seconds

        return row

    def 爬虫获得禅宗课程目录(self, 课程目录url, *, start_name=None, stop_name=None):
        """ 禅宗爬取课程目录

        :param 课程目录url: 课程目录管理页面
            在类似这样检索路径：内容/圈子管理/4阶/管理/课程/7-8期四阶/管理
        :param start_name: 遇到课程名称中包含什么词才开始爬取
        :param stop_name: 遇到课程名称中包含什么词就停止
        :return:
        """
        # 1 登录课程页面
        self.switch_shop('宗门学府')
        tab = self.tab
        tab.get(课程目录url)

        # 2 爬取课程链接等数据
        courses = {}
        group = tab('t:div@@class:ss-checkbox-group')
        catalogue_lists = group.eles('t:div@@id=catalogue_list')
        start_flag = not start_name  # 如果没有设置 start_name，直接开始爬取

        for chapter in catalogue_lists:
            # 1 获取每周标题
            chapter_div = chapter('t:span@@class=chapter-title-name')
            chapter_name = chapter_div.text
            if stop_name and stop_name in chapter_name:  # 遇到停止词
                break

            # 检查是否到达开始爬取的章节
            if not start_flag and start_name in chapter_name:
                start_flag = True

            if not start_flag:
                continue

            # 2 检查每周子课次清单
            task_cfgs = []
            tasks = chapter('t:div@@aria-label=checkbox-group').eles('t:div@@class=task')
            if not tasks:
                chapter_div.click()  # 确保展开该章节
                tasks = chapter('t:div@@aria-label=checkbox-group').eles('t:div@@class=task')

            # 3 保存课次的名称、链接
            for task in tasks:
                name = task('t:span@@class:ss-popover__reference').text
                tab2 = task('t:span@@text()=数据').click.for_new_tab()
                url = tab2.url
                tab2.close()
                task_cfgs.append([name, url])

            courses[chapter_name] = task_cfgs
            chapter_div.click()  # 折叠该章节

        return courses

    def 查找用户(self, 昵称='', 手机号='', 课程标准名='', 课程商品名=''):
        # 1 登录查找用户页面
        tab = self.tab
        target_url = 'https://admin.xiaoe-tech.com/t/user_manage/index#/user_list/list'
        if not str(getattr(tab, 'url', '')).startswith(target_url):
            tab.get(target_url)

        from .db import KqDb
        昵称, 手机号 = KqDb.标准化昵称手机号参数(昵称, 手机号)

        # 2 先按手机号检查
        def 采集数据(模式, 内容):
            # 1 打开筛选器
            筛选器 = tab('t:form@@class=el-form user_list-search-form')
            筛选项 = 筛选器('t:div@@class=user_list-search-content').eles('t:div@@class=el-form-item')
            关键词筛选 = 筛选项[0]

            def 获取当前筛选模式():
                try:
                    模式输入框 = 关键词筛选('t:input@@class=el-input__inner@@readonly', timeout=1)
                except Exception:
                    return ''
                if not 模式输入框:
                    return ''
                return (getattr(模式输入框, 'value', '') or 模式输入框.attr('value') or 模式输入框.text or '').strip()

            def 点击可见筛选模式选项():
                选项列表 = tab.eles(f't:li@@text()={模式}')
                for 选项 in 选项列表:
                    cur = 选项
                    for _ in range(6):
                        cur = cur.parent()
                        if not cur:
                            break
                        if 'el-select-dropdown el-popper' in (cur.attr('class') or ''):
                            style = cur.attr('style') or ''
                            if 'display: none' not in style:
                                选项.click(by_js=True)
                                return True
                            break
                if 选项列表:
                    选项列表[-1].click(by_js=True)
                    return True
                return False

            # 2 查找筛选
            if 获取当前筛选模式() != 模式:
                关键词筛选('t:span@@class=el-input__suffix').click()  # 弹出下拉列表
                time.sleep(1)
                if not 点击可见筛选模式选项():
                    raise DrissionPage.errors.ElementNotFoundError(
                        METHOD='visible dropdown option',
                        ARGS={'mode': 模式}
                    )
                time.sleep(1)
            关键词筛选('t:input@@type=text@@class=el-input__inner@!readonly').input(内容, clear=True)
            time.sleep(3)

            筛选器('t:button@@data-sensors=用户管理_用户列表_筛选').click(by_js=True)
            time.sleep(3)
            表格区域 = tab('t:div@@class:ss-multi-operate__table-wrap')
            字段行 = 表格区域('t:div@@class:ss-table__header-wrapper')
            字段名列表 = [x.text for x in 字段行.eles('t:th')]
            # ['', '用户', '来源渠道', '姓名', '账户绑定手机号', '最近采集手机号', '注册时间', '用户ID', '微信号', '最近访问时间', '操作', '']
            数据块 = 表格区域('t:div@@class:ss-table__body-wrapper')
            匹配条目列表 = 数据块.eles('t:tr')

            # 3 转成df
            数据列表 = []
            for 匹配条目 in 匹配条目列表:
                # 有几个字段会有干扰内容，要split后取最后一段
                字段值 = [x.text.split('\n')[-1] for x in 匹配条目.eles('t:td')]
                数据字典 = dict(zip(字段名列表, 字段值))
                数据列表.append(数据字典)
            df = pd.DataFrame(数据列表, columns=字段名列表)

            return df

        # 暂时只做标准手机号检查，且候选项只有一条的时候才填写
        if 手机号:
            df = 采集数据('手机号', 手机号[0])
            if len(df) == 1:
                return df['用户ID'][0]

        return
