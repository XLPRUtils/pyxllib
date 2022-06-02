#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/05/14 10:02

import base64
from collections import defaultdict, Counter
import datetime
import io
import json
import sys

import PIL.Image
import pandas as pd
from tqdm import tqdm

from pyxllib.prog.pupil import get_hostname
from pyxllib.data.sqlite import Connection
from pyxllib.file.specialist import get_etag, XlPath


class XlprDb(Connection):
    """ xlpr统一集中管理的一个数据库 """

    def __init__(self, dbfile=None, *args, **kwargs):
        if dbfile is None:
            dbfile = XlPath.userdir() / '.xlpr/xlpr.db'

        self.directory = XlPath(dbfile).parent
        self.dbfile = dbfile
        super().__init__(dbfile, *args, **kwargs)

        # 在十卡环境当前时间要加8小时
        self.in_tp10 = get_hostname() == 'prui'

    def _get_time(self):
        """ 获得当前时间 """
        d = datetime.datetime.today().today()
        if self.in_tp10:
            d += datetime.timedelta(hours=8)
        return d.strftime('%Y-%m-%d %H:%M:%S')

    def _render_chart(self, chart):
        from pyxllib.debug.specialist import browser

        # if sys.platform == 'win32':
        #     browser(chart)

        file = XlPath.tempfile(suffix='.html')
        chart.render(path=str(file))
        res = file.read_text()
        file.delete()
        return res

    def __1_jpg_images(self):
        """ 图片相关功能 """

    def init_jpgimages_table(self):
        """ 存储图片数据的表

        其实把图片存储到sql中很不规范，最好是用OSS
            但是相关框架太麻烦，还要花我钱、流量费
            我就暴力存储本地了
        而存储本地为了省空间，我又把所有图片格式转成jpg，这样整体能少消耗些空间

        etag: 来自  get_etag(xlcv.to_buffer(im))

        + phash？
        """
        if self.has_table('jpgimages'):
            return
        cols = ['etag text', 'filesize integer', 'height integer', 'width integer', 'base64_content text']
        cols = ', '.join(cols)
        self.execute(f'CREATE TABLE jpgimages ({cols}, PRIMARY KEY (etag))')
        self.commit()

    def insert_jpgimage_from_buffer(self, buffer, *, etag=None, **kwargs):
        """
        为了运算效率考虑，除了etag需要用于去重，是必填字段
        其他字段默认先不填充计算
        """
        # 1 已经有的不重复存储
        if etag is None:
            etag = get_etag(buffer)

        res = self.execute('SELECT etag FROM jpgimages WHERE etag=?', (etag,)).fetchone()
        if res:
            return

        # 2 没有的图，做个备份
        kwargs['etag'] = etag
        kwargs['base64_content'] = base64.b64encode(buffer)
        kwargs['filesize'] = len(buffer)  # 这个计算成本比较低

        # 本来图片尺寸等想惰性计算，额外写功能再更新的，但发现这个操作似乎挺快的，基本不占性能消耗~~
        im = PIL.Image.open(io.BytesIO(buffer))
        kwargs['height'] = im.height
        kwargs['width'] = im.width

        self.insert('jpgimages', kwargs)
        self.commit()

    def extract_jpgimages(self, size_threshold=50 * 1024):
        """ 将数据库中较大的图片数据，提取存储到文件中

        :param size_threshold: 尺寸大于阈值的数据，备份到子目录data里
            默认以50kb为阈值，用20%的图片能清出80%的空间
        """
        from pyxllib.cv.xlcvlib import xlcv
        for x in tqdm(self.exec_nametuple(f'SELECT * FROM jpgimages WHERE filesize>{size_threshold}')):
            if x['base64_content']:
                im = xlcv.read_from_buffer(x['base64_content'], b64decode=True)
                xlcv.write(im, self.directory / 'data' / (x['etag'] + '.jpg'))
            self.update('jpgimages', {'base64_content': None}, {'etag': x['etag']})
            self.commit()
        self.execute('vacuum')  # 执行这句才会重新整理sqlite文件空间
        self.commit()

    def __2_api(self):
        """ api调用记录相关功能 """

    def init_aipocr_table(self):
        """ 存储百度识别接口结果

        1、已处理过的图，可以不用重复调用接口
        2、某些图可以手动修改标注，以后再调用接口能获得更好的效果
        """
        if self.has_table('aipocr'):
            return
        cols = ['mode text', 'im_etag text', 'options text', 'result text', 'update_time text']
        cols = ', '.join(cols)
        self.execute(f'CREATE TABLE aipocr ({cols}, PRIMARY KEY (mode, im_etag, options))')
        self.commit()

    def get_aipocr_result(self, mode, im_etag, options):
        """

        加上日期限制？比如一年前的不获取，强制让其重调接口更新？
            思考：不推荐这么干，可以从数据入手，一年前的，手动清理备份到旧数据
        """
        x = self.execute('SELECT result FROM aipocr WHERE mode=? AND im_etag=? AND options=?',
                         (mode, im_etag, json.dumps(options, ensure_ascii=False))).fetchone()
        if x:
            return json.loads(x[0])

    def insert_aipocr_record(self, mode, im_etag, options, result, *, if_exists='IGNORE', **kwargs):
        """ 往数据库记录当前操作内容

        :param if_exists: 如果已存在，是否强制替换掉
        :return:
            True, 更新成功
            False, 未更新
        """
        kw = {'mode': mode,
              'im_etag': im_etag,
              'options': options,
              'result': result,
              'update_time': self._get_time()}
        kw.update(kwargs)
        self.insert('aipocr', kw, if_exists=if_exists)
        self.commit()

    def init_xlprapi_table(self):
        """ 开api服务用，所有人的调用记录
        """
        if self.has_table('xlprapi'):
            return
        cols = ['remote_addr text', 'token text', 'route text',
                'request_json text',  # 整个请求的etag标记，注意对大的数据进行精简
                'update_time text']
        cols = ', '.join(cols)
        self.execute(f'CREATE TABLE xlprapi ({cols})')
        self.commit()

    def insert_xlprapi_record(self, request, *, if_exists='IGNORE', **kwargs):
        d = {}
        for k, v in request.json.items():
            s = str(v)
            if len(s) > 100:
                n = len(s)
                d[k] = f'...{s[n // 2:n // 2 + 6]}...'  # 太长的，缩略存储
            else:
                d[k] = v

        kw = {'remote_addr': request.remote_addr,
              'token': request.headers.get('Token', None),
              'route': '/'.join(request.base_url.split('/')[3:]),
              'request_json': d,
              'update_time': self._get_time()}
        kw.update(kwargs)
        self.insert('xlprapi', kw, if_exists=if_exists)
        self.commit()

    def __3_检查数据库使用情况(self):
        pass

    def check_database_useage(self):
        """ 查看数据库一些统计情况 """
        from pyxllib.text.nestenv import NestEnv
        from pyxllib.text.xmllib import render_echart_html

        def check_api_useage():
            """ 检查接口调用量 """
            from pyxllib.data.echarts import Line

            # 1 调用量计算
            ct = defaultdict(Counter)
            for x in self.exec_nametuple('SELECT route, request_json, update_time FROM xlprapi'):
                d = datetime.datetime.fromisoformat(x['update_time']).date().toordinal()
                ct['total'][d] += 1
                if 'basicGeneral' in x['request_json']:
                    ct['basicGeneral'][d] += 1
                if x['route'] == 'api/aipocr':
                    ct['aipocr'][d] += 1

            # + 辅助函数
            min_day, max_day = min(ct['total'].keys()), max(ct['total'].keys())

            def to_list(ct):
                """ Counter转日期清单
                因为有的天数可能出现次数0，需要补充些特殊操作
                """
                return [(datetime.date.fromordinal(i), ct.get(i, 0)) for i in range(min_day, max_day + 1)]

            # 2 画图表
            chart = Line()
            chart.add_series('total', to_list(ct['total']), label={'show': True}, areaStyle={})
            chart.add_series('aipocr', to_list(ct['aipocr']), areaStyle={})
            chart.add_series('basicGeneral', to_list(ct['basicGeneral']), areaStyle={})

            chart.options['xAxis'][0].update({'min': datetime.date.fromordinal(min_day), 'type': 'time'})

            # 3 展示
            return self._render_chart(chart)

        def check_jpgimages_size():
            """ 检查图片文件尺寸所占比例

            请在windows平台运行，获得可视化最佳体验
            """
            from pyxllib.algo.stat import pareto_accumulate
            from pyxllib.data.echarts import Line

            filesizes = list(self.select_col('jpgimages', 'filesize'))
            recordsizes = [len(x) for x in self.select_col('aipocr', 'result')]

            x = Line()
            pts, labels = pareto_accumulate(filesizes, 0.1)
            x.add_series('图片', pts, labels=labels, label={'position': 'right'})
            x.add_series('json', pareto_accumulate(recordsizes, 0.1)[0])

            return self._render_chart(x)

        res = ['1、每个token调用量']
        ls = self.exec_dict('SELECT token, COUNT(*) cnt FROM xlprapi GROUP BY token ORDER BY cnt DESC').fetchall()
        df = pd.DataFrame.from_records(ls)
        res.append(df.to_html())

        res.append('<br/>2、每日API调用量')
        res.append(NestEnv(check_api_useage()).xmltag('body', inner=True).string())
        res.append('<br/>3、图片存储情况')
        res.append(NestEnv(check_jpgimages_size()).xmltag('body', inner=True).string())
        res.append('<br/>当前数据库大小：' + self.dbfile.size(human_readable=True))

        res = '<br/>'.join(res)

        return render_echart_html(title='check_user_useage', body=res)

    def recently_api_record(self):
        """ 最近调用的几条API记录 """
        # 其实这里没有echart表格，但反正都通用的，就直接用了
        from pyxllib.text.xmllib import render_echart_html

        import pandas as pd

        res = ['1、最近10条调用记录 xlprapi']
        sql = 'SELECT * FROM (SELECT * FROM xlprapi ORDER BY update_time DESC LIMIT 10)'
        df = pd.DataFrame.from_dict(self.exec_dict(sql).fetchall())
        res.append('<br/>' + df.to_html())

        res.append('2、最近10条识别内容 aipocr')
        sql = 'SELECT * FROM (SELECT * FROM aipocr ORDER BY update_time DESC LIMIT 10)'
        df = pd.DataFrame.from_dict(self.exec_dict(sql).fetchall())
        res.append('<br/>' + df.to_html())

        return render_echart_html(title='recently_api_record', body='<br/>'.join(res))


if __name__ == '__main__':
    from pyxllib.debug.specialist import browser

    db = XlprDb(check_same_thread=False)
    browser.html(db.check_database_useage())
