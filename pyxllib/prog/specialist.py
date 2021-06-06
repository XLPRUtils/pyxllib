#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2021/06/06 11:16

import concurrent.futures


def mtqdm(func, iterable, *args, max_workers=1, **kwargs):
    """ 对tqdm的封装，增加了多线程的支持

    这里名称前缀多出的m有multi的意思

    :param max_workers: 默认是单线程，改成None会自动变为多线程
        或者可以自己指定线程数
    :param smoothing: tqdm官方默认值是0.3
        这里关掉指数移动平均，直接计算整体平均速度
        因为对我个人来说，大部分时候需要严谨地分析性能，得到整体平均速度，而不是预估当前速度
    :param mininterval: 官方默认值是0.1，表示显示更新间隔秒数
        这里不用那么频繁，每秒更新就行了~~

    整体功能类似Iterate
    """
    from tqdm import tqdm

    # 0 个人习惯参数
    kwargs['smoothing'] = kwargs.get('smoothing', 0)
    kwargs['mininterval'] = kwargs.get('mininterval', 1)

    if max_workers == 1:
        # 1 如果只用一个线程，则不使用concurrent.futures.ThreadPoolExecutor，能加速
        for x in tqdm(iterable, *args, **kwargs):
            func(x)
    else:
        # 2 默认的多线程运行机制，出错是不会暂停的；这里对原函数功能进行封装，增加报错功能
        error = False

        def wrap_func(x):
            nonlocal error
            try:
                func(x)
            except Exception as e:
                error = e

        # 3 多线程和进度条功能的结合
        executor = concurrent.futures.ThreadPoolExecutor(max_workers)
        for x in tqdm(iterable, *args, **kwargs):
            while executor._work_queue.qsize():
                pass
            executor.submit(wrap_func, x)
            if error:
                raise error

        executor.shutdown()
