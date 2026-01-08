#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Author : 陈坤泽
# @Email  : 877362867@qq.com
# @Date   : 2022/02/25 17:54

from pyxllib.prog.lazyimport import lazy_import

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lazy_import('from tqdm import tqdm')

try:
    import numpy as np
except ModuleNotFoundError:
    np = lazy_import('numpy')

try:
    import cv2
except ModuleNotFoundError:
    cv2 = lazy_import('cv2', 'opencv-python')

try:
    from moviepy.editor import VideoFileClip, cvsecs
except ModuleNotFoundError:
    VideoFileClip = lazy_import('from moviepy.editor import VideoFileClip', 'moviepy')
    cvsecs = lazy_import('from moviepy.editor import cvsecs', 'moviepy')

from pyxllib.prog.pupil import inject_members
from pyxllib.file.specialist import XlPath
from pyxllib.cv.imhash import get_init_hash, phash
from pyxllib.cv.xlcvlib import xlcv


class XlVideoFileClip(VideoFileClip):

    def get_frame2(self, time_point, *, scale=None):
        """ 官方获得的图片通道是RGB，但是cv2处理的统一规则是BGR，要转换过来

        :param scale: 获取图片后是否要按统一的比例再缩放一下
        """
        frame = self.get_frame(time_point)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if scale:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        return frame

    def get_frames(self, time_points=None, *, interval_second=0.1, cur_hash=None, head_frame=None, scale=None,
                   filter_mode=2, print_mode=True):
        """ 同时获得多帧图片

        :param time_points: 类list对象，元素是时间点（可以是字符串格式，也可以是数值秒数）
            常见生成方法参考：np.array(0, clip.duration, 0.1)
        :param filter_mode: 过滤强度
            因为times如果取的非常密集，比如每0.1秒取一帧，会有很大概率是重复、相近的图
            此时可以用这个参数控制误差在多大以内的图会保留
            使用phash判断帧之间相似度，只要误差在该设定阈值内，都会被跳过
            可以设置0，不过这样过滤程度是最低的
            也可以输入-1，表示完全不过滤，不计算phash。当这种情况也不太必要用这个函数了，可以直接普通循环。
            也可以故意设置的特别大（phash最大差距是64），那这样相当于差异很大的也过滤掉了
        :param head_frame: 提供一张初始图片，主要是配合filter_mode用于去重的
        :param cur_hash: 类似head_frame作用，但直接提供hash值
        :return: 使用yield机制，防止图片数据太多，内存爆炸了
            当filter_mode>=0时，返回 时间点time_point和对应的图片im
        """
        if cur_hash is None:
            if head_frame is None:
                cur_hash = get_init_hash()
            else:
                cur_hash = phash(head_frame)

        if time_points is None:
            time_points = np.arange(0, self.duration, interval_second)

        for time_point in tqdm(time_points, disable=not print_mode):
            im = self.get_frame2(time_point, scale=scale)

            if filter_mode >= 0:
                last_hash, cur_hash = cur_hash, phash(im)
                if cur_hash - last_hash <= filter_mode:
                    continue
                yield time_point, im
            else:
                yield im

    def save_frames(self, out_dir, time_points=None, interval_second=0.1, **kwargs):
        """ 跟get_frames差不多，多了一步自动存储图片文件到目录里

        这个功能是比较定制化的，不是那么泛用，但会有很多相关类似的操作需求，可以参考这里的代码实现
        """
        # 1 time_points
        if time_points is None:
            time_points = np.arange(0, self.duration, interval_second)
        out_dir = XlPath(out_dir)

        # 2 按时间点取图片，保存文件
        for time_point, frame in self.get_frames(time_points, **kwargs):
            tt = int(time_point * 10)
            m, s, ms = tt // 600, (tt // 10) % 60, tt % 10
            xlcv.write(frame, out_dir / f'{m:03}_{s:02}.{ms}.jpg')

    def join_subtitles_image(self, time_points, ltrb_pos=None, *,
                             crop_first_frame=False,
                             filter_mode=2):
        """ 生成字幕拼图

        :param time_points: 在哪些时间点截图
        :param ltrb_pos: 字幕所在位置，没有输入则默认全部全图拼接，
            一般指定上下就行了，即 [None, 640, None, 700]
        :param crop_first_frame:
            False，保留第一帧的完整性
            True，第一帧也只裁剪字幕部分
        :param filter_mode: 对于给出的时间点图片，去除相邻相同的图片
        :return:

        参考用法：
            from pyxllib.file.movielib import VideoFileClip
            clip = VideoFileClip(str(Paths.videos / '觉观01.mp4'))  # 必须要转str类型
            im = clip.join_subtitles_image(np.arange(30, 40, 0.1), [None, 640, None, 695])
            xlcv.write(im, Paths.videos / 'test.jpg')
        """
        # 1 如果有左右裁剪要提前处理
        x1, y1, x2, y2 = ltrb_pos
        if x1 or x2:
            clip = self.crop(x1=x1, x2=x2)
        else:
            clip = self

        # 2 第1帧是否要保留全图
        frame_list = []
        if crop_first_frame:
            head_frame = None
        else:
            time_points = tuple(time_points)
            head_frame = clip.get_frame2(time_points[0])
            frame_list.append(head_frame[:y2])
            head_frame = head_frame[y1:y2]
            time_points = time_points[1:]

        # 3 裁剪字幕区域
        clip = clip.crop(y1=y1, y2=y2)
        for time_point, frame in clip.get_frames(time_points, filter_mode=filter_mode, head_frame=head_frame):
            frame_list.append(frame)

        # 4 拼接完整图
        im = xlcv.concat(frame_list, pad=0)
        return im


inject_members(XlVideoFileClip, VideoFileClip)


def clip_video(input_file, output_file, start_time, end_time):
    start_time = cvsecs(start_time)
    end_time = cvsecs(end_time)
    clip = VideoFileClip(input_file).subclip(start_time, end_time)
    clip.write_videofile(output_file)
    clip.reader.close()
