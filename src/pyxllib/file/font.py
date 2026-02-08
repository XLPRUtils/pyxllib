import os
import platform

from loguru import logger


def get_chinese_font_path(priority='heiti'):
    """
    跨平台获取一个可用的中文字体文件路径。
    返回: 字体文件的绝对路径 (str)
    如果未找到，返回 None
    """
    system = platform.system()

    # 定义不同系统下的常见中文字体文件名 (按优先级排序)
    # 优先找黑体/无衬线体，通用性更好
    font_filenames = []

    if system == 'Windows':
        font_filenames = [
            'msyh.ttf',  # 微软雅黑 (Win7+)
            'simhei.ttf',  # 黑体
            'simsun.ttc',  # 宋体
            'arialuni.ttf',  # Arial Unicode MS
        ]
        font_dirs = [r'C:\Windows\Fonts']

    elif system == 'Darwin':  # macOS
        font_filenames = [
            'PingFang.ttc',  # 苹方 (Mac首选)
            'Hiragino Sans GB.ttc',  # 冬青黑体
            'STHeiti Light.ttc',  # 华文细黑
            'Arial Unicode.ttf',  # 通用
        ]
        font_dirs = ['/System/Library/Fonts', '/Library/Fonts', os.path.expanduser('~/Library/Fonts')]

    else:  # Linux
        # Linux 字体情况复杂，优先找 Noto (Google) 或文泉驿
        font_filenames = [
            'NotoSansCJK-Regular.ttc',
            'NotoSansCJK-Bold.ttc',
            'wqy-zenhei.ttc',  # 文泉驿正黑
            'wqy-microhei.ttc',  # 文泉驿微米黑
        ]
        # 常见的 Linux 字体安装目录
        font_dirs = [
            '/usr/share/fonts',
            '/usr/share/fonts/truetype/noto',
            '/usr/share/fonts/truetype/wqy',
            os.path.expanduser('~/.local/share/fonts'),
        ]

    # 核心逻辑：双重循环查找文件是否存在
    for folder in font_dirs:
        # Linux下有时需要递归查找，这里简化处理，只查指定目录
        if not os.path.exists(folder):
            continue

        for filename in font_filenames:
            full_path = os.path.join(folder, filename)
            if os.path.exists(full_path):
                print(f'✅ 成功找到字体: {filename}')
                return full_path

    logger.warning('⚠️ 警告: 未在系统中找到常见中文字体，可能导致中文乱码。')
    return None
