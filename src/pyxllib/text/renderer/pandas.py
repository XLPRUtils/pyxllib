import sys
from functools import singledispatch

import pandas as pd

from pyxllib.text.pupil import shorten, east_asian_shorten
from pyxllib.text.renderer import to_text, to_html


@to_text.register(pd.DataFrame)
def _(df, *args, ambiguous_as_wide=None, shorten=True):
    """输出DataFrame
    DataFrame可以直接输出的，这里是增加了对中文字符的对齐效果支持

    :param df: DataFrame数据结构
    :param args: option_context格式控制
    :param ambiguous_as_wide: 是否对①②③这种域宽有歧义的设为宽字符
        win32平台上和linux上①域宽不同，默认win32是域宽2，linux是域宽1
    :param shorten: 是否对每个元素提前进行字符串化并控制长度在display.max_colwidth以内
        因为pandas的字符串截取遇到中文是有问题的，可以用我自定义的函数先做截取
        默认开启，不过这步比较消耗时间

    >> df = pd.DataFrame({'哈哈': ['a'*100, '哈\n①'*10, 'a哈'*100]})
                                                        哈哈
        0  aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...
        1   哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①哈 ①...
        2  a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a哈a...
    """
    import pandas as pd

    if ambiguous_as_wide is None:
        ambiguous_as_wide = sys.platform == "win32"
    with pd.option_context(
        "display.unicode.east_asian_width",
        True,  # 中文输出必备选项，用来控制正确的域宽
        "display.unicode.ambiguous_as_wide",
        ambiguous_as_wide,
        "display.max_columns",
        20,  # 最大列数设置到20列
        "display.width",
        200,  # 最大宽度设置到200
        *args,
    ):
        if shorten:  # applymap可以对所有的元素进行映射处理，并返回一个新的df
            if hasattr(df, "map"):
                df = df.map(lambda x: east_asian_shorten(str(x), pd.options.display.max_colwidth))
            else:
                df = df.applymap(lambda x: east_asian_shorten(str(x), pd.options.display.max_colwidth))
        s = str(df)
    return s
