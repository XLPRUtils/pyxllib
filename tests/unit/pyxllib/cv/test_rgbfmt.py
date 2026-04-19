import re

import pytest

from pyxllib.cv.rgbfmt import _get_hexs_names, hash_text_to_hex_color, hash_text_to_std_color


def _brightness(hex_color: str) -> float:
    value = hex_color.lstrip('#')
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (r * 299 + g * 587 + b * 114) / 1000


def test_hash_text_to_std_color_is_stable_and_uses_standard_palette():
    color1 = hash_text_to_std_color('2026/04/18', tone='dark')
    color2 = hash_text_to_std_color('2026/04/18', tone='dark')
    hex_color = color1.to_hex()

    assert color1.to_tuple() == color2.to_tuple()
    assert hex_color == hash_text_to_hex_color('2026/04/18', tone='dark')
    assert hex_color[1:] in _get_hexs_names(2)[0]
    assert re.fullmatch(r'#[0-9A-F]{6}', hex_color)


def test_hash_text_to_std_color_supports_dark_and_light_tones():
    dark_hex = hash_text_to_hex_color('吴菲', tone='dark')
    light_hex = hash_text_to_hex_color('吴菲', tone='light')

    assert _brightness(dark_hex) <= 150
    assert _brightness(light_hex) >= 170


def test_hash_text_to_std_color_rejects_unknown_tone():
    with pytest.raises(ValueError):
        hash_text_to_std_color('课程A', tone='unknown')
