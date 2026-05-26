import re

import pytest

from pyxllib.cv.rgbfmt import (
    _get_hexs_names,
    compare_bgr_pixel_tolerance,
    hash_text_to_hex_color,
    hash_text_to_std_color,
    jpeg_roundtrip_frame,
    to_bgr_frame,
)


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


def test_to_bgr_frame_converts_common_channel_orders():
    import numpy as np

    bgra = np.array([[[1, 2, 3, 255]]], dtype=np.uint8)
    rgba = np.array([[[3, 2, 1, 255]]], dtype=np.uint8)
    gray = np.array([[7]], dtype=np.uint8)

    assert to_bgr_frame(bgra, source_format='bgra').tolist() == [[[1, 2, 3]]]
    assert to_bgr_frame(rgba, source_format='rgba').tolist() == [[[1, 2, 3]]]
    assert to_bgr_frame(gray, source_format='gray').tolist() == [[[7, 7, 7]]]


def test_compare_bgr_pixel_tolerance_uses_per_channel_absolute_diff():
    import numpy as np

    reference = np.array([[[10, 20, 30], [50, 60, 70]]], dtype=np.uint8)
    current = np.array([[[12, 18, 31], [50, 60, 76]]], dtype=np.uint8)

    assert compare_bgr_pixel_tolerance(reference, current, pixel_tolerance=2) == (50, 0.5)
    assert compare_bgr_pixel_tolerance(reference, current, pixel_tolerance=6) == (100, 1.0)


def test_jpeg_roundtrip_frame_returns_decoded_bgr_and_bytes():
    import numpy as np

    frame = np.zeros((8, 8, 4), dtype=np.uint8)
    frame[:, :, 0] = 20
    frame[:, :, 1] = 40
    frame[:, :, 2] = 60
    frame[:, :, 3] = 255

    decoded, data = jpeg_roundtrip_frame(frame, quality=82, source_format='bgra', return_bytes=True)

    assert decoded.shape == (8, 8, 3)
    assert decoded.dtype == np.uint8
    assert data.startswith(b'\xff\xd8')
