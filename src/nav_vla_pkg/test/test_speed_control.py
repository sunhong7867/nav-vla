from nav_vla_pkg.speed_control import parse_speed_raw
from nav_vla_pkg.speed_control import limit_twist_to_raw_speed
from nav_vla_pkg.speed_control import raw_speed_to_mps


def test_explicit_speed_commands():
    assert parse_speed_raw("속도 100으로 가") == 100
    assert parse_speed_raw("속도를 80으로 줄여") == 80
    assert parse_speed_raw("set speed: 230") == 230
    assert parse_speed_raw("속도 999") == 250


def test_relative_and_named_speed_commands():
    assert parse_speed_raw("좀 더 빠르게 가", 150) == 170
    assert parse_speed_raw("속도 줄여", 150) == 130
    assert parse_speed_raw("천천히 가", 150) == 70
    assert parse_speed_raw("보통 속도로", 70) == 150
    assert parse_speed_raw("빠르게 가", 70) == 200


def test_unrelated_text_and_conversion():
    assert parse_speed_raw("T3까지 가") is None
    assert raw_speed_to_mps(140) == 140 / 255 * 5.0


def test_twist_limit_preserves_curvature_and_stops_at_zero():
    linear, angular = limit_twist_to_raw_speed(4.0, 0.8, 102)
    assert linear == 2.0
    assert angular == 0.4
    assert limit_twist_to_raw_speed(4.0, 0.8, 0) == (0.0, 0.0)
