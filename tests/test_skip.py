import pytest
from subtitles_ocr.pipeline.skip import parse_time, format_time


class TestParseTime:
    def test_seconds_only(self):
        assert parse_time("90") == 90.0

    def test_zero(self):
        assert parse_time("0") == 0.0

    def test_minutes_seconds(self):
        assert parse_time("1:30") == 90.0

    def test_hours_minutes_seconds(self):
        assert parse_time("0:01:30") == 90.0

    def test_fractional_seconds(self):
        assert parse_time("1:30.5") == pytest.approx(90.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_time("")

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            parse_time("-1")

    def test_malformed_raises(self):
        with pytest.raises(ValueError):
            parse_time("abc")

    def test_too_many_parts_raises(self):
        with pytest.raises(ValueError):
            parse_time("1:2:3:4")


class TestFormatTime:
    def test_zero(self):
        assert format_time(0.0) == "0:00"

    def test_minutes_seconds(self):
        assert format_time(90.0) == "1:30"

    def test_minutes_only(self):
        assert format_time(60.0) == "1:00"

    def test_hours(self):
        assert format_time(3661.0) == "1:01:01"
