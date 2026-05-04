import pytest
from subtitles_ocr.pipeline.skip import parse_time, format_time, parse_skip_range, normalize_ranges


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


class TestParseSkipRange:
    def test_seconds_range(self):
        assert parse_skip_range("0-90") == (0.0, 90.0)

    def test_mm_ss_range(self):
        assert parse_skip_range("1:30-22:00") == (90.0, 1320.0)

    def test_start_equals_end_raises(self):
        with pytest.raises(ValueError):
            parse_skip_range("90-90")

    def test_start_greater_than_end_raises(self):
        with pytest.raises(ValueError):
            parse_skip_range("90-0")

    def test_no_separator_raises(self):
        with pytest.raises(ValueError):
            parse_skip_range("90")

    def test_bad_start_raises(self):
        with pytest.raises(ValueError):
            parse_skip_range("abc-90")

    def test_bad_end_raises(self):
        with pytest.raises(ValueError):
            parse_skip_range("0-abc")


class TestNormalizeRanges:
    def test_empty(self):
        assert normalize_ranges([]) == []

    def test_single(self):
        assert normalize_ranges([(0.0, 90.0)]) == [(0.0, 90.0)]

    def test_non_overlapping_sorted(self):
        assert normalize_ranges([(0.0, 90.0), (120.0, 180.0)]) == [(0.0, 90.0), (120.0, 180.0)]

    def test_overlapping_merged(self):
        assert normalize_ranges([(0.0, 100.0), (90.0, 180.0)]) == [(0.0, 180.0)]

    def test_adjacent_merged(self):
        assert normalize_ranges([(0.0, 90.0), (90.0, 180.0)]) == [(0.0, 180.0)]

    def test_out_of_order_sorted(self):
        assert normalize_ranges([(120.0, 180.0), (0.0, 90.0)]) == [(0.0, 90.0), (120.0, 180.0)]

    def test_out_of_order_overlapping_merged(self):
        assert normalize_ranges([(100.0, 200.0), (0.0, 150.0)]) == [(0.0, 200.0)]
