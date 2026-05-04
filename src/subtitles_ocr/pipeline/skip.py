from subtitles_ocr.models import Frame


def parse_time(s: str) -> float:
    """Parse SS, MM:SS, or HH:MM:SS into total seconds."""
    parts = s.strip().split(":")
    msg = f"Invalid time {s!r}: expected SS, MM:SS, or HH:MM:SS"
    try:
        if len(parts) == 1:
            total = float(parts[0])
        elif len(parts) == 2:
            total = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            total = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError(msg)
    except (ValueError, TypeError):
        raise ValueError(msg)
    if total < 0:
        raise ValueError(msg)
    return total


def format_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS for display."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def parse_skip_range(s: str) -> tuple[float, float]:
    """Parse 'START-END' into (start_seconds, end_seconds). Splits on the first '-'."""
    parts = s.split("-", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid skip range {s!r}: expected START-END format")
    start = parse_time(parts[0])
    end = parse_time(parts[1])
    if start >= end:
        raise ValueError(f"Invalid skip range {s!r}: start ({start}) must be less than end ({end})")
    return (start, end)
