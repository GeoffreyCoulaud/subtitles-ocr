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
