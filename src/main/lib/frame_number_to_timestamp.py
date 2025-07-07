def frame_number_to_timestamp(frame_number: int, fps: float) -> str:
    """Convert a frame number to a timestamp string in the format HH:MM:SS,mmm."""
    total_seconds = frame_number / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((total_seconds - int(total_seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
