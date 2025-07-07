def print_banner(message: str, width: int = 80) -> None:
    """Print a banner with the given message."""
    padded_message = f"{message:^{width - 4}}"
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {padded_message} │")
    print("└" + "─" * (width - 2) + "┘")
