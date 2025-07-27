def print_banner(message: str) -> None:
    """Print a banner with the given message."""
    width = max(len(line) for line in message.splitlines()) + 4
    padded_message = f"{message:^{width - 4}}"
    print("┌" + "─" * (width - 2) + "┐")
    print(f"│ {padded_message} │")
    print("└" + "─" * (width - 2) + "┘")
