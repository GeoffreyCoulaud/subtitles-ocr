def handle_keyboard_interrupt(func):
    """
    Decorator to handle keyboard interrupts gracefully.
    It will print a message and exit the program when a keyboard interrupt is caught.
    """

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user. Exiting...")
            exit(0)

    return wrapper
