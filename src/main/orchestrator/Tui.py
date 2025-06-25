from multiprocessing import Queue
from queue import Empty
from typing import TypedDict, Unpack

from src.main.models.WorkerMessage import WorkerMessage
from src.main.workers.Worker import Worker


class TuiKwargs(TypedDict):
    workers: list[tuple[Worker, int]]
    worker_queues: list[Queue]
    worker_message_queue: Queue[WorkerMessage]


def tui(**kwargs: Unpack[TuiKwargs]) -> None:
    """
    Run the TUI process to display messages from the worker message queue.
    This is a placeholder for the actual TUI implementation.
    """

    workers = kwargs["workers"]
    worker_queues = kwargs["worker_queues"]
    worker_message_queue = kwargs["worker_message_queue"]

    # TODO use rich to display a nice TUI
    # - Display the queue sizes
    # - Display the messages from every worker in a dedicated section

    # --- temporary implementation ---

    print("TUI is running...")

    while True:

        message: None | WorkerMessage
        try:
            message = worker_message_queue.get(timeout=1)
        except (ValueError, Empty):
            message = None

        # Display the message if it exists
        if message is None:
            continue

        print(
            "[%s] [%s] %s"
            % (
                message["worker_id"],
                message["message_level"],
                message["message_text"],
            )
        )
