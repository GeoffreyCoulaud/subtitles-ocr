from multiprocessing import Queue
from queue import Empty
from typing import TypedDict, Unpack

from rich.console import Console

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

    # Use rich to display a nice TUI
    # - Display the queue sizes
    # - Display the messages from every worker in a dedicated section

    console = Console()

    # Loop to update the queue statuses and display messages
    with console.status("Updating worker statuses...") as status:
        while True:

            # Display the queue status and size in a sticky block on the bottom of the screen
            # (Not a progress bar, but a simple text display)
            status_message = "\n".join(
                "[%dx] %s : %d" % (worker_count, worker.name, queue.qsize())
                for (worker, worker_count), queue in zip(workers, worker_queues)
            )
            status.update(status_message)

            # Get a message from the worker message queue
            try:
                worker_message: WorkerMessage = worker_message_queue.get(timeout=1)
            except Empty:
                # No message received, continue to the next iteration
                continue
            except ValueError:
                # If the queue is closed, we can exit the loop
                break

            # Display the message in the console
            console.print(
                "[%s] [%s] %s"
                % (
                    worker_message["worker_name"],
                    worker_message["message_level"],
                    worker_message["message_text"],
                )
            )
