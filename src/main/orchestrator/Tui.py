from multiprocessing import Queue
from queue import Empty
from typing import TypedDict, Unpack

from rich.console import Console, Group
from rich.align import Align
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from rich.layout import Layout
from rich import print

from src.main.models.WorkerMessage import WorkerMessage
from src.main.workers.Worker import Worker


class TuiKwargs(TypedDict):
    workers: list[tuple[Worker, int]]
    worker_queues: list[Queue]
    worker_message_queue: "Queue[WorkerMessage]"


def level_to_color(level: str) -> str:
    """
    Convert a message level to a color string.
    """
    colors = {
        "error": "red",
        "warning": "yellow",
        "info": "white",
        "debug": "grey35",
    }
    return colors.get(level.lower(), "white")


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

    messages_table = Table("PID", "Worker", "Level", "Message Text", expand=True)
    status_layout = Layout(name="status")
    layout = Layout()
    layout.split_column(
        Layout(Panel(Spinner(name="dots", text="Working..."), height=3)),
        Layout(Panel(status_layout, height=6)),
        messages_table,
    )

    with Live(layout) as live:

        # Loop to update the queue statuses and display messages
        while True:

            # Create the queue status renderable
            status_messages = [
                "[%dx] %s : %d" % (worker_count, worker.name, queue.qsize())
                for (worker, worker_count), queue in zip(workers, worker_queues)
            ]
            status_renderable = Text("\n".join(status_messages))
            status_layout.update(status_renderable)

            # Get a message from the worker message queue
            try:
                worker_message: WorkerMessage = worker_message_queue.get(timeout=1)
            except Empty:
                pass
            except ValueError:
                break
            else:
                # Add the message to the messages table
                color = level_to_color(worker_message["message_level"])
                messages_table.add_row(
                    str(worker_message["worker_pid"]),
                    worker_message["worker_name"],
                    worker_message["message_level"],
                    worker_message["message_text"],
                    style=Style(color=color),
                )
