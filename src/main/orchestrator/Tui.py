from multiprocessing import Queue
from queue import Empty
from typing import TypedDict, Unpack
from collections import deque

from rich.console import Console, Group
from rich.align import Align
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from rich.layout import Layout
from rich.columns import Columns
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
    Enhanced version with better layout and console-based message display.
    """

    workers = kwargs["workers"]
    worker_queues = kwargs["worker_queues"]
    worker_message_queue = kwargs["worker_message_queue"]

    # Keep track of messages for display (limited to last N messages)
    messages_history = deque(maxlen=50)  # Keep last 50 messages

    # Create the main layout with fixed sections
    layout = Layout()
    layout.split_column(
        Layout(name="spinner", size=3),
        Layout(name="status", size=6),
        Layout(name="messages"),
    )

    with Live(layout, refresh_per_second=16) as live:
        console = live.console

        # Loop to update the display
        while True:

            # Update spinner section
            spinner_panel = Panel(
                Align.center(Spinner(name="dots", text="Processing...")),
                title="Status",
                border_style="blue",
            )
            layout["spinner"].update(spinner_panel)

            # Update worker status section
            status_items = []
            for (worker, worker_count), queue in zip(workers, worker_queues):
                queue_size = queue.qsize()
                status_text = Text(
                    f"[{worker_count}] {worker.name:<20} Queue: {queue_size:3d}",
                )
                status_items.append(status_text)

            status_group = Group(*status_items)
            status_panel = Panel(
                status_group, title="Worker Status", border_style="green"
            )
            layout["status"].update(status_panel)

            # Check for new messages
            try:
                worker_message: WorkerMessage = worker_message_queue.get(timeout=0.25)

                # Create formatted message with color
                color = level_to_color(worker_message["message_level"])
                msg = Text()
                msg.append(f"{worker_message['worker_pid']:>6}", style="dim")
                msg.append(" | ")
                msg.append(f"{worker_message['worker_name']:<20}", style="cyan")
                msg.append(" | ")
                msg.append(f"{worker_message['message_level']:<8}", style=color)
                msg.append(" | ")
                msg.append(worker_message["message_text"], style=color)

                messages_history.append(msg)

            except Empty:
                pass
            except ValueError:
                break

            # Create header
            header = Text()
            header.append("{:>6}".format("PID"), style="bold dim")
            header.append(" | ")
            header.append("{:<20}".format("Worker Name"), style="bold cyan")
            header.append(" | ")
            header.append("{:<8}".format("Level"), style="bold")
            header.append(" | ")
            header.append("Message", style="bold")

            # Create separator
            separator = Text("─" * (console.width - 4), style="dim")

            # Combine header, separator, and messages
            messages_panel = Panel(
                Group(header, separator, *list(messages_history)),
                title="Worker Messages",
                border_style="yellow",
            )
            layout["messages"].update(messages_panel)
