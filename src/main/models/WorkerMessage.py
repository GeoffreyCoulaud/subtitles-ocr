from typing import TypedDict


class WorkerMessage(TypedDict):
    """Message sent by a worker to its manager"""

    worker_name: str
    worker_pid: int
    message_level: str
    message_text: str
