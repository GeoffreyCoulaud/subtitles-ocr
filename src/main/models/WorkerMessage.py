from typing import TypedDict


class WorkerMessage(TypedDict):
    """Message sent by a worker to its manager"""

    worker_id: str
    message_level: str
    message_text: str
