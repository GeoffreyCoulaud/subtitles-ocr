from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import Generic, TypeVar

from src.main.models.WorkerMessage import WorkerMessage

I = TypeVar("I")
O = TypeVar("O")


class Worker(ABC, Generic[I, O]):
    """Base class for all workers"""

    name: str
    input_queue_name: str
    output_queue_name: str

    _message_queue: "Queue[WorkerMessage]"
    """Message queue that may be used to send messages to the orchestrator while processing items."""

    def run(
        self,
        input_queue: "Queue[I]",
        output_queue: "Queue[O]",
        message_queue: "Queue[WorkerMessage]",
    ) -> None:
        """Run the service, processing items from the input queue and putting results in the output queue."""

        self._message_queue = message_queue

        while True:
            try:
                item = input_queue.get()
            except ValueError:
                # Close the output queue if the input queue is shut down and empty
                # This signals that no more items will be outputted
                break

            # Process the item and put the results in the output queue
            try:
                results = self.process_item(item)
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                break
            for result in results:
                output_queue.put(result)

        # Let the worker finalize its work
        final_results = self.process_no_more_items()
        for result in final_results:
            output_queue.put(result)

        # Close the output queue to signal that no more items will be outputted
        output_queue.close()

        # Close the messages queue to signal that no more messages will be sent
        self._message_queue.close()

    @abstractmethod
    def process_item(self, item: I) -> list[O]:
        """Process an item from the input queue and return the result."""

    def process_no_more_items(self) -> list[O]:
        """
        Product eventual final output after the input queue is closed.
        May be overridden by subclasses to produce final output after all items have been processed.
        By default, returns an empty list.
        """
        return []
