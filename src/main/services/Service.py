from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class Service(ABC, Generic[I, O]):
    """Base class for all services"""

    __input_queue: "Queue[I]"
    __output_queue: "Queue[O]"

    def __init__(
        self,
        input_queue: "Queue[I]",
        output_queue: "Queue[O]",
    ) -> None:
        self.__input_queue = input_queue
        self.__output_queue = output_queue

    def run(self) -> None:
        """Run the service, processing items from the input queue and putting results in the output queue."""
        while True:
            try:
                item = self.__input_queue.get()
            except ValueError:
                # Close the output queue if the input queue is shut down and empty
                # This signals that no more items will be outputted
                self.__output_queue.close()
                break

            # Process the item and put the results in the output queue
            try:
                results = self.process_item(item)
            except Exception as e:
                print(f"Error processing item {item}: {e}")
                break
            for result in results:
                self.__output_queue.put(result)

    @abstractmethod
    def process_item(self, item: I) -> list[O]:
        """Process an item from the input queue and return the result."""
