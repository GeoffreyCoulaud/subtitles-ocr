from abc import ABC, abstractmethod
from logging import Logger
from typing import Generic, TypeVar
from multiprocessing import JoinableQueue

I = TypeVar("I")
O = TypeVar("O")


class Service(ABC, Generic[I, O]):
    """Base class for all services"""

    _logger: Logger
    __input_queue: JoinableQueue[I]
    __output_queue: JoinableQueue[O]

    def __init__(
        self,
        input_queue: JoinableQueue[I],
        output_queue: JoinableQueue[O],
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
            else:
                results = self.process_item(item)
                for result in results:
                    self.__output_queue.put(result)
                self.__input_queue.task_done()

    @abstractmethod
    def process_item(self, item: I) -> list[O]:
        """Process an item from the input queue and return the result."""
