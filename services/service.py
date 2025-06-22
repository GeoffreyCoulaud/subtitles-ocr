from abc import ABC, abstractmethod
from queue import Queue, ShutDown
from typing import Any, Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class Service(ABC, Generic[I, O]):
    """Base class for all services"""

    __input_queue: Queue[I]
    __output_queue: Queue[O]

    def __init__(
        self,
        input_queue: Queue[I],
        output_queue: Queue[O],
    ) -> None:
        self.__input_queue = input_queue
        self.__output_queue = output_queue

    def run(self) -> None:
        """Run the service, processing items from the input queue and putting results in the output queue."""
        while True:
            try:
                item = self.__input_queue.get()
            except ShutDown:
                is_result, result = self.handle_end_of_input()
                if is_result:
                    self.__output_queue.put(result)
                break
            if item is None:
                is_result, result = self.handle_end_of_input()
                if is_result:
                    self.__output_queue.put(result)
                break
            else:
                result = self.process_item(item)
                self.__output_queue.put(result)

    def handle_end_of_input(self) -> tuple[bool, Any | O]:
        """
        Handle the end of input by putting None in the output queue.
        If the the return value's first element is True, the return value will be put in the output queue.
        """
        return False, None

    @abstractmethod
    def process_item(self, item: I) -> O:
        """Process an item from the input queue and return the result."""
