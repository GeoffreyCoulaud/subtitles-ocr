from services.Service import Service


from multiprocessing import Process
from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class ServiceRunner(Process, Generic[I, O]):
    """Runner for a service that is run asynchronously in a separate process."""

    __service: Service[I, O]

    def __init__(self, service: Service[I, O]) -> None:
        super().__init__()
        self.__service = service

    def run(self):
        """Run the service in a separate process."""
        self.__service.run()
