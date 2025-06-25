from multiprocessing import Process

from src.main.runners.ServiceRunner import ServiceRunner
from src.main.services.Service import Service


class ProcessServiceRunner(ServiceRunner):
    """Runner for a service that is run asynchronously in a separate process."""

    _service: Service
    _process: Process

    def __init__(self, service: Service) -> None:
        super().__init__()
        self._service = service

    def run(self):
        """Run the service in a separate process."""
        self._process = Process(target=self._service.run)
        self._process.start()

    def join(self):
        """Wait for the service process to finish."""
        self._process.join()
