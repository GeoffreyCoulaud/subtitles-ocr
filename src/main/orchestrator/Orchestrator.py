from multiprocessing import Process
from multiprocessing.managers import SyncManager
from multiprocessing.queues import Queue
from typing import Generic, TypeVar

from src.main.models.WorkerMessage import WorkerMessage
from src.main.orchestrator.Tui import TuiKwargs, tui
from src.main.workers.Worker import Worker

I = TypeVar("I")
O = TypeVar("O")


class Orchestrator(Generic[I, O]):
    """
    Class in charge of running different workers in parallel.

    Takes in:
    - Workers with their number of parallel instances
    - Input data

    Returns:
    - Output data from the workers
    """

    __workers: list[tuple[Worker, int]]

    __worker_processes: list[Process]
    __worker_queues: list[Queue]
    __worker_message_queue: Queue[WorkerMessage]
    __tui_process: Process
    __manager: SyncManager

    def __init__(self, workers: list[tuple[Worker, int]]):
        self.__workers = workers
        self.__worker_processes = []
        self.__worker_queues = []
        self.__manager = SyncManager()

    def run(self, input_data: list[I]) -> None:
        """Run the orchestrator with the given input data."""

        # Start the manager for inter-process communication
        self.__manager.start()

        # Create queues
        self.__worker_message_queue = self.__manager.Queue()  # type: ignore
        self.__worker_queues = [  # type: ignore
            self.__manager.Queue() for _ in range(len(self.__workers) + 1)
        ]

        # Create worker processes
        for i, (worker, worker_pool_size) in enumerate(self.__workers):
            input_queue = self.__worker_queues[i]
            output_queue = self.__worker_queues[i + 1]
            for _ in range(worker_pool_size):
                self.__worker_processes.append(
                    Process(
                        target=worker.run,
                        kwargs={
                            "input_queue": input_queue,
                            "output_queue": output_queue,
                            "message_queue": self.__worker_message_queue,
                        },
                    )
                )

        # Create TUI process
        tui_kwargs: TuiKwargs = {
            "workers": self.__workers,
            "worker_queues": self.__worker_queues,
            "worker_message_queue": self.__worker_message_queue,
        }
        self.__tui_process = Process(target=tui, kwargs=tui_kwargs)

        # Input the data into the first worker's input queue
        for item in input_data:
            self.__worker_queues[0].put(item)

        # Start worker processes
        for process in self.__worker_processes:
            process.start()

        # Start the TUI process
        self.__tui_process.start()

    def join(self) -> list[O]:
        """Wait for all workers to finish and collect the output data."""

        # Join worker processes
        for process in self.__worker_processes:
            process.join()

        # Close the tui process
        self.__tui_process.terminate()

        # Close the manager
        self.__manager.shutdown()

        # Collect output from the last worker's output queue
        output_queue = self.__worker_queues[-1]
        output_data = []
        while True:
            try:
                item = output_queue.get(timeout=1)
            except ValueError:
                break
            output_data.append(item)

        # Return the collected output data
        return output_data
