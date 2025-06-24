from abc import ABC, abstractmethod


class ServiceRunner(ABC):

    @abstractmethod
    def run(self):
        """
        Run the service.
        This method should be implemented by subclasses to define the service's behavior.
        """
        pass

    @abstractmethod
    def join(self):
        """
        Wait for the service to finish.
        This method should be implemented by subclasses to handle any necessary cleanup or finalization.
        """
        pass
