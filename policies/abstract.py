from abc import ABC, abstractmethod
import logging

# Configure logger
logger = logging.getLogger(__name__)

class Policy(ABC):
    def __init__(self):
        logger.info(f"Initializing {self.__class__.__name__}")
        
    @abstractmethod
    def __call__(self):
        logger.debug(f"Abstract __call__ method called in {self.__class__.__name__}")
        raise NotImplementedError

class MessagePolicy(Policy,ABC):
    def __init__(self):
        super().__init__()
        logger.info(f"Initializing MessagePolicy: {self.__class__.__name__}")

class FactorPolicy(Policy,ABC):
    def __init__(self):
        super().__init__()
        logger.info(f"Initializing FactorPolicy: {self.__class__.__name__}")

class CostRedcutionPolicy(Policy,ABC):
    def __init__(self):
        super().__init__()
        logger.info(f"Initializing CostRedcutionPolicy: {self.__class__.__name__}")
