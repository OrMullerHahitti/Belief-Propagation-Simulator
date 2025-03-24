from abc import ABC, abstractmethod

class Policy(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

class MessagePolicy(Policy,ABC):
    pass

class FactorPolicy(Policy,ABC):
    pass

class CostRedcutionPolicy(Policy,ABC):
    pass


