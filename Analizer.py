from abc import ABC, abstractmethod

class AbstractAnalizer(ABC):

    @abstractmethod
    def draw(self, results, img):
        pass

    