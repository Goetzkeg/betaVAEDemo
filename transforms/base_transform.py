

from abc import ABC, abstractmethod

"""This module implements an abstract base class (ABC) 'BaseTransforms' for preprocessing data and labels. """


class BaseTransform(ABC):

    @abstractmethod
    def __call__(self, pic):
        pass
