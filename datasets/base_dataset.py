#!/usr/bin/env python3


from abc import ABC, abstractmethod
import torch.utils.data as data

"""This module implements an abstract base class (ABC) BaseDataset for all map-stype datasets. """


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for iterable datasets.
    """

    def __init__(self, configuration):
        self.configuration = configuration

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass
