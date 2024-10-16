#!/usr/bin/env python3

import importlib
from datasets.base_dataset import BaseDataset


def find_dataset_using_name(dataset_name):
    """Import the module "[dataset_name]_dataset.py". In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset. [dataset_name] is read from the config file.
    """
    dataset_filename = "datasets." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError(
            'In {0}.py, there should be a subclass of BaseDataset with class name that matches '
            '{1} in lowercase.'.format(dataset_filename, target_dataset_name))

    return dataset


def create_dataset(configuration):
    """Create a dataset given the configuration (loaded from the json file).
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and train.py/validate.py
    Example:
        from datasets import create_dataset
        dataset = create_dataset(configuration)
    """

    dataset_class = find_dataset_using_name(configuration['dataset_name'])
    dataset = dataset_class(configuration)

    return dataset
