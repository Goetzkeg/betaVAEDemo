import importlib

import torchvision
from transforms.base_transform import BaseTransform


def find_transform_using_name(transform_name):
    """Import the module "[transform_name]_transforms.py". In the file, the class called TransformNameTransforms() will
    be instantiated. It has to be a subclass of BaseTransforms. [Transform_name] is read from the config file.
    """
    transform_filename = "transforms." + transform_name + "_transform"
    transformlib = importlib.import_module(transform_filename)

    transform = None
    target_transform_name = transform_name.replace('_', '') + 'transform'
    for name, cls in transformlib.__dict__.items():
        if name.lower() == target_transform_name.lower():
            transform = cls

    if transform is None:
        raise NotImplementedError(
            'In {0}.py, there should be a subclass of BaseTransform with class name that matches '
            '{1} in lowercase.'.format(transform_filename, target_transform_name))

    return transform


def create_transform(key, transform_dict):
    """Create a transform given the configuration (loaded from the json file).
    Example:
        from transforms import create_transform
        transform = create_transform(configuration)
    """

    transform_class = find_transform_using_name(key)
    transform = transform_class(**transform_dict)

    return transform


def get_transforms(transformations:list, transformation_kwargs: dict):
    """

    :transformconfig: Dictionary with contains the wanted transforms and parameters
    :return: Composes list of all transforms
    """
    list_of_transforms = []

    for key in transformations:
        list_of_transforms.append(create_transform(key, transformation_kwargs[key]))

    return torchvision.transforms.Compose(list_of_transforms)
