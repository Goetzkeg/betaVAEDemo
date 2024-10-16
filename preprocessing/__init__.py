import numpy as np
import importlib


def find_preproc_using_name(prepro_name):
    """Import the module "[transform_name]_transforms.py". In the file, the class called TransformNameTransforms() will
    be instantiated. It has to be a subclass of BaseTransforms. [Transform_name] is read from the config file.
    """
    if prepro_name[0:7] == 'filter_':

        prepro_filename = "filters." + prepro_name[7:] + "_filter"
        transformlib = importlib.import_module(prepro_filename)

        transform = None
        target_transform_name = prepro_name.replace('_', '') + 'filter'
        for name, cls in transformlib.__dict__.items():
            if name.lower() == target_transform_name.lower():
                transform = cls

        if transform is None:
            raise NotImplementedError(
                'In {0}.py, there should be a subclass of BaseTransform with class name that matches '
                '{1} in lowercase.'.format(prepro_filename, target_transform_name))

        return transform

    else:

        transform_filename = "transforms." + prepro_name + "_transform"
        transformlib = importlib.import_module(transform_filename)

        transform = None
        target_transform_name = prepro_name.replace('_', '') + 'transform'
        for name, cls in transformlib.__dict__.items():
            if name.lower() == target_transform_name.lower():
                transform = cls

        if transform is None:
            raise NotImplementedError(
                'In {0}.py, there should be a subclass of BaseTransform with class name that matches '
                '{1} in lowercase.'.format(transform_filename, target_transform_name))

        return transform


def create_preproc(key, transform_dict):
    """Create a transform given the configuration (loaded from the json file).
    Example:
        from transforms import create_transform
        transform = create_transform(configuration)
    """

    preproc_class = find_preproc_using_name(key)
    preproc = preproc_class(**transform_dict)

    return preproc


def preprocessing_steps(input_arr, preprocessing: list, preprocessing_kwargs: dict, runnumber: int):
    arr = input_arr
    all_ind = np.arange(arr.shape[0])
    passed_ind = np.arange(arr.shape[0])
    for pre in preprocessing:
        if pre[0:7] == 'filter_':
            filter_cl = create_preproc(pre, preprocessing_kwargs[pre])
            ind = all_ind[filter_cl(arr)]
            passed_ind = np.intersect1d(passed_ind, ind)
        else:
            transform = create_preproc(pre, preprocessing_kwargs[pre])
            if pre in ['flip', 'adapt_gain']:
                arr = transform(arr, runnumber)
            else:
                arr = transform(arr)
    return arr[passed_ind], passed_ind
