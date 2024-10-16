import importlib
from models.base_model import BaseModel


def find_model_using_name(model_name):
    """Import the module "[model_name]_model.py".
        In the file, the class called ModelNameModel() will
        be instantiated. It has to be a subclass of BaseModel,
        and it is case-insensitive.
    """
    # Find respective model by name
    model_filename = "models." + model_name + "_model"

    # Import respective module
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace('_', '') + 'model'
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (
            model_filename, target_model_name))
        exit(0)

    return model


def create_model(configuration):
    """Create a model given the configuration.
    """
    model_class = find_model_using_name(configuration['model_name'])
    model = model_class(**configuration['model_kwargs'])

    return model
