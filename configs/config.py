import yaml
import sys

sys.path.append('./../')


def get_config(configname, path='./configs/'):
    with open(f'{path}/{configname}', "r") as stream:
        config_dict = yaml.safe_load(stream)
    return config_dict
