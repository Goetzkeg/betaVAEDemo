import yaml
import os
from copy import deepcopy

def parse_configuration_yaml(config_file,rootpath='./configs'):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        if config_file[-5:] != '.yaml':
            config_file += '.yaml'

        assert os.path.exists(rootpath+'/'+config_file), f'{config_file} not found in {os.getcwd()}{rootpath}/'
        with open(rootpath+'/'+config_file) as yaml_file:
            return yaml.full_load(yaml_file)
    else:
        return config_file


def replace_placeholders(d, fulldict={}):
    if len(fulldict)==0:
        fulldict = deepcopy(d)
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = replace_placeholders(value, fulldict)
        elif isinstance(value, str) and '%' in value:

            parts = value.split('%')
            for i in range(1, len(parts), 2):
                part = parts[i] # 'path/to/valueinDict'
                iterdict = fulldict
                for key_path in part.split('/'):
                    try:
                        iterdict = iterdict[key_path]
                    except KeyError:
                        raise KeyError(f'key {key_path} not found in {iterdict}')
                parts[i] = iterdict
            d[key] = ''.join(parts)
    return d
