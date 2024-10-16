import sys
import os
import torch

full_path = os.path.realpath(__file__)
dirname = os.path.dirname(full_path)
main_dir = os.path.dirname(dirname)
#print(full_path)
sys.path.append(dirname+'/../')
#sys.path.append("../")

from configs import parse_configuration_yaml
from models import find_model_using_name


def get_trained_model(config ,ckpt_name = 'last.ckpt',rootpath_config='./configs',device = 'cpu'):
    config = parse_configuration_yaml(config, rootpath=rootpath_config)

    plmodule = find_model_using_name(config['model_params']['model_name'])
    path = main_dir+ config['callbacks']['checkpoint_time']['dirpath'][1:]+ config['wandb_params']['id']+'/'+ckpt_name
    assert os.path.exists(path), f'{path} not found'
    model = plmodule.load_from_checkpoint(path,**config["model_params"]['model_kwargs'],map_location=torch.device(device))

    return model
