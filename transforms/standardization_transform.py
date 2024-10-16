#!/usr/bin/env python3

import numpy as np
from transforms.base_transform import BaseTransform


class StandardizationTransform(BaseTransform):

    def __init__(self, config):
        self.config = config

        if 'mean' in config and 'std' in config:
            self.mean = config['mean']
            self.std = config['std']
        else:
            self.mean = None
            self.std = None

    def __call__(self, pic):

        if (self.mean is None) and (self.std is None):
            mean = np.mean(pic.flatten())
            std = np.std(pic.flatten())
        else:
            mean = self.mean
            std = self.std

        new_pic = pic - mean
        new_pic = new_pic / std

        return new_pic

    def __repr__(self):
        return self.__class__.__name__
