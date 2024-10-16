#!/usr/bin/env python3

import numpy as np
from transforms.base_transform import BaseTransform


class UniformNoiseTransform(BaseTransform):

    def __init__(self, config):
        self.config = config

        self.noise = float(config['noise_level'])

        self.random = False
        if 'random' in config:
            self.random = config['random']

    def __call__(self, pic):
        """
        Args:
            pic numpy.ndarray: Add noise to image

        Returns:
            numpy.ndarray: Image with added noise

        """

        # return untouched image if noise is 0
        if self.noise == 0:
            return pic

        # If random is false use self.noise as min/max values for noise distribution.
        # If random is true calculate new min/max values for noise distribution with self.noise as limits.
        if not self.random:
            noise = self.noise
        else:
            noise = np.random.uniform()*self.noise

        noise_range = np.amax(pic) * noise
        noise_array = np.random.uniform(low=-noise_range, high=noise_range,
                                        size=pic.shape[1] * pic.shape[2])

        # Add noise to image
        pic = pic + noise_array.reshape(pic.shape[0], pic.shape[1], pic.shape[2]).astype(np.float32)

        return pic

    def __repr__(self):
        return self.__class__.__name__ + '(noise={0})'.format(self.noise)
