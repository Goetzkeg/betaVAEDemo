#!/usr/bin/env python3

import numpy as np
from transforms.base_transform import BaseTransform


class NormalizationTransform(BaseTransform):

    def __init__(self, minv=None,maxv=None):

        if minv and maxv:
            self.min = minv
            self.max = maxv
        else:
            self.min = None
            self.max = None

    def __call__(self, pic):

        if (self.max is None) and (self.min is None):
            # Use values of individual image
            x_max = np.amax(pic)
            x_min = np.amin(pic)
        else:
            # Use given values for min and max
            x_max = self.max
            x_min = self.min

        pic = (pic - x_min) / (x_max - x_min)
        return pic

    def __repr__(self):
        return self.__class__.__name__
