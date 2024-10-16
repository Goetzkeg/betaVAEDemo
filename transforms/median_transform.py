import numpy as np
import torch
from scipy import ndimage
from transforms.base_transform import BaseTransform


class MedianTransform(BaseTransform):
    """One iteration of a median transform with one iteration. Borders are zero"""

    def __init__(self, size = 3):
        self.footprint = np.ones((1,size,size))

    def __call__(self, images):
        result = ndimage.median_filter(np.array(images),footprint = self.footprint, mode = 'constant')
        return torch.tensor(result)

    def __repr__(self):
        return self.__class__.__name__
