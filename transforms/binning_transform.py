import numpy as np
import torch
from transforms.base_transform import BaseTransform


class BinningTransform(BaseTransform):
    """This one wants gerade Teiler of the image size."""

    def __init__(self, binning_dims):
        self.binning_dims = binning_dims

    def __call__(self, images):
        bs = self.binning_dims
        arr = images
        s = 1  # skip first dim
        shape = [arr.shape[0], arr.shape[s + 0] // bs[0], bs[0], arr.shape[s + 1] // bs[1], bs[1]]
        result = np.array(arr).reshape(*shape).mean(axis=(2, 4))
        return torch.tensor(result)

    def __repr__(self):
        return self.__class__.__name__
