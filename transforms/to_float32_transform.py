import torch
from transforms.base_transform import BaseTransform


class ToFloat32Transform(BaseTransform):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

    def __repr__(self):
        return self.__class__.__name__
