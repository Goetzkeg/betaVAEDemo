import torch
from transforms.base_transform import BaseTransform


class CutImagesTransform(BaseTransform):
    def __init__(self, borders):
        self.borders = borders

    def __call__(self, images):
        b = self.borders
        return torch.tensor(images[:, b[0]:b[1], b[2]:b[3]])

    def __repr__(self):
        return self.__class__.__name__
