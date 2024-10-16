import numpy as np
from transforms.base_transform import BaseTransform
import torch


class CenterTransform(BaseTransform):

    def __init__(self):
        pass

    def __call__(self, images):

        nr_images = images.shape[0]
        imshape0 = images.shape[1]
        imshape1 = images.shape[2]

        one_d = np.sum(np.array(images), axis=1)
        one_d -= np.min(one_d, axis=1)[0]
        x_axis = np.arange(one_d.shape[1])
        cog = np.sum(x_axis * one_d, axis=1) / np.sum(one_d, axis=1)

        shift = cog - imshape1 / 2  ##ungenau.. aber vermutlich erst mal ausreichend?
        shift = shift.astype('int')

        new_images = torch.zeros((nr_images, imshape0, imshape1))

        for i, shift in enumerate(shift):
            if shift < 0:
                new_images[i, :, -shift:] = images[i, :, :shift]
            elif shift > 0:
                new_images[i, :, :-shift] = images[i, :, shift:]
            else:
                new_images[i] = images[i]

        return new_images

    def __repr__(self):
        return self.__class__.__name__
