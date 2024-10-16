import numpy as np
import torch
from transforms.base_transform import BaseTransform
from evaluation.eval_utils import multiply_along_axis

class BackgroundSubtractionTransform(BaseTransform):
    """This one removes background. If use dark factor is true, use a factor to adapt the dark image to the measured one.
    Background image has already to be cut in the right shape and flipped in the right direction."""

    def __init__(self, path_to_background_npy, use_dark_factor = True, border_area = 3 ):
        self.bg_image = np.load(path_to_background_npy)
        self.use_dark_factor = use_dark_factor
        self.border_area = border_area


    def __call__(self, im_arr):
        mean_dark = self.bg_image
        border_area = self.border_area
        if self.use_dark_factor:
            im_arr = np.array(im_arr)
            dark_factor = (np.sum(im_arr[:, -border_area:, :], axis=(1, 2)) + np.sum(im_arr[:, :border_area, :], axis=(1, 2))) / (
                    np.sum(mean_dark[-border_area:, :]) + np.sum(mean_dark[:border_area, :]))
            mean_dark = np.broadcast_to(mean_dark[None, ...], (im_arr.shape[0], mean_dark.shape[0], mean_dark.shape[1]))
            mean_dark = torch.tensor(multiply_along_axis(mean_dark, dark_factor, 0))
            im_arr = torch.tensor(im_arr)
        else:
            mean_dark = torch.tensor(mean_dark)
        result = im_arr - mean_dark

        return torch.tensor(result)

    def __repr__(self):
        return self.__class__.__name__
