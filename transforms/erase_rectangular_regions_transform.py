#%%
import torch
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from typing import Tuple
from transforms.base_transform import BaseTransform


class EraseRectangularRegions(BaseTransform):
    def __init__(self, width: int, distance_range: Tuple[int, int], leftmost: int, rightmost: int):
        super().__init__()
        self.width = width
        self.distance_range = distance_range
        self.leftmost = leftmost
        self.rightmost = rightmost

        # Validate the specified positions
        self._validate_positions()

    def _validate_positions(self):
        if self.rightmost - self.leftmost < (2 * self.width + self.distance_range[1]):
            raise ValueError("Invalid positions: Cannot accommodate two rectangular regions with the specified width and distance range.")

    def erase_regions(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape[-3:]  # Get the height and width of the image

        # Create a copy of the original image
        erased_image = image.clone()

        # Calculate the left and right indices for the first rectangular region
        left1 = random.randint(self.leftmost, self.rightmost - (2 * self.width + self.distance_range[1]))
        right1 = left1 + self.width

        # Calculate the left and right indices for the second rectangular region
        distance = random.randint(self.distance_range[0], self.distance_range[1])
        left2 = left1 + self.width + distance
        right2 = left2 + self.width

        # Erase the first rectangular region in the copied image
        erased_image[:, :, left1:right1] = 0

        # Erase the second rectangular region in the copied image
        erased_image[:, :, left2:right2] = 0

        return erased_image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.erase_regions(image)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.erase_regions(image)


    def __repr__(self):
        return self.__class__.__name__