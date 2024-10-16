import torch
import torch.nn.functional as F
import random
import pytorch_lightning as pl
from typing import Tuple

class RectangularMasks(pl.LightningDataModule):
    def __init__(self, width: int, distance_range: Tuple[int, int], image_dimensions: Tuple[int, int],
                 leftmost: int, rightmost: int):
        super().__init__()
        self.width = width
        self.distance_range = distance_range
        self.image_height, self.image_width = image_dimensions
        self.leftmost = leftmost
        self.rightmost = rightmost

        # Validate the specified positions
        self.validate_positions()

    def validate_positions(self):
        max_distance = self.distance_range[1]
        max_width = self.width

        if self.leftmost < 0 or (self.rightmost+self.distance_range[1]+self.width) >= self.image_width:
            raise ValueError("Invalid positions: The leftmost and rightmost positions are out of range.")

        if self.rightmost - self.width <= self.leftmost:
            raise ValueError("Invalid positions: The specified positions do not allow for valid rectangular regions.")

    def get_mask(self) -> torch.Tensor:
        # Create an empty mask tensor with ones everywhere
        mask = torch.ones((self.image_height, self.image_width), dtype=torch.float32)

        # Calculate the left and right indices for the first rectangular region
        left1 = random.randint(self.leftmost, self.rightmost - self.width)
        right1 = left1 + self.width

        # Calculate the left and right indices for the second rectangular region
        distance = random.randint(self.distance_range[0], self.distance_range[1])
        left2 = right1 + distance
        right2 = left2 + self.width

        # Set the values in the mask to zero within the rectangular regions
        mask[:, left1:right1] = 0
        mask[:, left2:right2] = 0

        return mask


    def get_mask_and_borders(self) -> torch.Tensor:
        # Create an empty mask tensor with ones everywhere
        mask = torch.ones((self.image_height, self.image_width), dtype=torch.float32)

        # Calculate the left and right indices for the first rectangular region
        left1 = random.randint(self.leftmost, self.rightmost - self.width)
        right1 = left1 + self.width

        # Calculate the left and right indices for the second rectangular region
        distance = random.randint(self.distance_range[0], self.distance_range[1])
        left2 = right1 + distance
        right2 = left2 + self.width

        # Set the values in the mask to zero within the rectangular regions
        mask[:, left1:right1] = 0
        mask[:, left2:right2] = 0

        return mask , [left1,right1, left2,right2]
