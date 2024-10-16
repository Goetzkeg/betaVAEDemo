import numpy as np
from transforms.base_transform import BaseTransform


class FlipTransform():

    def __init__(self, flip_runs: list[int]):
        self.flipruns = flip_runs

    def __call__(self, arr, runnumber):
        arr = np.array(arr)
        if runnumber in self.flipruns:
            arr = np.flip(arr, axis=2)
        return arr

    def __repr__(self):
        return self.__class__.__name__
