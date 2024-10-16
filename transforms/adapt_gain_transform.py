import numpy as np
from transforms.base_transform import BaseTransform


class AdaptGainTransform():

    def __init__(self, run_nr: list[int], multiply: float):
        self.run_nr = run_nr
        self.mult = multiply

    def __call__(self, arr, runnumber):
        if runnumber in self.run_nr:
            arr = arr*self.mult
        return arr

    def __repr__(self):
        return self.__class__.__name__
