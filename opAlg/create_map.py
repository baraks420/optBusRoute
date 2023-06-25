import numpy as np


class StationsMap:
    def __init__(self, distances, exp):
        self.distances = distances + distances.T # creating the full matrix from triangular one (easier to make)
        self.exp = exp
