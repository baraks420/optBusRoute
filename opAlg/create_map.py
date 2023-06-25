import numpy as np


class StationsMap:
    def __init__(self, distances, exp):
        self.distances = distances + distances.T
        self.exp = exp
