import numpy as np


class StationsMap:
    def __init__(self, distances, exp):
        self.distances = distances + distances.T
        self.exp = exp


if __name__ == '__main__':
    dis = np.array([
        [0, 1,np.inf, 3, 4],
        [0, 0, 5, 6, 7],
        [0, 0, 0, 8, 9],
        [0, 0, 0, 0, 10],
        [0, 0, 0, 0, 0]
    ])
    exp = [1, 2, 3, 4, 5]
    mapStations = StationsMap(dis,exp)
    print(mapStations.distances)
    print(mapStations.exp)