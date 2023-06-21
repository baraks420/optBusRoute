import random
from datetime import datetime

import pandas as pd
import numpy as np

from create_map import StationsMap


class Optimize:
    def __init__(self, stations_map: StationsMap, number_of_combinations: int = 50, max_dis: int = 100,
                 max_iter: int = 1000, alpa: float = 0.10, n_shuffle: int = 5):
        self.stations_map = stations_map
        self.stations = np.array(range(1, len(self.stations_map.exp) + 1))
        if len(self.stations) != self.stations_map.distances.shape[0] - 1:
            raise ValueError(f"the number of exp values not equal to the metrix given")
        self.number_of_top_route = round(number_of_combinations * alpa)
        self.max_dis = max_dis
        self.max_iter = max_iter
        self.number_of_combinations = number_of_combinations
        self.generation = []
        self.n_shuffle = n_shuffle
        self.total_exp = sum(stations_map.exp)
        self.exp_threshold = self.total_exp * 0.95
        self.best_route = None
        self.accuracy = 0

    def maximizer(self):
        self.generation = [self.create_random_combinations()]
        for _ in range(self.max_iter):
            generation = self.generation[-1]
            generation.sort_values("expectation", ascending=False, inplace=True)
            if generation.head(1).iloc[0]["expectation"] >= self.exp_threshold:
                self.accuracy = 100 * generation.head(1).iloc[0]["expectation"] / self.total_exp
                self.best_route = generation.head(1)
                break
            top_of_generation = generation.drop_duplicates("stations", ignore_index=True).head(self.number_of_top_route)
            top_of_generation = top_of_generation[top_of_generation.distance != np.inf]
            new_generation = self.create_new_generation(top_of_generation)
            self.generation.append(new_generation)

        generation = self.generation[-1]
        generation.sort_values("expectation", ascending=False, inplace=True)
        self.accuracy = 100 * generation.head(1).iloc[0]["expectation"] / self.total_exp
        self.best_route = generation.head(1)

    def cost(self, stations: tuple) -> (int, float):
        dis = 0
        stations_np = np.array(stations)
        for _ in range(self.n_shuffle):
            np.random.shuffle(stations_np)
            exp, dis = self.compute_distance_and_expectation(stations_np)
            if dis <= self.max_dis:
                return exp, dis , tuple(stations_np)
        return -1, dis ,stations

    def create_new_generation(self, generation: pd.DataFrame):
        new_gen = generation.copy()
        while len(new_gen.index) <= self.number_of_combinations:
            cross_over = generation.sample(2)
            child1, child2 = self.single_point_crossover(cross_over.iloc[0], cross_over.iloc[1])
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            child1.at[0, 'expectation'], child1.at[0, 'distance'], child1.at[0, 'stations'] = self.cost(child1.at[0, 'stations'])
            child2.at[0, 'expectation'], child2.at[0, 'distance'], child2.at[0, 'stations'] = self.cost(child2.at[0, 'stations'])
            new_gen = pd.concat([new_gen, child1, child2], ignore_index=True)
        return new_gen

    def mutation(self, child: pd.DataFrame, num_of_mutation: int = 1, probability: float = 0.5) -> pd.DataFrame:
        for _ in range(num_of_mutation):
            genome = child.at[0, 'genome']
            stations = np.array(child.at[0, 'stations'])
            index = random.randrange(len(genome))
            if random.random() <= probability and len(stations) > 1:
                genome[index] = bool(genome[index] - 1)
                if genome[index]:
                    stations = np.append(stations, index + 1)
                else:
                    stations = np.setdiff1d(stations, index + 1)
                child.at[0, 'stations'] = tuple(np.sort(stations))
        return child

    def single_point_crossover(self, cross1: pd.DataFrame, cross2: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        genome1 = cross1['genome']
        genome2 = cross2['genome']

        if len(genome1) != len(genome2):
            raise ValueError("Genomes child1 and child2 must be of same length")

        length = len(genome1)
        if length < 2:
            return cross1, cross2

        p = random.randint(1, length - 1)
        new_genome1 = np.append(genome1[0:p], genome2[p:])
        new_genome2 = np.append(genome2[0:p], genome1[p:])
        child1 = pd.DataFrame(
            [{'stations': tuple(self.stations[new_genome1]), 'genome': new_genome1, 'expectation': np.NAN,
              'distance': np.NAN}])
        child2 = pd.DataFrame(
            [{'stations': tuple(self.stations[new_genome2]), 'genome': new_genome2, 'expectation': np.NAN,
              'distance': np.NAN}])
        return child1, child2

    def create_random_combinations(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=['stations', 'genome', 'expectation', 'distance'])
        n_station = len(self.stations)
        for _ in range(self.number_of_combinations):
            genome = np.array(random.choices([True, False], weights=(90, 90), k=n_station))
            stations = tuple(self.stations[genome])
            expectation, distance,stations = self.cost(stations)
            new_raw = pd.DataFrame(
                [{'stations': stations, 'genome': genome, 'expectation': expectation, 'distance': distance}])
            df = pd.concat([df, new_raw], ignore_index=True)
        return df

    def compute_distance_and_expectation(self, stations) -> (int, float):
        old_station = 0
        distance = 0
        exp = 0
        if len(np.unique(stations)) < len(stations):
            raise ValueError("there is duplication in the stations")
        for new_station in stations:
            distance += self.stations_map.distances[old_station][new_station]
            exp += self.stations_map.exp[new_station - 1]
            old_station = new_station
        distance += self.stations_map.distances[old_station][0]
        return exp, distance


if __name__ == '__main__':
    now = datetime.now()
    # dis = np.array([
    #     [0, 1, np.inf, 3, 4],
    #     [0, 0, 5, 6, 7],
    #     [0, 0, 0, 8, 9],
    #     [0, 0, 0, 0, 10],
    #     [0, 0, 0, 0, 0]
    # ])
    # exp = [1, 2, 3, 4]
    random.seed(1)
    size = 100
    m = np.random.randint(1, 10, size=(size + 1, size + 1))
    dis = np.triu(m, k=1)
    exp = np.random.randint(1, 10, size=size)

    mapStations = StationsMap(dis, exp)
    # print(mapStations.distances)
    b = Optimize(
        mapStations
    )
    b.maximizer()
    print("time is:", datetime.now() - now)
    # print(b.generation)
