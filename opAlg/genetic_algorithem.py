import random
from datetime import datetime
import pandas as pd
import numpy as np
from create_map import StationsMap


class Optimize:
    """
    Parameters (input): 
        - stations_map : np.ndarray (n,n) matrix that represent the Graph, with distance between stations and expectations
        - number_of_combinations : (int) Number of routes generated at each iteration
        - max_dis : (int) Max distance allowed in a route
        - max_iter :(int) max iteration
        - alpha : (float) Percentatage of route to keep for building the next generation
        - n_shuffle : (int) constant for searching best combination among (n_shuffle) combination 
                      with the same genome
        - number_of_same_values : (int) stop codition. It stop after seeing (number_of_same_values) 
                                  same expectation over iteration

    
    Outputs :
    - generation : A list of all generations generated during the process
    - best_route : The best route  
    
    """

    def __init__(self, stations_map: StationsMap, number_of_combinations: int = 50, max_dis: float = 100,
                 max_iter: int = 1000, alpa: float = 0.10, n_shuffle: int = 5, number_of_same_values: int = 1000):
        self.number_of_same_values = number_of_same_values
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
        """
            calculating the maximizer function with 3 stopping conditions
            1) reaching maximum number of iteration
            2) reaching a plato (no change in the expectation)
            3) reaching expectation threshold (0.95 of the maximum expectation possible)
        """
        self.generation = [self.create_random_combinations()]
        self.generation[-1].sort_values("expectation", ascending=False, inplace=True)
        last_best_exp = self.generation[-1].head(1).iloc[0]["expectation"]
        counter_best = 0
        for _ in range(self.max_iter):
            generation = self.generation[-1]
            generation.sort_values("expectation", ascending=False, inplace=True)
            best_exp = generation.head(1).iloc[0]["expectation"]
            if last_best_exp == best_exp:  # breaking point reaching a plato
                counter_best += 1
                if counter_best >= self.number_of_same_values:
                    break
            else:
                counter_best = 0
            last_best_exp = best_exp
            if best_exp >= self.exp_threshold:  # breaking point reaching exp threshold
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
        """
        calculating the cost function.
        it will shuffle the station self.n_shuffle times or until he finds route with sufficient distance
        Parameters (input):
            - stations : tuple containing the route
        Outputs :
            - expectation : expectation of the route (-1 if no sufficient distance found)
            - distance : the distance of the root
            - stations : the ordered stations
        """
        dis = 0
        stations_np = np.array(stations)
        for _ in range(self.n_shuffle):
            np.random.shuffle(stations_np)
            exp, dis = self.compute_distance_and_expectation(stations_np)
            if dis <= self.max_dis:
                return exp, dis, tuple(stations_np)
        return -1, dis, stations

    def create_new_generation(self, generation: pd.DataFrame):
        """
        randomly create new generation, using genetic algorithm principle, cross over and mutation
        Parameters (input):
            - generation : top route of last generation
        Outputs :
            - new_gen  : new generation contains the top route of last generation and the new created generations
        """
        new_gen = generation.copy()
        while len(new_gen.index) <= self.number_of_combinations:
            cross_over = generation.sample(2)
            child1, child2 = self.single_point_crossover(cross_over.iloc[0], cross_over.iloc[1])
            child1 = self.mutation(child1)
            child2 = self.mutation(child2)
            child1.at[0, 'expectation'], child1.at[0, 'distance'], child1.at[0, 'stations'] = self.cost(
                child1.at[0, 'stations'])
            child2.at[0, 'expectation'], child2.at[0, 'distance'], child2.at[0, 'stations'] = self.cost(
                child2.at[0, 'stations'])
            new_gen = pd.concat([new_gen, child1, child2], ignore_index=True)
        return new_gen

    def mutation(self, child: pd.DataFrame, num_of_mutation: int = 1, probability: float = 0.5) -> pd.DataFrame:
        """
        randomly create a mutation (switching 1 bit in the code)
        this will change one of the station in the route
        Parameters (input):
            - child : a {0,1} code that represent a route
            - num_of_mutation: number of mutations
            - probability: probability that the mutation will happen
        Outputs :
            - the route after the mutation
        """
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
        """
        randomly create a crossover (switching the code of two route in random place)
        Parameters (input):
            - cross1,cross2 : a {0,1} code that represent a route
        Outputs :
            - the routes after the crossover
        """

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
        """
        creating random route
        we will use wight in order to create the code as we want to start with small route that will not reach the max distance
        Outputs :
            - df containing the routes
        """
        df = pd.DataFrame(columns=['stations', 'genome', 'expectation', 'distance'])
        n_station = len(self.stations)
        for _ in range(self.number_of_combinations):
            genome = np.array(random.choices([True, False], weights=(10, 90), k=n_station))
            stations = tuple(self.stations[genome])
            expectation, distance, stations = self.cost(stations)
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
    random.seed(1)
    size = 100
    m = np.random.randint(1, 10, size=(size + 1, size + 1))
    dis = np.triu(m, k=1)
    exp = np.random.randint(1, 10, size=size)

    mapStations = StationsMap(dis, exp)
    b = Optimize(
        mapStations
    )
    b.maximizer()
    print("time is:", datetime.now() - now)
