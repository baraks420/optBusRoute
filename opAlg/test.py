import random
from datetime import datetime

import pandas as pd
import numpy as np

from create_map import StationsMap
from utils import plot_best_route, plot_iteration
from genetic_algorithem import Optimize



# Define the graph array
no = np.inf # In case there is not a route (edge) between station, the distance is infinit

dis = np.array([
    [0, no, 3.0, no, no, no, no, 1, no, no, no, no, no, 2, no, 4.3],
    [0, 0, no, no, no, no, no, no, no, no, 2.2, 3.2, no, no, 8, no],
    [0, 0, 0, 5, no, no, no, no, no, no, no, no, no, no, no, no],
    [0, 0, 0, 0, no, no, no, no, no, no, no, no, 2.5, 3.3, no, no],
    [0, 0, 0, 0, 0, no, no, no, no, 1, no, 1.5, no, 7.5, no, no],
    [0, 0, 0, 0, 0, 0, no, no, 2.3, no, no, 2, no, 6, no, no],
    [0, 0, 0, 0, 0, 0, 0, no, no, 1.2, no, 5.1, 3.1, no, no, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 1.2, no, no, no, no, 5.1, 2, 1.8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, no, no, no, no, no, 2, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no, no, 2.6, no, no, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no, 7.1, no, no, 8.9],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no, no, no, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no, no, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, no],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# station expectations (From Station 1 to station n) 
# The station 0 is the starting point
exp = [10, 7, 5, 3, 7, 11, 10, 4, 2, 5, 9, 11, 6, 4, 9]


mapStations = StationsMap(dis, exp)
now = datetime.now()

nb_comb = 100
alpha = 0.3


b = Optimize(
    mapStations,
    max_dis=30,
    number_of_combinations=nb_comb,
    alpa = alpha
)
b.maximizer()

print("************* nb_comb = ",nb_comb," and alpha = ",alpha,"****************")
print("computation time is: ", datetime.now() - now)
print(b.best_route)
sum_exp = b.best_route["expectation"][0]
plot_best_route(dis,exp,b.best_route["stations"][0],nb_comb,alpha,sum_exp)
plot_iteration(b.generation,nb_comb,alpha)
