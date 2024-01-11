from functools import lru_cache
import numpy as np
import pandas as pd
import re

# (Berrington & Kingston 1987)
# https://iopscience.iop.org/article/10.1088/0022-3700/20/24/014/pdf
coefficient_temperatures = [1000, 2000, 5000, 10000]
species = ["11S", "23S", "21S", "23P", "21P", "33S", "31S", "33P", "33D", "31D", "31P"]

collision_table = {
    "11S": {
        "23S": [3.09e-2, 4.95e-2, 6.82e-2, 7.24e-2],
        "21S": [1.60e-2, 2.36e-2, 3.31e-2, 3.83e-2],
        "23P": [6.37e-3, 1.01e-2, 1.71e-2, 2.44e-2],
        "21P": [5.29e-3, 1.05e-2, 1.63e-2],
        "33S": [1.87e-2, 1.82e-2, 1.81e-2],
        "31S": [9.42e-3, 9.18e-3, 9.62e-3],
        "33P": [6.19e-3, 7.01e-3, 8.40e-3],
        "33D": [1.67e-3, 2.34e-3, 2.60e-3],
        "31D": [5.05e-3, 5.19e-3, 5.42e-3],
        "31P": [1.10e-3, 2.24e-3, 4.24e-3]},
    "23S": {
        "21S": [1.60e-0, 2.24e-0, 2.40e+0],
        "23P": [5.59e+0, 1.36e+1, 2.40e+1],
        "21P": [4.79e-1, 7.60e-1, 2.40e-1],
        "33S": [2.89e-0, 2.67e-0, 2.40e+0],
        "31S": [4.99e-1, 4.62e-1, 2.40e-1],
        "33P": [1.84e-0, 1.84e-0, 1.84e-0],
        "33D": [7.45e-1, 1.37e-0, 2.14e+0],
        "31D": [2.73e-1, 2.98e-1, 3.16e-1],
        "31P": [8.14e-2, 1.42e-1, 1.69e-1]},
    "21S": {
        "23P": [9.96e-1, 1.22e-0, 1.51e-0],
        "21P": [1.10e-0, 2.97e-0, 9.16e-0],
        "33S": [8.73e-1, 8.94e-1, 7.48e-1],
        "31S": [6.20e-1, 6.17e-1, 6.23e-1],
        "33P": [5.90e-1, 6.03e-1, 6.12e-1],
        "33D": [9.61e-2, 1.67e-1, 2.79e-1],
        "31D": [5.29e-1, 6.44e-1, 8.73e-1],
        "31P": [1.14e-1, 1.88e-1, 2.93e-1]},
    "23P": {
        "21P": [1.92e-0, 2.60e-0, 3.57e-0],
        "33S": [7.43e-0, 7.11e-0, 6.25e-0],
        "31S": [6.51e-1, 8.04e-1, 1.07e-0],
        "33P": [8.59e-0, 1.05e+1, 1.23e+1],
        "33D": [3.53e-0, 5.71e-0, 9.81e-0],
        "31D": [1.33e-0, 1.50e-0, 1.60e-0],
        "31P": [3.41e-1, 5.22e-1, 7.96e-1]},
    "21P": {
        "33S": [1.46e-0, 1.38e-0, 1.16e-0],
        "31S": [7.39e-1, 8.04e-1, 8.84e-1],
        "33P": [1.77e-0, 1.87e-0, 1.76e-0],
        "33D": [8.47e-1, 1.24e-0, 2.52e-0],
        "31D": [3.96e-0, 5.19e-0, 7.23e-0],
        "31P": [9.70e-1, 1.57e-0, 2.22e-0]},
    "33S": {
        "31S": [4.03e-0, 3.68e-0, 2.92e-0],
        "31D": [3.26e-0, 3.10e-0, 2.78e-0],
        "31P": [9.44e-1, 1.40e-0, 1.50e-0]},
    "31S": {
        "33P": [3.80e-0, 3.70e-0, 3.20e-0],
        "33D": [8.69e-1, 1.54e-0, 1.76e-0]},
    "33P": {
        "31D": [1.27e+1, 1.20e+1, 1.06e+1],
        "31P": [3.71e+0, 4.85e-0, 5.27e-0]},
    "33D": {
        "31D": [8.78e-0, 1.19e+1, 1.32e+1],
        "31P": [2.64e+0, 4.66e+0, 5.55e+0]},
    }
"""
@lru_cache
def get_effective_collision_strengths_table():
    collision_coefficients = np.zeros((len(species),len(species), len(coefficient_temperatures)))
    for state_i, subtable in collision_table.items():
        for state_j, coeff in subtable.items():
            collision_coefficients[species.index(state_i),species.index(state_j)] = coeff
    collision_coefficients = collision_coefficients + collision_coefficients.swapaxes(0,1)
    return collision_coefficients, species, coefficient_temperatures
"""

@lru_cache
def read_effective_collision_strengths_table():
    data = pd.read_csv("atomic data/Transition_rates.csv", delimiter=";")


    # The data is formatted wierdly, and we need to transform columns as such:
    # The first column "Transition" has some rows like "XXX-YYY" followed by lines with only "YYY" (ie implying XXX-YYY)
    # We need to transform this to a column "with XXX" and a column "to YYY"

    # we do this by splitting the column into two columns, and then filling the empty cells with the previous
    # non-empty cell. This is done with the following code:
    # All other columns are numbers formatted as eg "1.75", "9.52-2", "4.54+" or "1.75-2" which we want to 
    # interpret as "1.75", "9.52e-2", "4.54e+2" and "1.75e-2" respectively

    for column in data.columns[1:]:
        new_data = []
        for element in data[column]:
            element = element.replace("¹","1")
            element = element.replace("'","1")
            element = element.replace(",",".")
            parts = re.match("(\d\.\d\d)(.*)", element).groups()
            if parts[1] == "":
                parts = (parts[0],"+1")
            if parts[1] in ["+", "-"]:
                parts = (parts[0],parts[1] + "1")
            new_data.append(float(parts[0] + "e" + parts[1]))
        data[column] = np.array(new_data)
    data["Transition"] = data["Transition"].str.split("-")
    state_1 = data["Transition"].apply(lambda x: x[0] if len(x) == 1 else x[1])
    state_2 = data["Transition"].apply(lambda x: np.nan if len(x) == 1 else x[0])
    state_2 = state_2.ffill()
    species = list(np.union1d(state_1, state_2))
    collision_coefficients = np.zeros((len(species),len(species), len(data.columns[1:])))
    for row, i, j in zip(range(data.shape[0]), state_1, state_2):
        row_data = data.iloc[row, 1:].to_numpy(dtype=float)
        collision_coefficients[species.index(i), species.index(j), :] = row_data
        collision_coefficients[species.index(j), species.index(i), :] = row_data
    return collision_coefficients, species, data.columns[1:].str.replace(" ", "").to_numpy(dtype=float)

# rewrites the columns and rows to be in the order of the state_list, and inserts zeros for missing values (if any)
# in return for a warning
@lru_cache
def get_effective_collision_strengths_table(state_list):
    gamma_table, species, temperatures = read_effective_collision_strengths_table()
    missing_states = np.setdiff1d(state_list, species)
    if len(missing_states) > 0:
        print("Error: missing states in collision table: ", missing_states)
    state_indices = [species.index(state) for state in state_list]
    gamma_table = gamma_table[state_indices, :, :][:, state_indices, :]
    return gamma_table, temperatures