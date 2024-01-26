from functools import lru_cache
import numpy as np
import pandas as pd
import re

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
        print("Warning: missing states in collision table: ", missing_states)
        print("Inserting ones for missing states")
        padded_gamma_table = np.ones([gamma_table.shape[0] + len(missing_states)]*2 + [len(temperatures)])
        padded_gamma_table[:gamma_table.shape[0], :gamma_table.shape[1], :] = gamma_table
        gamma_table = padded_gamma_table
        species = species + list(missing_states)
    state_indices = [species.index(state) for state in state_list]
    return gamma_table[state_indices, :, :][:, state_indices, :], temperatures