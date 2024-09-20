from functools import lru_cache, partial
import numpy as np
import pandas as pd
import re
import astropy.units as u
import astropy.constants as const


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
def get_effective_collision_strengths_table_Kington(state_list):
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

@lru_cache
def get_sigmas(states):
    get_E = lambda name: states.energies[states.names.index(name)]
    get_g = lambda name: states.multiplicities[states.names.index(name)]
    def get_cross_sections(filename, fit_function):
        table = pd.read_csv(filename, skiprows=6, skipfooter=5, engine="python", sep="\s+", index_col=[0,1], header=None, skip_blank_lines=True)
        clamped_fit_function = lambda E, A, i, f: np.where(E>get_E(f) - get_E(i), fit_function(E/(get_E(f) - get_E(i)), A), 0) * np.pi * const.a0**2 * u.Ry / (get_g(i) * E)
        return {(i,f): partial(clamped_fit_function, A=A, i = i, f=f) for (i,f), A in table.T.to_dict("list").items()}

    # Load all transitions as a dictionary (lower, upper) : crossection(E)
    return get_cross_sections("atomic data/dipole-allowed.csv",    lambda x, A: (A[0]*np.log(x) + A[1] + A[2]/x + A[3]/x**2 + A[4]/x**3)*(x+1)/(x+A[5]))\
         | get_cross_sections("atomic data/dipole-forbidden.csv", lambda x, A: (A[0] + A[1]/x + A[2]/x**2 + A[3]/x**3)*(x**2)/(x**2+A[4]))\
         | get_cross_sections("atomic data/spin-forbidden.csv",   lambda x, A: (A[0] + A[1]/x + A[2]/x**2 + A[3]/x**3)*(1)/(x**2+A[4]))


@lru_cache
def get_collision_rates_Ralchenko(states, T):
    T = T * u.K
    T_ev = T.to(u.eV, equivalencies=u.temperature_energy())

    get_E = lambda name: states.energies[states.names.index(name)]
    get_g = lambda name: states.multiplicities[states.names.index(name)]

    electron_v_distibution = lambda v: (const.m_e /(2*np.pi*const.k_B * T))**(3/2) * 4* np.pi * v**2 * np.exp(-const.m_e * v**2 /(2*const.k_B * T))
    sigmas = get_sigmas(states)
    v_to_E = lambda v: 1/2 * const.m_e * v**2
    integrand = lambda v, sigma: electron_v_distibution(v) * v * sigma(v_to_E(v))

    rate_matrix = np.zeros((len(states.all_names), len(states.all_names)))
    for (lower,upper), sigma in sigmas.items(): 
        if lower not in states.names or upper not in states.names:
            continue
        # calculate excitations rates:
        lower_index = states.names.index(lower)
        upper_index = states.names.index(upper)
        w_ratio =  get_g(lower) / get_g(upper)
        delta_E = get_E(lower)- get_E(upper)

        E_range = np.geomspace(-delta_E.to(u.eV).value, T_ev.value*1e2, 1000) * u.eV
        v_range = np.sqrt(2*E_range/const.m_e)
        rate = np.trapz(integrand(v_range, sigma), v_range).cgs.value
        rate_matrix[upper_index, lower_index] = rate
        # then canculate inverse deexcitation rates
        # Easier to come down if lower multiplicity is higher, and if the energy difference greater
        rate_matrix[lower_index, upper_index] = rate * w_ratio * np.exp(-(delta_E/(const.k_B* T)).cgs.value)
    return rate_matrix