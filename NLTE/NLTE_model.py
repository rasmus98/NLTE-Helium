from scipy.interpolate import interp1d
import numpy as np
import pandas
import astropy.units as u
import astropy.constants as consts
from astropy.modeling.physical_models import BlackBody
from dataclasses import dataclass, field
from scipy.integrate import quad, solve_ivp
from functools import lru_cache, partial
import NLTE.collision_rates
import io 
import re
import warnings


#################### Hyperparameters: Included atomic levels and physical parameters ####################

# TODO: handle fine structure level splitting properly. For now, we just assume equal splitting
# This is valid in the case of much higher transition rates between the ortho levels, than from either of them to the para levels
@dataclass 
class States:
    names : np.array = field(default_factory=lambda: ["11S", "23S", "21S", "23P", "21P", "33S", "31S", "33P", "33D", "31D", "31P"])#, "41S", "41P", "41D", "41F", "43s", "43P", "43D", "43F"])
    multiplicities : np.array = np.array([1, 3, 1, 9, 3, 3, 1, 9, 15, 5, 3])
    energies : np.array = np.array([0.00, 19.819614525, 20.615774823, 20.96408688908, 21.2180227112, 22.718466419, 22.920317359, 23.00707314571, 23.07365070854, 23.07407479777, 23.08701852960]) * u.eV
    ionization_species : np.array = field(default_factory=lambda: ["HeII", "HeIII"])

    def __post_init__(self):
        self.all_names = self.names + self.ionization_species

    def __hash__(self):
        return hash(tuple(self.names))
    
    def __eq__(self, __value: object) -> bool:
        return self.names == __value.names and \
        (self.multiplicities == __value.multiplicities).all() and \
        (self.energies == __value.energies).all()

    def filter(self, names):
        mask = np.isin(self.names, names)
        return States(list(np.array(self.names)[mask]), self.multiplicities[mask], self.energies[mask])
    
    def add_state(self, name, multiplicity, energy):
        self.names = np.append(self.names, name)
        self.multiplicities = np.append(self.multiplicities, multiplicity)
        self.energies = np.append(self.energies, energy)
    
    def get_fancy_names(self):
        return {name: f"$1s{name[0:-2]}{name[-1].lower()}^{name[-2]}{name[-1]}$" for name in self.names} | {"HeII": "$He^{+}$", "HeIII": "$He^{2+}$"}
    
    @staticmethod
    def read_states(filter = lambda table: table["n"] <= 4):
        nist_table = pandas.read_csv("atomic data/levels_nist.csv", delimiter=r",", usecols=lambda x: x not in ["Prefix", "Suffix"], dtype=str)
        nist_table = nist_table.apply(lambda x: x.str.removeprefix("=\"").str.removesuffix("\""), axis=1)
        nist_table = nist_table.dropna(subset=['g']) # drop the 
        for col in nist_table.columns:
            nist_table[col] = nist_table[col].str.removeprefix("=\"").str.removesuffix("\"")
            if col in ["Level (eV)", "Uncertainty (eV)"]:
                nist_table[col] = pandas.to_numeric(nist_table[col])
            if col in ["j", "g"]:
                nist_table[col] = pandas.to_numeric(nist_table[col])
        nist_table = nist_table[nist_table["Level (eV)"] < 30]
        nist_table["n"] = nist_table["Configuration"].str.split(".").apply(lambda x: sum(int(re.match("(\d+)(\w)(2?)", y).group(1))-1 for y in x)+1)

        nist_table["name"] = nist_table["n"].astype(str) + nist_table["Term"].str.replace("*", "")
        #for i, row in nist_table[nist_table["n"] <= 4].iterrows():
            #print(row)
        states = nist_table.groupby("name").agg({"Level (eV)": "min", "n": "min", "g" : "sum"}).sort_values("Level (eV)")
        selected_states = states[filter(states)]
        return States(list(selected_states.index.values), selected_states["g"].values, selected_states["Level (eV)"].values * u.eV)

# Radial density profile of the ejecta, solving the normalisation constant for a given ejecta mass

rho = lambda v,t, rho0, p=5, v_0=0.2, t_0=1 : rho0 * (v/v_0)**-p * (t/t_0)**-3
# M = int dM/dV*dV/dr*dr/dv wrt dv
# r = 1 day * v
# M = int rho(v)*dV/dr*dr/dv wrt dv
# output in g/c
def dMdv(v, t):
    v = v*consts.c
    t = t*u.day
    dr_dv = 1*u.day
    r = 1*u.day * v
    dV_dr = 4*np.pi*r**2
    dM_dV = rho(v, t, 1*u.g/u.cm**3, v_0=0.2*consts.c, t_0=1*u.day)
    return (dM_dV.cgs * dV_dr.cgs * dr_dv.cgs * consts.c).cgs.value

@lru_cache
def get_density_profile(M_ej, atomic_mass, mass_fraction):
    M_ej = M_ej * u.M_sun
    atomic_mass = atomic_mass * u.u
    # we calculate the density profile at t=1 day, and normalize it to 0.04 solar mass
    rho_0 =  M_ej / (quad(dMdv, 0.1, 0.5, args=1)[0] * u.g/u.cm**3)
    number_density_0 = (rho_0 * mass_fraction / atomic_mass).cgs.value
    return lambda v, t: rho(v, t, number_density_0)

# Environment class, contains all the parameters of the environment at a given time and radius, the inputs to the NLTE calculation
# The following parameters are calculated from the input parameters:
#   - Doppler shifted temperature
#   - Electron density
#   - Helium density
#   - Radiative power of non-thermal electrons
@dataclass
class Environment:
    # input values (can be set as named parameters to the constructor)
    t_d: float = 1.43 # days. Yes Days. The second cgs unit of time, apparently.
    T_phot: float = 4400  # K
    M_ejecta: float = 0.04 # solar masses ejected
    mass_fraction: float = 0.002 # mass fraction of helium
    atomic_mass: float = 4 # atomic mass of helium [u]
    photosphere_velocity: float = 0.245 # photosheric velocity as a fraction of c
    line_velocity: float = 0.245 # velocity of the region to calculate at as a fraction of c

    # calculated values (Will be calculated from the input values)
    spectrum : BlackBody = None # Experienced spectrum at the ROI. Contains the doppler shifted temperature
    T_electrons: float = None # K temperature of the electrons (doppler shifted photosphere temperature)
    n_e: float = None # count/cm^3	
    n_He: float = None # count/cm^3
    q_dot: float = None # eV/s/ion
    # Calculate derived values based on the input values
    def __post_init__(self):
        # Doppler shifted temperature according to the paper. Note that the paper incorrectly did not do this
        delta_v = self.line_velocity - self.photosphere_velocity
        # commenting out the dopler shift reproduce Tarumi's results
        self.T_electrons = self.T_phot/(1/np.sqrt(1 - delta_v**2) * (1+delta_v)) # Doppler shifted temperature
        W = 0.5*(1-np.sqrt(1-(self.photosphere_velocity/self.line_velocity)**2)) # geometric dilution factor
        self.spectrum = BlackBody(self.T_electrons * u.K, scale=W*u.Unit("erg/(s Hz sr cm2)")) 
        self.n_e = (1.5e8*self.t_d**-3) * (self.line_velocity/0.284)**-5 # Extracted from the paper, see electron_model_reconstruction.ipynb
        self.n_He = get_density_profile(self.M_ejecta, self.atomic_mass, self.mass_fraction)(self.line_velocity, self.t_d)
        self.q_dot = 1 * self.t_d**-1.3 # Radiative power of non-thermal electrons

# Solves the NLTE equations for the given environment and states
# The solver uses the sobolev depth method to solve the NLTE equations
# Iteration works follows:
# First the NLTE equations are solved assuming optically thin conditions, 
# then the sobolev depth is calculated, the transition rates are adjusted, and the NLTE equations are solved again
# These are 
def solve_NLTE_sob(environment, states, relaxation_steps = 5): 
    nlte_solver = NLTE.NLTE_model.NLTESolver(environment, states=states)
    radiative_process = nlte_solver.processes[1]
    # save the original rates
    A = radiative_process.A 
    absorbtion_rate = radiative_process.arbsorbtion_rate
    stimulated_emission_rate = radiative_process.stimulated_emission_rate

    # perform a few relaxation steps to get the correct sobolev depth
    for _ in range(relaxation_steps):
        _, y = nlte_solver.solve(1e6)
        n = y.T[-1,:-2] * u.cm**-3
        eps0 = 1/(4 * np.pi) # i dont like gauss units
        tau = np.zeros((len(n), len(n)))
        # The sobolev depth is calculated individually for each transition
        # Yes i know, loop, very inefficient, but this is a small matrix
        for i in range(len(n)): 
            for j in range(len(n)):
                dE = states.energies[j] - states.energies[i]
                if dE <= 0:
                    continue
                lam = dE.to("cm", equivalencies=u.spectral())
                tau[i,j] = lam**3*eps0 * A[i,j]* u.s**-1 * n[i]/2 * (states.multiplicities[j]/ states.multiplicities[i] - n[j]/n[i]) * environment.t_d * u.day
        # calculate optical depth and escape probability        
        tau = np.maximum(tau,tau.T)+1e-8
        beta = np.array((1-np.exp(-tau)) / tau)
        # adjust the transition rates
        radiative_process.A = A * beta
        radiative_process.arbsorbtion_rate = absorbtion_rate * beta
        radiative_process.stimulated_emission_rate = stimulated_emission_rate * beta
    return nlte_solver.solve(1e6)[1][:,-1], nlte_solver
        

# primary class, contains all the states and processes, and solves the system of differential equations
class NLTESolver:
    def __init__(self, environment, states = States(), processes = None):
        self.states = states
        self.environment = environment
        if processes is None:
            self.processes = [CollisionProcess(states, environment), 
                              RadiativeProcess(states, environment), 
                              PhotoionizationProcess(states, environment),
                              RecombinationProcess(states, environment), 
                              HotElectronIonizationProcess(states, environment)]
        else:
            self.processes = processes

    def get_transition_rate_matrix(self):
        return sum([process.get_transition_rate_matrix() for process in self.processes])

    def solve(self, times):        
        rate_matrix = self.get_transition_rate_matrix()
        np.fill_diagonal(rate_matrix, -np.sum(rate_matrix, axis=0) + np.diag(rate_matrix))
        initial = np.ones(len(self.states.all_names)) / len(self.states.all_names)
        diff_eq = lambda t, n: rate_matrix@n
        if isinstance(times, np.ndarray):
            solution = solve_ivp(diff_eq, (0, max(times)), 
                             initial, t_eval=times, method="LSODA",  rtol=1e-8, atol=1e-40)
        else:   
            solution = solve_ivp(diff_eq, (0, times), 
                                 initial, method="LSODA",  rtol=1e-8, atol=1e-40)
        return solution.t, solution.y * self.environment.n_He
        

# Handles state -> state transitions due to electron collisions
class CollisionProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.collision_rates = self.get_collision_rates()
        self.name = "Collision"
        
    def get_collision_rates(self):
        gamma_table, temperatures = NLTE.collision_rates.get_effective_collision_strengths_table_Kington(tuple(self.states.names))
        gamma = interp1d(temperatures, gamma_table, bounds_error=True)(self.environment.T_electrons) 
        E_diff = self.states.energies[:,np.newaxis] - self.states.energies[None,:]
        exponential = np.exp(-np.maximum(E_diff, 0*u.eV) / (consts.k_B * self.environment.T_electrons * u.K))
        return 8.63*10**-6/(np.sqrt(self.environment.T_electrons) * self.states.multiplicities[None,:]) * gamma * exponential
        
    def get_transition_rate_matrix_Kingston(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        coeff_mat[:len(self.states.names), :len(self.states.names)] = self.collision_rates
        return coeff_mat*self.environment.n_e 
    
    def get_transition_rate_matrix(self):
        return NLTE.collision_rates.get_collision_rates_Ralchenko(self.states, self.environment.T_electrons)*self.environment.n_e 
    
# Handles state -> state transitions due to radiative processes (spontaneous emission, stimulated emission and absorption)
class RadiativeProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.A, self.arbsorbtion_rate, self.stimulated_emission_rate = self.get_einstein_rates()
        self.name = "Radiative"

    # calculates the naural decay, arbsorbtion rate and stimulated emission rate
    def get_einstein_rates(self):
        A = get_A_rates(tuple(self.states.names)) * u.s**-1
        E_diff = self.states.energies - self.states.energies[:,np.newaxis]
        nu = np.maximum(np.abs(E_diff.to(u.Hz, equivalencies=u.spectral())), 1 * u.Hz)
        F_nu = (2 * consts.h * nu**3) / consts.c**2
        B_stimulation = A / F_nu
        B_absorbtion = B_stimulation.T * self.states.multiplicities[:,np.newaxis] / self.states.multiplicities
        rho_nu = u.sr * self.environment.spectrum(nu)
        stimulation_rate = rho_nu * B_stimulation
        absorbtion_rate = rho_nu * B_absorbtion
        return A.to("1/s").value, stimulation_rate.to("1/s").value, absorbtion_rate.to("1/s").value
        

    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        coeff_mat[:len(self.states.names), :len(self.states.names)] = self.A + self.arbsorbtion_rate + self.stimulated_emission_rate
        return coeff_mat
    
# Handles state -> state transitions due to photoionization
class PhotoionizationProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.ionization_rates = get_ionization_rates(states, environment.spectrum)
        self.name = "Photoionization"
        
    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        coeff_mat[self.states.all_names.index("HeII"), :len(self.states.names)] = self.ionization_rates
        return coeff_mat
        
class RecombinationProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.alphaHI = get_HI_recombination_funcs(tuple(states.names))
        self.alphaHII = get_HII_recombination_func()
        self.name = "Recombination"

    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        T = self.environment.T_electrons
        for name, func in self.alphaHI.items():
            coeff_mat[self.states.all_names.index(name), self.states.all_names.index("HeII")] = func(np.log10(T)) * self.environment.n_e
        coeff_mat[self.states.all_names.index("HeII"), self.states.all_names.index("HeIII")] = self.alphaHII(np.log10(T)) * self.environment.n_e
        return coeff_mat
    
class HotElectronIonizationProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.w = [593, 3076] # work per ionization in eV for HeII and HeIII respectively
        self.name = "Non-thermal electrons"
        
    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        names = self.states.all_names
        # TODO: fix back
        coeff_mat[names.index("HeII"), :len(self.states.names)] = self.environment.q_dot / self.w[0] 
        # coeff_mat[names.index("HeII"), names.index("11S")] = self.environment.q_dot / self.w[0]
        coeff_mat[names.index("HeIII"), names.index("HeII")] = self.environment.q_dot / self.w[1]
        return coeff_mat 

    
@lru_cache
def get_ionization_dict():
    with open("atomic data/he1.txt") as f:
        text = f.read()
    sections = re.split(r"^((?:\s+\d+){4,6}(?:\s+[\d.E+-]+){0,2})$", text, flags=re.MULTILINE)
    species = dict()

    for state, content in zip(sections[1::2], sections[2::2]):
        if content == "\n":
            continue
        spin_multiplicity, orbital_l, parity, n_symetry = map(int, state.split()[:4])
        n = n_symetry + orbital_l
        if orbital_l == 0 and spin_multiplicity == 3:
            n = n + 1 # because the 13s state does not exist, the 23s is first in the symmetry
        l = "SPDFGHIK"[orbital_l]
        state = str(n) + str(spin_multiplicity) + l
        (energies, cross) = np.loadtxt(io.StringIO(content), unpack=True)
        if len(energies) > 0:
            species[state] = (energies, cross)
    return species

def get_ionization_rates(states, spectrum):
    ionization_rates = []
    ionization_dict = get_ionization_dict()
    for state in states.names:
        energies, crossection = ionization_dict[state]
        E = (energies * u.Ry).cgs
        nu = E.to(u.Hz, equivalencies=u.spectral()) 
        sigma = (crossection * u.Mbarn).cgs
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            ionization_flux_article = u.sr * sigma * (spectrum(nu)/E)
        ionization_rates.append(np.trapz(x=nu, y=ionization_flux_article).to(1/u.s).value) 
    return np.array(ionization_rates) * 4 * np.pi # TODO find out why the 16*pi is needed

@lru_cache
def get_A_table():
    get_n = lambda n, l, count: (int(n)-1)*2 if count else (int(n)-1)
    nist_table = pandas.read_csv("atomic data/A_rates_NIST.csv", dtype=str)
    nist_table = nist_table.apply(lambda x: x.str.removeprefix("=\"").str.removesuffix("\"") if x.dtype == object else x, axis=1)
    nist_table = nist_table[(nist_table['Aki(s^-1)'] != "")] # drop lines of no interest

    def get_state_name(config_series, term_series):
        n = config_series.str.findall("(\d+)(\w)(2?)").apply(lambda x: str(1+sum([get_n(*nlm) for nlm in x])))
        return n+term_series.str.replace("*", "")

    nist_table["lower_name"] = get_state_name(nist_table["conf_i"], nist_table["term_i"])
    nist_table["upper_name"] = get_state_name(nist_table["conf_k"], nist_table["term_k"])
    nist_table["A_rates"] = pandas.to_numeric(nist_table["Aki(s^-1)"])
    nist_table["g_k"] = pandas.to_numeric(nist_table["g_k"])
    nist_table["g_i"] = pandas.to_numeric(nist_table["g_i"])
    return nist_table

@lru_cache
def get_A_rates(names):
    nist_table = NLTE.NLTE_model.get_A_table()
    A_coefficients = np.zeros((len(names), len(names)))
    for (lower_name, upper_name, _), subtable in nist_table.groupby(["lower_name", "upper_name", "J_i"]):
        if not (lower_name in names and upper_name in names):
            continue

        weighted_A = np.average(subtable["A_rates"], weights = subtable["g_k"])
        A_coefficients[names.index(lower_name),names.index(upper_name)] += weighted_A
    return A_coefficients

@lru_cache()
def load_cross_sections():
    all_states = NLTE.NLTE_model.States.read_states(lambda table: (table["n"] <= 4))
    get_E = lambda name: all_states.energies[all_states.names.index(name)]
    get_g = lambda name: all_states.multiplicities[all_states.names.index(name)]

    def get_cross_sections(filename, fit_function):
        table = pandas.read_csv(filename, skiprows=6, skipfooter=5, engine="python", sep="\s+", index_col=[0,1], header=None, skip_blank_lines=True)
        clamped_fit_function = lambda E, A, i, f: np.where(E>get_E(f) - get_E(i), fit_function(E/(get_E(f) - get_E(i)), A), 0) * np.pi * consts.a0**2 * u.Ry / (get_g(i) * E)
        return {(i,f): partial(clamped_fit_function, A=A, i = i, f=f) for (i,f), A in table.T.to_dict("list").items()}

    sigmas = get_cross_sections("atomic data/dipole-allowed.csv",    lambda x, A: (A[0]*np.log(x) + A[1] + A[2]/x + A[3]/x**2 + A[4]/x**3)*(x+1)/(x+A[5]))\
        | get_cross_sections("atomic data/dipole-forbidden.csv", lambda x, A: (A[0] + A[1]/x + A[2]/x**2 + A[3]/x**3)*(x**2)/(x**2+A[4]))\
        | get_cross_sections("atomic data/spin-forbidden.csv",   lambda x, A: (A[0] + A[1]/x + A[2]/x**2 + A[3]/x**3)*(1)/(x**2+A[4]))
    return sigmas

# Returns the collision strenths, which should be mutliplied with the electron density to get the rates
@lru_cache()
def get_collision_strengths(select_state_names, T):
    all_states = NLTE.NLTE_model.States.read_states(lambda table: (table["n"] <= 4))
    get_E = lambda name: all_states.energies[all_states.names.index(name)]
    get_g = lambda name: all_states.multiplicities[all_states.names.index(name)]
    sigmas = load_cross_sections()
    electron_v_distibution = lambda v: (consts.m_e /(2*np.pi*consts.k_B * T))**(3/2) * 4* np.pi * v**2 * np.exp(-consts.m_e * v**2 /(2*consts.k_B * T))
    v_to_E = lambda v: 1/2 * consts.m_e * v**2
    integrand = lambda v, sigma: electron_v_distibution(v) * v * sigma(v_to_E(v))
    E_range = np.geomspace(1e-7, 1e5, 10000) * u.eV
    v_range = np.sqrt(2*E_range/consts.m_e)
    index = list(select_state_names)
    collision_strengths = np.zeros((len(index), len(index)))
    for (i,f), sigma in sigmas.items(): 
        if i not in index or f not in index:
            continue
        # these are excitations, so i is the lower state and f is the upper state
        # rate_matrix[i,f] on the other hand is the rate from f to i
        i_index = index.index(i)
        f_index = index.index(f)
        rate = np.trapz(integrand(v_range, sigma), v_range).to(u.cm**3/u.s).value
        collision_strengths[f_index, i_index] = rate
        # Calculate the deexcitation rate from f to i
        # Easier to come down if lower multiplicity is lower, and if the energy difference greater
        w_ratio =  get_g(i) / get_g(f)
        delta_E = get_E(f) - get_E(i)
        collision_strengths[i_index, f_index] = rate * w_ratio * np.exp(delta_E/(consts.k_B* T))
    return collision_strengths

# Calculate recombination coefficients
@lru_cache()
def get_HII_recombination_func():   
    return interp1d([3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0], 
                    [9.73e-12, 8.42e-12, 7.28e-12, 6.28e-12, 5.42e-12, 4.67e-12, 4.02e-12, 3.46e-12, 2.96e-12, 2.55e-12, 2.18e-12])

# Returns the collision strenths, which should be mutliplied with the electron density to get the rates
@lru_cache()
def get_HI_recombination_funcs(select_state_names):
    T, recombination_dict, total_recomb_rate = read_recombination_data()
    subset_dict = {name: data for name, data in recombination_dict.items() if name in select_state_names and name != "11S"}
    subset_sum = sum(subset_dict.values(), T*0)
    ground_coeff = recombination_dict["11S"]
    normalization_coeff = (total_recomb_rate - ground_coeff) / subset_sum
    renormalized_dict = {name: interp1d(np.log10(T), data * normalization_coeff) for name, data in subset_dict.items() }
    renormalized_dict["11S"] = interp1d(np.log10(T), ground_coeff)
    return renormalized_dict

def read_recombination_data():
    with open("atomic data/he1.rrc.ls.txt") as recomb_data:
        filecontent = recomb_data.readlines()
    
    T = np.array(list(map(float,"".join(filecontent[298:312]).strip().replace("(Ry)",  "").replace("(K)", "").replace("\n", "").split())))

    coeff_filecontent = "".join(filecontent[314:])
    exp_number_format = r"[-+]?\d\.\d+E[+-]\d{2}"
    recombination_dict = {}
    for statelabel, title, data in re.findall(f"(?:(\d{{8}}\.\d{{4}}) {exp_number_format}|(.*)=)\s+((?:{exp_number_format}\s+)+)", coeff_filecontent):
        recombination_rates = np.array(list(map(float, data.strip().split())))
        if statelabel:
            n = int(statelabel[-4:-2])
            n = max(1,n) # for some fucking reason the ground state is n=0, but the first exited is n=2
            l = int(statelabel[-2])
            label = f"{n}{statelabel[0]}{'SPDFGHIK'[l]}"
            recombination_dict[label] = recombination_rates
        else:
            if title == "with dr rc added, net total":
                total_recomb_rate = recombination_rates
    return T, recombination_dict, total_recomb_rate