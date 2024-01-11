from scipy.interpolate import interp1d
import numpy as np
import pandas
import astropy.units as u
import astropy.constants as consts
from astropy.modeling.physical_models import BlackBody
from dataclasses import dataclass, field
from scipy.integrate import quad, solve_ivp
from functools import lru_cache
import NLTE.collision_rates
import io 
import re
import warnings


#################### Hyperparameters: Included atomic levels and physical parameters ####################

# TODO: handle fine structure level splitting properly. For now, we just assume equal splitting
# This is valid in the case of much higher transition rates between the ortho levels, than from either of them to the para levels
@dataclass 
class States:
    names : np.array = field(default_factory=lambda: ["11S", "23S", "21S", "23P", "21P", "33S", "31S", "33P", "33D", "31D", "31P"])
    multiplicities : np.array = np.array([1, 3, 1, 9, 3, 3, 1, 9, 15, 5, 3])
    energies : np.array = np.array([0.00, 19.819614525, 20.615774823, 20.96408688908, 21.2180227112, 22.718466419, 22.920317359, 23.00707314571, 23.07365070854, 23.07407479777, 23.08701852960]) * u.eV
    ionization_species : np.array = field(default_factory=lambda: ["HeII", "HeIII"])

    def __post_init__(self):
        self.all_names = self.names + self.ionization_species

    def filter(self, names):
        mask = np.isin(self.names, names)
        return States(list(np.array(self.names)[mask]), self.multiplicities[mask], self.energies[mask])
    
    def add_state(self, name, multiplicity, energy):
        self.names = np.append(self.names, name)
        self.multiplicities = np.append(self.multiplicities, multiplicity)
        self.energies = np.append(self.energies, energy)

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
@dataclass
class Environment:
    # input values
    t_d: float = 1.43 # days. Yes Days. The second cgs unit of time, apparently.
    T_phot: float = 4400  # K
    M_ejecta: float = 0.04 # solar masses ejected
    mass_fraction: float = 0.002 # mass fraction of helium
    atomic_mass: float = 4 # atomic mass of helium [u]
    photosphere_velocity: float = 0.245 # photosheric velocity as a fraction of c
    line_velocity: float = 0.245 # velocity of the region to calculate at as a fraction of c
    #q_dot_model = lambda self, t_d: 1 * t_d**-1.3

    spectrum : BlackBody = None # Experienced spectrum at the ROI. Contains the doppler shifted temperature
    T_electrons: float = None # K temperature of the electrons (doppler shifted photosphere temperature)
    n_e: float = None # count/cm^3	
    n_He: float = None # count/cm^3
    q_dot: float = None # eV/s/ion
    # Calculate derived values based on the input values
    def __post_init__(self):
        # Doppler shifted temperature according to the paper. Note that the paper incorrectly did not do this
        delta_v = self.line_velocity - self.photosphere_velocity
        # TODO: fix back
        self.T_electrons = self.T_phot/(1/np.sqrt(1 - delta_v**2) * (1+delta_v)) 
        W = 0.5*(1-np.sqrt(1-(self.photosphere_velocity/self.line_velocity)**2))
        self.spectrum = BlackBody(self.T_electrons * u.K, scale=W*4*np.pi*u.Unit("erg/(s Hz sr cm2)")) 
        self.n_e = (1.5e8*self.t_d**-3) * (self.line_velocity/0.284)**-5 # Extracted from the paper, see electron_model_reconstruction.ipynb
        self.n_He = get_density_profile(self.M_ejecta, self.atomic_mass, self.mass_fraction)(self.line_velocity, self.t_d)
        self.q_dot = 1 * self.t_d**-1.3

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
        np.fill_diagonal(rate_matrix, -np.sum(rate_matrix, axis=0))
        initial = np.ones(len(self.states.all_names)) / len(self.states.all_names)
        diff_eq = lambda t, n: rate_matrix@n
        if isinstance(times, np.ndarray):
            solution = solve_ivp(diff_eq, (0, max(times)), 
                             initial, t_eval=times, method="LSODA",  rtol=1e-6, atol=1e-40)
        else:   
            solution = solve_ivp(diff_eq, (0, times), 
                                 initial, method="LSODA",  rtol=1e-6, atol=1e-40)
        return solution.t, solution.y * self.environment.n_He
        

# Handles state -> state transitions due to electron collisions
class CollisionProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.collision_rates = self.get_collision_rates()
        self.name = "Collision"
        
    def get_collision_rates(self):
        gamma_table, temperatures = NLTE.collision_rates.get_effective_collision_strengths_table(tuple(self.states.names))
        gamma = interp1d(temperatures, gamma_table, bounds_error=True)(self.environment.T_electrons) 
        E_diff = np.maximum(self.states.energies[:,np.newaxis] - self.states.energies, 0*u.eV)
        exponential = np.exp(-E_diff / (consts.k_B * self.environment.T_electrons * u.K))
        return 8.63*10**-6/(np.sqrt(self.environment.T_electrons) * self.states.multiplicities) * gamma * exponential
        
    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        coeff_mat[:len(self.states.names), :len(self.states.names)] = self.collision_rates
        return coeff_mat*self.environment.n_e
    
class RadiativeProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.A, self.arbsorbtion_rate, self.stimulated_emission_rate = self.get_einstein_rates()
        self.name = "Radiative"

            
    def get_A_rates(self):
        get_n = lambda n, l, count: (int(n)-1)*2 if count else (int(n)-1)
        nist_table = pandas.read_csv("atomic data/A_rates_NIST.csv")
        def get_state_name(config_series, term_series):
            n = config_series.str.findall("([\d+]+)([spdf])(2?)").apply(lambda x: str(1+sum([get_n(*nlm) for nlm in x])))
            term = term_series.str.strip("=\"*")
            return n+term

        i_state = get_state_name(nist_table["conf_i"], nist_table["term_i"])
        j_state = get_state_name(nist_table["conf_k"], nist_table["term_k"])
        names = self.states.names
        A_coefficients = np.zeros((len(names), len(names)))

        for state_i in names:
            selection = (i_state == state_i) & (j_state.isin(names)) & (nist_table["Aki(s^-1)"] != '=""')
            selected_A = nist_table[selection]["Aki(s^-1)"].str.strip("=\"*")
            for state_j, A in zip(j_state[selection], selected_A):
                A_coefficients[self.states.names.index(state_i),self.states.names.index(state_j)] = float(A)
        
        A_coefficients[names.index("11S"), names.index("23P")] = 1.764e+02# 1.764e+02 	#3.27e-1
        """
        A_coefficients[names.index("11S"), names.index("23P")] = 0# 1.764e+02 	#3.27e-1
        A_coefficients[names.index("23P"), names.index("31D")] = 0# 1.764e+02 	#3.27e-1
        A_coefficients[names.index("21P"), names.index("33D")] = 0# 1.764e+02 	#3.27e-1
        A_coefficients[names.index("21P"), names.index("23S")] = 0# 1.764e+02 	#3.27e-1
        A_coefficients[names.index("31D"), names.index("33P")] = 0# 1.764e+02 	#3.27e-1
        """
        """
        # F. Drake 1969
        drake_coeff = {"11S": {"21S": 5.13e1}}#, "23S": 4.02e-9}}
        for state_i, subtable in drake_coeff.items():
            if state_i not in states:
                continue
            for state_j, coeff in subtable.items():
                if state_j not in states:
                    continue
                A_coefficients[states.index(state_i),states.index(state_j)] = coeff
        """
        return A_coefficients

    # calculates the naural decay, arbsorbtion rate and stimulated emission rate
    def get_einstein_rates(self):
        A = self.get_A_rates() * u.s**-1
        E_diff = np.maximum(self.states.energies - self.states.energies[:,np.newaxis], 1e-5 * u.eV)
        nu = E_diff.to(u.Hz, equivalencies=u.spectral())
        const = consts.c**2 / (2 * consts.h * nu**3)
        g_ratio = self.states.multiplicities[:,np.newaxis] / self.states.multiplicities
        rho = u.sr * self.environment.spectrum(nu)
        return A.value, (A * const * rho).to("1/s").value, (A * const * rho * g_ratio).T.to("1/s").value
        

    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        coeff_mat[:len(self.states.names), :len(self.states.names)] = self.A + self.arbsorbtion_rate + self.stimulated_emission_rate
        return coeff_mat
    
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
        self.alpha = calculate_alpha_coefficients(environment.T_electrons)
        self.name = "Recombination"

    def get_transition_rate_matrix(self):
        coeff_mat = np.zeros((len(self.states.all_names),len(self.states.all_names)))
        names = self.states.all_names
        coeff_mat[names.index("23S"), names.index("HeII")] = self.alpha[0] * self.environment.n_e * 0.75
        coeff_mat[names.index("21S"), names.index("HeII")] = self.alpha[0] * self.environment.n_e * 0.25
        coeff_mat[names.index("HeII"), names.index("HeIII")] = self.alpha[1] * self.environment.n_e
        return coeff_mat
    
class HotElectronIonizationProcess:
    def __init__(self, states, environment):
        self.states = states
        self.environment = environment
        self.w = [600, 3000] # work per ionization in eV for HeII and HeIII respectively
        self.name = "Hot electron ionization"
        
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
        l = "SPDFABCD"[orbital_l]
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
    return np.array(ionization_rates)


# Calculate recombination coefficients
def calculate_alpha_coefficients(T):   
    T_list = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
    alpha_HeII = [1.99e-12, 1.71e-12, 1.47e-12, 1.27e-12, 1.09e-12, 9.32e-13, 7.98e-13, 6.84e-13, 5.85e-13, 5.00e-13, 4.28e-13]
    alpha_HeIII = [9.73e-12, 8.42e-12, 7.28e-12, 6.28e-12, 5.42e-12, 4.67e-12, 4.02e-12, 3.46e-12, 2.96e-12, 2.55e-12, 2.18e-12]

    return interp1d(T_list, [alpha_HeII, alpha_HeIII])( np.log10(T) )