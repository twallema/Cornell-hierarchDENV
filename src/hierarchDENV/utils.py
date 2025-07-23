import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from hierarchDENV.model import imsSIR

##########################
## Model initialisation ##
##########################

def initialise_model(strains=False, uf='MG'):
    """
    A function to intialise the hierarchDENV model

    input
    -----

    - strains: int
        - How many (independent) strains are modeled? No serotypes: 1. Serotypes: 4.

    uf: int
        - Abbreviation of Brazilian federative unit
    """

    # Parameters
    parameters = {
        # initial condition function
        'f_I': 1e-4 * np.ones(strains),
        'f_R': 0 * np.ones(strains),
        # SIR parameters
        'beta': 0.2 * np.ones(strains),
        'gamma': 1/5 * np.ones(strains),
        # transmission coefficient modifiers
        'delta_beta_temporal': np.array([1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5, 0.5, 1.5])-1,
        'modifier_length': 30,
        'sigma': 7,
        # reporting parameters
        'rho_report': 1,
        'T_report': 7,
        # immunity linking (unused)
        'season': '2024-2025'
        }
    
    # get inhabitants
    population = np.ones(strains) * get_demography(uf)

    # initialise initial condition function
    ICF = initial_condition_function(population).wo_immunity_linking

    return imsSIR(parameters, ICF, strains)


class initial_condition_function():

    def __init__(self, population):
        """
        Set up the model's initial condition function

        input
        -----

        - population: int
            - number of individuals in the modeled population.
        """
        self.population = population 
        pass

    def wo_immunity_linking(self, f_I, f_R, season):
        """
        A function generating the model's initial condition -- no immunity linking; direct estimation of recovered population at start of simulation
        
        input
        -----

        - population: int
            - Number of inhabitants in modeled Brazilian federal unit

        - f_I: float
            - Fraction of the population initially infected
        
        - f_R: float
            - Fraction of the population initially immune

        - season: str
            - Current season (unused)

        output
        ------

        - initial_condition: dict
            - Keys: 'S0', 'I0', 'R0'. Values: int.
        """

        # construct initial condition
        return {'S0':  (1 - f_I - f_R) * self.population,
                'I0': f_I * self.population,   
                'R0': f_R * self.population,
                }

def get_demography(uf: str) -> int:
    """
    A function retrieving the total population of a Brazilian federative unit (uf)

    input
    -----

    - uf: str
        - Abbreviation of Brazilian federative unit

    output
    ------

    - population: int
        - population size
    """ 

    # load demography
    demography = pd.read_csv(os.path.join(os.path.dirname(__file__), f'../../data/interim/state_covariates.csv'), index_col=0)['population']
    
    # check validity of UF abbrevation
    assert uf in demography.index, 'invalid Brazilian federative unit.'

    return int(demography.loc[uf])


################################
## Data and output formatting ##
################################

def get_BR_DENV_data(uf: str,
                     startdate: datetime,
                     enddate: datetime) -> pd.DataFrame:
    """
    Get the DENV incidence per serotype in a Brazilian federative unit between `startdate` and `enddate`

    input
    -----

    - uf: str
        - Brazilian federative unit abbreviation 

    - startdate: str/datetime
        - start of dataset
    
    - enddate: str/datetime
        - end of dataset

    output
    ------

    - data: pd.DataFrame
        - index: 'date' [datetime], columns: 'DENV_1', 'DENV_2', 'DENV_3', 'DENV_4' (frequency: monthly, converted to daily)
    """

    # load total DENV data and serotype distribution
    df = pd.read_csv(os.path.join(os.path.dirname(__file__),f'../../data/interim/imputed_DENV_datasus/DENV-serotypes-imputed_1996-2025_monthly.csv'), index_col=[0,1])
    
    # format dates
    df.index = df.index.set_levels(
                    pd.to_datetime(df.index.levels[0]), level=0
    )

    # impute total DENV data with serotype distribution
    df['DENV_1'] = df['DENV_total'] * df['p_1']
    df['DENV_2'] = df['DENV_total'] * df['p_2']
    df['DENV_3'] = df['DENV_total'] * df['p_3']
    df['DENV_4'] = df['DENV_total'] * df['p_4']
    df = df[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']]/30 # normalise month --> day

    # select right daterange and state
    df = df.loc[(slice(startdate, enddate), uf), :].droplevel('UF')

    return df

def get_cdc_week_saturday(year, week):
    # CDC epiweeks start on Sunday and end on Saturday
    # CDC week 1 is the week with at least 4 days in January
    # Start from Jan 4th and find the Sunday of that week
    jan4 = datetime(year, 1, 4)
    start_of_week1 = jan4 - timedelta(days=jan4.weekday() + 1)  # Move to previous Sunday

    # Add (week - 1) weeks and 6 days to get Saturday
    saturday_of_week = start_of_week1 + timedelta(weeks=week-1, days=6)
    return saturday_of_week

from pySODM.optimization.objective_functions import ll_poisson
def make_data_pySODM_compatible(uf: str,
                                serotypes: int,
                                start_date: datetime,
                                end_date: datetime): 
    """
    A function formatting the Brazilian DENV data in pySODM format

    
    input:
    ------

    - uf: str
        - Brazilian federative unit abbreviation 

    - serotypes: int
        - how many serotypes are modeled? 1: flu, 2: flu A, flu B, 3: flu A H1, flu A H3, flu B.

    - start_date: datetime
        - desired startdate of data

    - end_date: datetime
        - desired enddate of data
    
    output:
    -------

    - data: list containing pd.DataFrame
        - contains datasets the model should be calibrated to.
    
    - states: list containing str
        - contains names of model states that should be matched to the datasets in `data`.
        - length: `len(data)`
    
    - log_likelihood_fnc: list containing log likelihood function
        - pySODM.optimization.objective_functions.ll_poisson
        - length: `len(data)`
    
    - log_likelihood_fnc_args: list containing empty lists
        - length: `len(data)`
    """

    if serotypes == False:
        # pySODM llp data arguments
        states = ['I_inc',]
        log_likelihood_fnc = len(states) * [ll_poisson,]
        log_likelihood_fnc_args = len(states) * [[],]
        # pySODM data
        df = get_BR_DENV_data(uf, start_date, end_date).sum(axis=1)
        df = df.rename('I_inc')
        data = [df, ]

    elif serotypes == True:
        # pySODM llp data arguments
        states = ['I_inc', 'I_inc', 'I_inc', 'I_inc'] + ['I_inc',]
        log_likelihood_fnc = len(states) * [ll_poisson,]
        log_likelihood_fnc_args = len(states) * [[],]
        # pySODM data
        df = get_BR_DENV_data(uf, start_date, end_date)
        ## serotyped DENV incidence
        data = []
        for i in range(4):
            d = df[f'DENV_{i+1}']
            d = d.rename('I_inc')
            d = d.reset_index()
            d['strain'] = i
            data.append(d.set_index(['date', 'strain']).squeeze())
        ## total DENV incidence
        data.append(get_BR_DENV_data(uf, start_date, end_date).sum(axis=1))

    return data, states, log_likelihood_fnc, log_likelihood_fnc_args


def samples_to_csv(ds: xr.Dataset) -> pd.DataFrame:
    """
    A function used convert the median value of parameter across MCMC chains and iterations into a flattened csv format

    Parameters
    ----------
    - ds: xarray.Dataset
        - Average or median parameter samples
        - Typically obtained after MCMC sampling in `incremental_forecasting.py` as: `ds = samples_xr.median(dim=['chain', 'iteration'])`

    Returns
    -------

    - df: pd.DataFrame
        - Index: Expanded parameter name
        - Values: Average or median parameter values
        - 1D parameter names are expanded: 'rho_h' --> 'rho_h_0' , 'rho_h_1', ...
    """
    param_dict = {}

    for var_name, da in ds.data_vars.items():
        if da.ndim == 0:
            # Scalar variable
            param_dict[var_name] = da.item()
        elif da.ndim == 1:
            # 1D variable
            for i, val in enumerate(da.values):
                param_dict[f"{var_name}_{i}"] = val
        else:
            raise ValueError(f"Variable '{var_name}' has more than 1 dimension ({da.dims}); this script handles only scalars and 1D variables.")

    df = pd.DataFrame.from_dict(param_dict, orient='index', columns=['value'])
    df.index.name = 'parameter'
    return df


from pySODM.optimization.objective_functions import log_prior_normal, log_prior_lognormal, log_prior_uniform, log_prior_gamma, log_prior_normal, log_prior_beta
def get_priors(strains, hyperparameters):
    """
    A function to help prepare the pySODM-compatible priors
    """

    # Derive model name from number of strains
    model_name = f'SIR-{strains}S'
    # Define parameters, bounds and labels
    pars = ['rho_report', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']                              # parameters to calibrate
    bounds = [(0,1), (0,1), (1e-9,1e-2), (0,1), (-0.50,0.50)]                                       # parameter bounds
    labels = [r'$\rho_{report}$',  r'$f_{R}$', r'$f_{I}$', r'$\beta$', r'$\Delta \beta_{t}$']       # labels in output figures
    
    # UNINFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if not hyperparameters:
        # assign priors (R0 ~ N(2.5, 0.5); modifiers centered around zero; f_R ~ N(0.5, 0.1); reporting parameters nudged to lowest value)
        log_prior_prob_fcn = 2*[log_prior_beta,] + [log_prior_gamma,] + 2*[log_prior_normal,]
        log_prior_prob_fcn_args = [{'a': 4, 'b': 1, 'loc': 0, 'scale': 1},
                                    {'a': 1, 'b': 4, 'loc': 0, 'scale': 1},
                                    {'a': 1, 'loc': 0, 'scale': 0.1*max(bounds[2])},
                                    {'avg':  0.5, 'stdev': 0.1},
                                    {'avg':  0, 'stdev': 0.10}]
    # INFORMED: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    else:
        # load and select priors
        priors = pd.read_csv('../../data/interim/calibration/hyperparameters.csv')
        priors = priors.loc[(priors['model'] == model_name), (['parameter', f'{hyperparameters}'])].set_index('parameter').squeeze()
        # assign values
        if strains == 1:
            log_prior_prob_fcn = [log_prior_lognormal,] + 1*[log_prior_normal,] + 1*[log_prior_lognormal,] + 13*[log_prior_normal,]
            log_prior_prob_fcn_args = [ 
                                    # reporting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'s': priors['rho_report_s'], 'scale': priors['rho_report_scale']},                             # rho_report
                                    # initial condition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'avg': priors['f_R_mu'], 'stdev': priors['f_R_sigma']},                                        # f_R
                                    {'s': priors['f_I_s'], 'scale': priors['f_I_scale']},                                           # f_I
                                    # transmission coefficient >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'avg': priors['beta_mu'], 'stdev': priors['beta_sigma']},                                      # beta
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]          
        elif strains == 4:
            log_prior_prob_fcn = [log_prior_lognormal,] + 4*[log_prior_normal,] + 4*[log_prior_lognormal,] + (4+12)*[log_prior_normal,]
            log_prior_prob_fcn_args = [ 
                                    # reporting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'s': priors['rho_report_s'], 'scale': priors['rho_report_scale']},                             # rho_report
                                    # initial condition >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'avg': priors['f_R_mu_0'], 'stdev': priors['f_R_sigma_0']},                                    # f_R_0
                                    {'avg': priors['f_R_mu_1'], 'stdev': priors['f_R_sigma_1']},                                    # f_R_1
                                    {'avg': priors['f_R_mu_2'], 'stdev': priors['f_R_sigma_2']},                                    # f_R_2
                                    {'avg': priors['f_R_mu_3'], 'stdev': priors['f_R_sigma_3']},                                    # f_R_3
                                    {'s': priors['f_I_s_0'], 'scale': priors['f_I_scale_0']},                                       # f_I_0
                                    {'s': priors['f_I_s_1'], 'scale': priors['f_I_scale_1']},                                       # f_I_1
                                    {'s': priors['f_I_s_2'], 'scale': priors['f_I_scale_2']},                                       # f_I_2
                                    {'s': priors['f_I_s_3'], 'scale': priors['f_I_scale_3']},                                       # f_I_3
                                    # transmission coefficient >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                                    {'avg': priors['beta_mu_0'], 'stdev': priors['beta_sigma_0']},                                  # beta_0
                                    {'avg': priors['beta_mu_1'], 'stdev': priors['beta_sigma_1']},                                  # beta_1
                                    {'avg': priors['beta_mu_2'], 'stdev': priors['beta_sigma_2']},                                  # beta_2
                                    {'avg': priors['beta_mu_3'], 'stdev': priors['beta_sigma_3']},                                  # beta_3
                                    {'avg': priors['delta_beta_temporal_mu_0'], 'stdev': priors['delta_beta_temporal_sigma_0']},    # delta_beta_temporal
                                    {'avg': priors['delta_beta_temporal_mu_1'], 'stdev': priors['delta_beta_temporal_sigma_1']},    # ...
                                    {'avg': priors['delta_beta_temporal_mu_2'], 'stdev': priors['delta_beta_temporal_sigma_2']},
                                    {'avg': priors['delta_beta_temporal_mu_3'], 'stdev': priors['delta_beta_temporal_sigma_3']},
                                    {'avg': priors['delta_beta_temporal_mu_4'], 'stdev': priors['delta_beta_temporal_sigma_4']},
                                    {'avg': priors['delta_beta_temporal_mu_5'], 'stdev': priors['delta_beta_temporal_sigma_5']},
                                    {'avg': priors['delta_beta_temporal_mu_6'], 'stdev': priors['delta_beta_temporal_sigma_6']},
                                    {'avg': priors['delta_beta_temporal_mu_7'], 'stdev': priors['delta_beta_temporal_sigma_7']},
                                    {'avg': priors['delta_beta_temporal_mu_8'], 'stdev': priors['delta_beta_temporal_sigma_8']},
                                    {'avg': priors['delta_beta_temporal_mu_9'], 'stdev': priors['delta_beta_temporal_sigma_9']},
                                    {'avg': priors['delta_beta_temporal_mu_10'], 'stdev': priors['delta_beta_temporal_sigma_10']},
                                    {'avg': priors['delta_beta_temporal_mu_11'], 'stdev': priors['delta_beta_temporal_sigma_11']},
                                    ]     
    return pars, bounds, labels, log_prior_prob_fcn, log_prior_prob_fcn_args


#########################################################
## Transmission rate: equivalent Python implementation ##
#########################################################

from scipy.ndimage import gaussian_filter1d
def get_transmission_coefficient_timeseries(modifier_vector: np.ndarray,
                                            sigma: float=7) -> np.ndarray:
    """
    A function mapping the delta_beta_temporal vector between Oct 1 and Sept 30 and smoothing it with a gaussian filter

    input
    -----

    - modifier_vector: np.ndarray
        - 1D numpy array (time) or 2D numpy array (time x spatial unit).
        - Each entry represents a value of a knotted temporal modifier, the length of each modifier is equal to the time between Oct 15 and Apr 15 (182 days) divided by `len(modifier_vector)`.

    - sigma: float 
        - gaussian smoother's standard deviation. higher values represent more smooth trajectories but increase runtime. `None` represents no smoothing (fastest).

    output
    ------

    - smooth_temporal_modifier: np.ndarray
        - 1D array of smoothed modifiers.
    """

    # Ensure the input is at least 2D
    if modifier_vector.ndim == 1:
        modifier_vector = modifier_vector[:, np.newaxis]
    _, num_space = modifier_vector.shape

    # Define number of days between Oct 15 and Apr 15
    num_days = 182

    # Step 1: Project the input vector onto the daily time scale
    interval_size = num_days / len(modifier_vector)
    positions = (np.arange(num_days) // interval_size).astype(int)
    expanded_vector = modifier_vector[positions, :]

    # Step 2: Prepend and append 31 days of ones
    padding = np.zeros((31, num_space))
    padded_vector = np.vstack([padding, expanded_vector, padding])

    # Step 3: apply the Gaussian filter
    return np.squeeze(gaussian_filter1d(padded_vector, sigma=sigma, axis=0, mode="nearest"))


##############################
## Plot fit helper function ##
##############################


def plot_fit(simout: xr.Dataset,
             data_calibration: list,
             data_validation: list,
             states: list,
             fig_path: str,
             identifier: str,
             coordinates_data_also_in_model: list,
             aggregate_over: list,
             additional_axes_data: list,
             rescaling: int) -> None:
    """
    A function used to visualise the goodness of fit 

    #TODO: LIMITED TO ONE COORDINATE PER DIMENSION PER DATASET !!!

    input
    -----

    - simout: xr.Dataset
        - simulation output (pySODM-compatible) . must contain all states listed in `states`.
    
    - data_calibration: list containing pySODM-compatible pd.DataFrame
        - data model was calibrated to.
        - obtained using hierarchSIR.utils.make_data_pySODM_compatible.
        - length: `len(states)`
    
    - data_validation: list containing pySODM-compatible pd.DataFrame
        - data model was not calibrated to (validation).
        - obtained using hierarchSIR.utils.make_data_pySODM_compatible.
        - length: `len(states)`

    - states: list containing str
        - names of model states that were matched with data in `data_calibration`.
    
    - fig_path: str
        - path where figure should be stored, relative to path of file this function is called from.
    
    - identifier: str
        - an ID used to name the output figure.
    
    - coordinates_data_also_in_model: list
        - contains a list for every dataset. contains a list for every model dimension besides 'date'/'time', containing the coordinates present in the data and also in the model.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - aggregate_over: list
        - list of length len(data). contains, per dataset, the remaining model dimensions not present in the dataset. these are then automatically summed over while calculating the log likelihood.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - additional_axes_data: list
        - axes in dataset, excluding the 'time'/'date' axes.
        - obtained from pySODM.optimization.log_posterior_probability
    
    - rescaling: int
        - scales the simulation output and data by `rescaling`
    """

    # check if 'draws' are provided
    samples = False
    if 'draws' in simout.dims:
        samples = True
    
    # compute the amount of timeseries to visualise
    nrows = sum(1 if not coords else len(coords) for coords in coordinates_data_also_in_model)

    # generate figure
    _,ax=plt.subplots(nrows=nrows, sharex=True, figsize=(8.3, 11.7/5*nrows))

    # vectorise ax object
    if nrows==1:
        ax = [ax,]

    # save a copy to reset
    out_copy = simout

    # loop over datasets
    k=0
    for i, (df_calib, df_valid) in enumerate(zip(data_calibration, data_validation)):
        
        # aggregate data
        for dimension in simout.dims:
            if dimension in aggregate_over[i]:
                simout = simout.sum(dim=dimension)
        
        # loop over coordinates 
        if coordinates_data_also_in_model[i]:
            for coord in coordinates_data_also_in_model[i]:
                # get dimension coord is in 
                dim_name = additional_axes_data[i][0]
                coord = coord[0]
                # plot
                ax[k].scatter(df_calib.index.get_level_values('date').values, rescaling*df_calib.loc[slice(None), coord].values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                if not df_valid.empty:
                    ax[k].scatter(df_valid.index.get_level_values('date').values, rescaling*df_valid.loc[slice(None), coord].values, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
                
                if samples:
                    ax[k].fill_between(simout['date'], rescaling*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.05/2),
                            rescaling*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                    ax[k].fill_between(simout['date'], rescaling*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=0.50/2),
                            rescaling*simout[states[i]].sel({dim_name: coord}).quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
                else:
                    ax[k].plot(simout['date'], rescaling*simout[states[i]].sel({dim_name: coord}), color='blue')
                ax[k].set_title(f'State: {states[i]}; Dim: {dim_name} ({coord})')
                k += 1
        else:
            # plot
            ax[k].scatter(df_calib.index, rescaling*df_calib.values, color='black', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
            if not df_valid.empty:
                ax[k].scatter(df_valid.index, rescaling*df_valid.values, color='red', alpha=1, linestyle='None', facecolors='None', s=60, linewidth=2)
            if samples:
                ax[k].fill_between(simout['date'], rescaling*simout[states[i]].quantile(dim='draws', q=0.05/2),
                            rescaling*simout[states[i]].quantile(dim='draws', q=1-0.05/2), color='blue', alpha=0.15)
                ax[k].fill_between(simout['date'], rescaling*simout[states[i]].quantile(dim='draws', q=0.50/2),
                            rescaling*simout[states[i]].quantile(dim='draws', q=1-0.50/2), color='blue', alpha=0.20)
            else:
                ax[k].plot(simout['date'], rescaling*simout[states[i]], color='blue')
            ax[k].set_title(f'State: {states[i]}')
            k += 1
        
        # reset output
        simout = out_copy

    plt.tight_layout()
    plt.savefig(fig_path+f'{identifier}-FIT.pdf')
    plt.close()

# helper function
def str_to_bool(value):
    """Convert string arguments to boolean (for SLURM environment variables)."""
    return value.lower() in ["true", "1", "yes"]