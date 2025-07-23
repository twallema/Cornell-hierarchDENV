"""
This script does..
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, Bento Lab, Cornell University CVM Public Health. All Rights Reserved."

import sys,os
import emcee
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from multiprocessing import get_context
from hierarchDENV.training import log_posterior_probability, dump_sampler_to_xarray, traceplot, plot_fit, hyperdistributions
from hierarchDENV.utils import initialise_model, make_data_pySODM_compatible, str_to_bool


##############
## Settings ##
##############

# calibration settings
## datasets
identifier = 'validation_1'                                                                                     # identifiers of the training
seasons = ['2021-2022', '2020-2021', '2019-2020', '2018-2019', '2017-2018', '2016-2017', '2015-2016']           # season to include in training                                                                                                       
# season length
season_start_month = 9
season_end_month = 9
run_date = datetime.today().strftime("%Y-%m-%d")
## define number of chains
max_n = 200
pert = 0.05
processes = int(os.environ.get('NUM_CORES', mp.cpu_count()))
## printing and postprocessing
print_n = 100
backend = None
discard = 50
thin = 1


#####################
## Parse arguments ##
#####################

# arguments determine the model + data combo used to forecast
parser = argparse.ArgumentParser()
parser.add_argument("--serotypes", type=str_to_bool, help="Include serotypes. False: 1 strain model, True: 4 strain model.")
parser.add_argument("--ufs", type=lambda s: s.split(','), help="List of Brazilian federative unit abbreviations.")
args = parser.parse_args()

# assign to desired variables
serotypes = args.serotypes
ufs = args.ufs

# format number of strains and model name
strains = 4 if serotypes is True else 1
model_name = f'SIR-{strains}S'


##################
## Optimisation ##
##################

# Needed for multiprocessing to work properly
if __name__ == '__main__':

    # Loop over states
    for uf in ufs:
        print(f"\nWorking on calibration with ID: {identifier}")
        sys.stdout.flush()
        
        #########################
        ## Make results folder ##
        #########################

        ## format model name
        model_name = f'SIR-{strains}S'
        ## define samples path
        samples_path=fig_path=f'../../data/interim/calibration/hierarchical-training/{model_name}/{uf}/{identifier}/' # Path to backend
        ## check if samples folder exists, if not, make it
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        ################
        ## Setup data ##
        ################

        # convert to a list of start and enddates (datetime)
        start_calibrations = [datetime(int(season[0:4]), season_start_month, 1) for season in seasons]
        end_calibrations = [datetime(int(season[0:4])+1, season_end_month, 1) for season in seasons]
        start_simulations = start_calibrations

        # get data
        datasets = []
        for start_calibration, end_calibration, season in zip(start_calibrations, end_calibrations, seasons):
            data, _, _, _ = make_data_pySODM_compatible(uf, serotypes, start_calibration, end_calibration)
            datasets.append(data)

        #################
        ## Setup model ##
        #################

        model = initialise_model(strains=strains, uf=uf)

        ##########################################
        ## Setup posterior probability function ##
        ##########################################

        # define model parameters to calibrate to every season and their bounds
        par_names = ['rho_report', 'f_R', 'f_I', 'beta', 'delta_beta_temporal']                             # parameters to calibrate
        par_bounds = [(0,1), (0,1), (1e-9,1e-2), (0,1), (-0.50,0.50)]                                       # parameter bounds
        par_hyperdistributions = ['beta', 'beta', 'lognorm', 'norm', 'norm']                            
        # setup lpp function
        lpp = log_posterior_probability(model, par_names, par_bounds, par_hyperdistributions, datasets, seasons, start_simulations)

        ####################################
        ## Fetch initial guess parameters ##
        ####################################

        # parameters: get optimal independent fit with weakly informative prior on R0 and immunity
        pars_model_0 = pd.read_csv('../../data/interim/calibration/initial_guesses.csv', index_col=[0,1,2,3])
        pars_0 = list(pars_model_0.loc[(model_name, uf, slice(None), slice(None)), seasons].transpose().values.flatten().tolist())

        # hyperparameters: use all seasons included as the default starting point
        hyperpars_0 = pd.read_csv('../../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2])
        hyperpars_0 = hyperpars_0.loc[(model_name, uf, slice(None)), 'initial_guess'].values.tolist()

        # combine
        theta_0 = hyperpars_0 + pars_0

        # run with chain multiplier of two (minimal configuration)
        n_chains = 2*len(theta_0)

        ###################
        ## Setup sampler ##
        ###################

        # Generate random perturbations from a normal distribution
        perturbations = np.random.normal(
                loc=1, scale=pert, size=(n_chains, len(theta_0))
            )

        # Apply perturbations to create the 2D array
        pos = np.array(theta_0)[None, :] * perturbations
        nwalkers, ndim = pos.shape

        # By default: set up a fresh hdf5 backend in samples_path
        if not backend:
            fn_backend = str(identifier)+'_BACKEND_'+run_date+'.hdf5'
            backend_file = emcee.backends.HDFBackend(samples_path+fn_backend)
        # If user provides an existing backend: continue sampling 
        else:
            try:
                backend_file = emcee.backends.HDFBackend(samples_path+backend)
                pos = backend_file.get_chain(discard=discard, thin=thin, flat=False)[-1, ...]
            except:
                raise FileNotFoundError("backend not found.")    

        # setup and run sampler
        with get_context("spawn").Pool(processes=processes) as pool:
            # setup sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lpp, backend=backend_file, pool=pool,
                                            moves=[(emcee.moves.DEMove(), 0.5*0.9),(emcee.moves.DEMove(gamma0=1.0), 0.5*0.1),
                                                    (emcee.moves.StretchMove(live_dangerously=True), 0.50)]
                                            )
            # sample
            for sample in sampler.sample(pos, iterations=max_n, progress=True, store=True, skip_initial_state_check=True):

                if sampler.iteration % print_n:
                    continue
                else:

                    # every print_n steps do..
                    # >>>>>>>>>>>>>>>>>>>>>>>>>

                    # ..dump samples without discarding and generate traceplots
                    samples = dump_sampler_to_xarray(sampler.get_chain(discard=0, thin=thin), samples_path+str(identifier)+'_SAMPLES_'+run_date+'.nc', lpp.hyperpar_shapes, lpp.par_shapes, seasons)
                    traceplot(samples, lpp.par_shapes, lpp.hyperpar_shapes, samples_path, identifier, run_date)

                    # ..dump samples with discarding and generate other results
                    samples = dump_sampler_to_xarray(sampler.get_chain(discard=discard, thin=thin), samples_path+str(identifier)+'_SAMPLES_'+run_date+'.nc', lpp.hyperpar_shapes, lpp.par_shapes, seasons)
                    # write median hyperpars to .csv
                    hyperpars_names = []
                    hyperpars_values = []
                    for hyperpar_name, hyperpar_shape in lpp.hyperpar_shapes.items():
                        # append value
                        hyperpars_values.append(samples.median(dim=['chain', 'iteration'])[hyperpar_name].values.tolist())
                        # append name
                        hyperpars_names.extend([f'{hyperpar_name}_{i}' if hyperpar_shape[0] > 1 else f'{hyperpar_name}' for i in range(hyperpar_shape[0])])
                    hyperpars_values = np.hstack(hyperpars_values)
                    # save to .csv
                    hyperpars_df = pd.Series(index=hyperpars_names, data=hyperpars_values, name=identifier)
                    hyperpars_df.to_csv(samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.csv')
                    # add to hyperparameters file
                    hyperpars_0 = pd.read_csv('../../data/interim/calibration/hyperparameters.csv', index_col=[0,1,2])
                    hyperpars_0.loc[(model_name, uf, slice(None)), identifier] = hyperpars_df.values
                    hyperpars_0.to_csv('../../data/interim/calibration/hyperparameters.csv')
                    # .. visualise hyperdistributions
                    hyperdistributions(samples, samples_path+str(identifier)+'_HYPERDIST_'+run_date+'.pdf', lpp.par_shapes, lpp.hyperpar_shapes, par_hyperdistributions, par_bounds, 100)
                    # ..generate traceplots
                    traceplot(samples, lpp.par_shapes, lpp.hyperpar_shapes, samples_path, identifier, run_date)
                    # ..generate goodness-of-fit
                    plot_fit(model, datasets, lpp.simtimes, samples, model.parameter_shapes, samples_path, identifier, run_date,
                                lpp.coordinates_data_also_in_model, lpp.aggregate_over, lpp.additional_axes_data, lpp.corresponding_model_states)