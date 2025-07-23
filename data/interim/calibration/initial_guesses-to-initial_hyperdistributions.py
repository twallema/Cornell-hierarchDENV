import numpy as np
import pandas as pd
from scipy.stats import beta, gamma, lognorm

# column name in resulting hyperdistribution file
column_name = 'initial_guess'

# open the initial guesses file
intial_guesses = pd.read_csv('initial_guesses.csv', index_col=[0,1,2,3])

# determine the model names and ufs
model_names = intial_guesses.index.get_level_values('model').unique().to_list()
ufs = intial_guesses.index.get_level_values('uf').unique().to_list()

# estimate the hyperdistribution parameters based on the data
data_collection = []
for model in model_names:
    for uf in ufs:
        # slice intial guesses
        df = intial_guesses.loc[(model, uf, slice(None), slice(None)), :].droplevel(['model', 'uf'])
        # retrieve parameter names and their hyperdistributions
        parameter_names = df.index.get_level_values('parameter').to_list()
        parameter_hyperdistributions = df.index.get_level_values('hyperdistribution')
        # construct the names of the hyperparameters, and estimate their magnitude using scipy.dist.fit
        names = []
        values = []
        for parname, hyperdist in zip(parameter_names, parameter_hyperdistributions):
            # get the data
            data = df.loc[parname].values
            # find right hyperdistribution
            if hyperdist == 'gamma':
                # names
                names.append(f'{parname}_a')
                names.append(f'{parname}_scale')
                # values
                a, _, scale = gamma.fit(data, floc=0)
                values.append(a)
                values.append(scale)
            elif hyperdist == 'expon':
                # name
                names.append(f'{parname}_scale')
                # value
                a, _, _ = gamma.fit(data, floc=0, fscale=1)
                values.append(a)
            elif hyperdist == 'norm':
                # names
                names.append(f'{parname}_mu')
                names.append(f'{parname}_sigma')
                # values
                values.append(np.mean(data))
                values.append(np.std(data))
            elif hyperdist == 'beta':
                # names
                names.append(f'{parname}_a')
                names.append(f'{parname}_b')
                # fit values
                a, b, _, _ = beta.fit(data, floc=0, fscale=1)
                values.append(a)
                values.append(b)
            elif hyperdist == 'lognorm':
                # names
                names.append(f'{parname}_s')
                names.append(f'{parname}_scale')
                # values
                s, _, scale = lognorm.fit(data, floc=0)
                values.append(s)
                values.append(scale)
            else:
                raise ValueError(f"'{hyperdist}' is not a valid hyperdistribution.")
        pass

        # send to a pandas dataframe
        df = pd.DataFrame({'hyperparameter': names, column_name: values})
        df.set_index('hyperparameter', inplace=True)

        # SUPER DORKY: REARRANGE
        if model == 'SIR-1S':
            df = df.loc[
                [
                    'rho_report_a', 'rho_report_b', 'f_R_a', 'f_R_b', 'f_I_s', 'f_I_scale', 'beta_mu', 'beta_sigma',
                    'delta_beta_temporal_0_mu', 'delta_beta_temporal_1_mu', 'delta_beta_temporal_2_mu', 'delta_beta_temporal_3_mu', 'delta_beta_temporal_4_mu', 'delta_beta_temporal_5_mu', 'delta_beta_temporal_6_mu', 'delta_beta_temporal_7_mu', 'delta_beta_temporal_8_mu', 'delta_beta_temporal_9_mu', 'delta_beta_temporal_10_mu', 
                    'delta_beta_temporal_0_sigma', 'delta_beta_temporal_1_sigma', 'delta_beta_temporal_2_sigma', 'delta_beta_temporal_3_sigma', 'delta_beta_temporal_4_sigma', 'delta_beta_temporal_5_sigma', 'delta_beta_temporal_6_sigma', 'delta_beta_temporal_7_sigma', 'delta_beta_temporal_8_sigma', 'delta_beta_temporal_9_sigma', 'delta_beta_temporal_10_sigma', 
                 ]
            ]
        elif model == 'SIR-4S':
             df = df.loc[
                [
                    'rho_report_a', 'rho_report_b',
                    'f_R_0_a', 'f_R_1_a', 'f_R_2_a', 'f_R_3_a', 'f_R_0_b', 'f_R_1_b', 'f_R_2_b', 'f_R_3_b', 
                    'f_I_0_s', 'f_I_1_s', 'f_I_2_s', 'f_I_3_s', 'f_I_0_scale', 'f_I_1_scale', 'f_I_2_scale', 'f_I_3_scale',
                    'beta_0_mu', 'beta_1_mu', 'beta_2_mu', 'beta_3_mu', 'beta_0_sigma', 'beta_1_sigma', 'beta_2_sigma', 'beta_3_sigma',
                    'delta_beta_temporal_0_mu', 'delta_beta_temporal_1_mu', 'delta_beta_temporal_2_mu', 'delta_beta_temporal_3_mu', 'delta_beta_temporal_4_mu', 'delta_beta_temporal_5_mu', 'delta_beta_temporal_6_mu', 'delta_beta_temporal_7_mu', 'delta_beta_temporal_8_mu', 'delta_beta_temporal_9_mu', 'delta_beta_temporal_10_mu', 
                    'delta_beta_temporal_0_sigma', 'delta_beta_temporal_1_sigma', 'delta_beta_temporal_2_sigma', 'delta_beta_temporal_3_sigma', 'delta_beta_temporal_4_sigma', 'delta_beta_temporal_5_sigma', 'delta_beta_temporal_6_sigma', 'delta_beta_temporal_7_sigma', 'delta_beta_temporal_8_sigma', 'delta_beta_temporal_9_sigma', 'delta_beta_temporal_10_sigma', 
                 ]
            ]

        # add column for model name and uf
        df = df.reset_index()
        df['model'] = model
        df['uf'] = uf
        df = df.set_index(['model', 'uf', 'hyperparameter'])

        # collect
        data_collection.append(df)

# concatenate over models and ufs
result = pd.concat(data_collection)

# make a new excel file with an initial guess of the hyperdistributions
result.to_csv('hyperparameters.csv')

