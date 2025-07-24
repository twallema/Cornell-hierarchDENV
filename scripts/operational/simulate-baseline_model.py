"""
This script simulates a baseline model
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2025 by T.W. Alleman, Bento Lab, Cornell University CVM Public Health. All Rights Reserved."

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import nbinom
from scipy.optimize import minimize

# exclude years
exclude_years = ['2022-2023']

# desired quantiles
quantiles = [0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]

# helper functions
def neg_log_likelihood(params):
    r, p = params
    # Keep parameters in valid range
    if r <= 0 or p <= 0 or p >= 1:
        return np.inf
    return -np.sum(nbinom.logpmf(x, r, p))

# load data
data = pd.read_csv('../../data/raw/sprint_2025/dengue.csv', dtype={'epiweek': str})

# aggregate by UF
data = data[['epiweek', 'uf', 'casos']].groupby(by=['epiweek', 'uf']).sum().reset_index()

# split epiweek year and month
data['epiweek_year'] = data['epiweek'].apply(lambda x: x[:-2])
data['epiweek_week'] = data['epiweek'].apply(lambda x: x[-2:])
data = data[['epiweek_year', 'epiweek_week', 'uf', 'casos']]

# exclude years if needed
data = data[~data['epiweek_year'].isin(exclude_years)]

# compute unique epiweek/ufs
epiweek_weeks = data['epiweek_week'].unique().tolist()
ufs = data['uf'].unique().tolist()

# loop over them
results= []
for uf in ufs:
    for epiweek_week in epiweek_weeks:
        # get data
        x = data[((data['uf'] == uf) & (data['epiweek_week'] == epiweek_week))]['casos']
        # method of moments estimate for r and p
        mean_x = np.mean(x)
        var_x = np.var(x, ddof=1)
        p0 = mean_x / var_x
        r0 = mean_x * p0 / (1 - p0)
        p0 = max(min(p0, 0.99), 0.01)  # keep p0 in valid range
        r0 = max(r0, 0.1)
        # fit distribution
        res = minimize(neg_log_likelihood, x0=[r0, p0], bounds=[(1e-3, None), (1e-3, 1-1e-3)])
        r_hat, p_hat = res.x
        # simulate desired quantiles
        q_values = nbinom.ppf(quantiles, r_hat, p_hat)
        quantile_df = pd.DataFrame({'quantile': quantiles, 'casos': q_values})
        # attach the uf + epiweek
        quantile_df['uf'] = uf
        quantile_df['epiweek_week'] = epiweek_week
        # fill any nan with zero
        quantile_df = quantile_df.fillna(0)
        # append result
        results.append(quantile_df)
# concatenate all fitted distributions    
result = pd.concat(results, axis=0)
# order columns
result[['uf', 'epiweek_week', 'quantile' , 'casos']].to_csv('../../data/interim/baseline_model/simout.csv')

# plot result per uf
for uf in ufs:
    median = []
    ll_50 = []
    ul_50 = []
    ll_95 = []
    ul_95 = []
    fig,ax=plt.subplots(figsize=(11.7, 8.3/3))
    for epiweek_week in epiweek_weeks:
        # get data
        x = data[((data['uf'] == uf) & (data['epiweek_week'] == epiweek_week))]['casos']
        # plot data
        ax.scatter(int(epiweek_week) * np.ones(len(x)), x, color='black', alpha=0.2, facecolors='none')
        # get 50% and 95% quantiles
        median.append(int(result[((result['uf'] == uf) & (result['epiweek_week'] == epiweek_week) & (result['quantile'] == 0.50))]['casos'].values))
        ll_50.append(int(result[((result['uf'] == uf) & (result['epiweek_week'] == epiweek_week) & (result['quantile'] == 0.25))]['casos'].values))
        ul_50.append(int(result[((result['uf'] == uf) & (result['epiweek_week'] == epiweek_week) & (result['quantile'] == 0.75))]['casos'].values))
        ll_95.append(int(result[((result['uf'] == uf) & (result['epiweek_week'] == epiweek_week) & (result['quantile'] == 0.025))]['casos'].values))
        ul_95.append(int(result[((result['uf'] == uf) & (result['epiweek_week'] == epiweek_week) & (result['quantile'] == 0.975))]['casos'].values))
    # visualise quantiles
    ax.fill_between(epiweek_weeks, ll_50, ul_50, color='red', alpha=0.3)
    ax.fill_between(epiweek_weeks, ll_95, ul_95, color='red', alpha=0.1)
    # visualise median
    ax.plot(epiweek_weeks, median, color='red', linestyle='--')
    # make figure pretty
    ax.set_title(f'{uf}')
    ax.set_xlabel('CDC epiweek (-)')
    ax.set_ylabel('DENV incidence (-)')
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.savefig(f'../../data/interim/baseline_model/fig/{uf}.pdf')
    plt.close()

