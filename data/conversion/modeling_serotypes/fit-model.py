import arviz
import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import pytensor
import pytensor.tensor as pt
pytensor.config.cxx = '/usr/bin/clang++'
pytensor.config.on_opt_error = "ignore"


########################
## Preparing the data ##
########################


# Distance matrix
# ~~~~~~~~~~~~~~~

# Load distance matrix
D = pd.read_csv('../../interim/weighted_distance_matrix.csv', index_col=0).values
D_matrix = D
# weigh with negative exponential decay model to avoid overparametrisation (internalise zeta later)
zeta = 1000 # km
W = np.exp(-D / zeta)
np.fill_diagonal(W, 0)
# Degree matrix D: diagonal of row sums of W
D = np.diag(W.sum(axis=1))
# Fixed alpha
alpha_fixed = 0.5
# Build Q
Q = D - alpha_fixed * W
Q += 1e-6 * np.eye(Q.shape[0])  # Numerical stabilization

# Covariates
# ~~~~~~~~~~

# Fetch covariates
X = pd.read_csv('../../state_covariates.csv', index_col=0)
# Normalise all covariates
# Standardization ensures a change of 1 unit in X_j corresponds to a 1 SD change — makes priors and posteriors comparable across covariates.
# X: shape (n_states, n_covariates)
# Rows: states, Columns: covariates
X_mean = np.mean(X, axis=0)         # mean per covariate
X_std = np.std(X, axis=0, ddof=0)   # std dev per covariate
# Avoid divide-by-zero
X_std[X_std == 0] = 1.0
# Standardize
X = ((X - X_mean) / X_std).values


# Region mapping
# ~~~~~~~~~~~~~~

uf2region_map = pd.read_csv('uf2region.csv')[['uf', 'region']].drop_duplicates().set_index('uf')['region'].to_dict()


# Incidence data
# ~~~~~~~~~~~~~~

# Fetch incidence data
df = pd.read_csv('../../interim/DENV_datasus/DENV-serotypes_1996-2025_monthly.csv', parse_dates=['date'])

# 1. Check if all columns are present
sero_cols = ["DENV_1", "DENV_2", "DENV_3", "DENV_4"]
required_cols = ["UF", "date", "DENV_total"] + sero_cols
assert all(col in df.columns for col in required_cols)

# 2. Sort for safety
df = df.sort_values(["date", "UF"]).reset_index(drop=True)

# 3. Factorize states and time
df["state_idx"], _ = pd.factorize(df["UF"])
df["month_idx"], _ = pd.factorize(df["date"])

# 4. Fill NaNs in a principled way
def fill_serotypes(row):
    sero = row[sero_cols]
    if sero.notna().any():
        # If at least one serotype is observed, treat missing ones as 0
        for col in sero_cols:
            if pd.isna(row[col]):
                row[col] = 0.0
    return row
df = df.apply(fill_serotypes, axis=1)

# 5. Compute N_typed
df["N_typed"] = df[sero_cols].sum(axis=1, skipna=False)                                     # if serotypes available --> sum them
df.loc[df[['DENV_1', 'DENV_2', 'DENV_3', 'DENV_4']].isna().all(axis=1), 'N_typed'] = np.nan      # if all serotypes are Nan --> N_typed = 0 --> Wait, I don't think this is appropriate.

# 6. Compute delta (typing fraction)
df["delta"] = df["N_typed"] / df["DENV_total"]
df['delta'] = df['delta'].where(df['N_typed'] > 0, np.nan) # When N_typed == 0, we don't know delta — mark as missing
df["delta"] = df["delta"].clip(lower=1e-12, upper=1 - 1e-12)

# 7. Compute year index
df["year"] = pd.to_datetime(df["date"]).dt.year
df["year_idx"] = df["year"] - df["year"].min()

# 8. Compute year-state index
df["state_year_idx"] = df["state_idx"].astype(str) + "_" + df["year_idx"].astype(str)
df["state_year_idx"], state_year_labels = pd.factorize(df["state_year_idx"])

# 9. Add year-region index
df['region'] = df['UF'].map(uf2region_map)
df["region_idx"], region_labels = pd.factorize(df["region"])
df["region_year_idx"] = df["region_idx"].astype(str) + "_" + df["year_idx"].astype(str)
df["region_year_idx"], region_year_labels = pd.factorize(df["region_year_idx"])

# 10. Build PyMC arrays

# --- For Beta model (typing fraction, always available) ---
delta_obs = df["delta"].to_numpy().astype(float)
N_total = df["DENV_total"].to_numpy().astype(int)

# --- For Multinomial model (subtypes, only when typed) ---
Y_multinomial = df[sero_cols].to_numpy().astype(int)
N_typed = df["N_typed"].to_numpy().astype(int)

# --- Indices ---
state_idx = df["state_idx"].to_numpy().astype(int)
region_idx = df['region_idx'].to_numpy().astype(int)
month_idx = df["month_idx"].to_numpy().astype(int)
year_idx = df["year_idx"].to_numpy().astype(int)
state_year_idx = df["state_year_idx"].to_numpy().astype(int)
region_year_idx = df["region_year_idx"].to_numpy().astype(int)
n_states = int(len(df['UF'].unique()))
n_months = int(len(df["month_idx"].unique()))
n_years = int(df["year_idx"].max() + 1)
n_state_years = len(state_year_labels)
n_region_years = len(region_year_labels)
n_covariates = int(X.shape[1])
n_serotypes = len(sero_cols)
n_regions = len(region_labels)

# This assumes each state-year belongs to exactly 1 region-year
state_year_to_region_year = df.groupby("state_year_idx")["region_year_idx"].first().sort_index().tolist()

########################
## Preparing the model##
########################

with pm.Model() as dengue_model:

    # --- Typing Effort Model ---
    # (original plan)
    # N^*_{s,t} ~ Binomial(N_{total,s,t}, \delta_{s,t}),
    # where N_{total,s,t} the observed total dengue incidence and \delta_{s,t} the fraction that gets subtyped.
    #
    # 𝛿_{s,t} ~ 𝐵𝑒𝑡𝑎(𝜇_{s,t}.𝜙, (1 − 𝜇_{s,t}).𝜙)
    # logit(𝜇_{s,t}) = \beta + \beta_s + \beta_t + \sum_j \beta_j X_{s,j}

    # \beta (global intercept)
    beta = pm.Normal("beta", mu=-4.5, sigma=1.5)

    # \beta_{s,t}: State-year-specific typing effort random effect: \beta_{s,t} = \beta_{r[s],t} + \epsilon_{s,t}
    # Region-year effect
    beta_rt_shrinkage = pm.Exponential("beta_rt_shrinkage", 1)
    beta_rt_sigma = pm.HalfNormal("beta_rt_sigma", sigma=beta_rt_shrinkage, shape=n_region_years)
    beta_rt = pm.Normal("beta_rt", mu=0.0, sigma=beta_rt_sigma, shape=n_region_years)
    # State-year deviation from region-year
    eps_st_sigma = pm.Deterministic("eps_st_sigma", beta_rt_sigma[state_year_to_region_year]/2)
    eps_st = pm.Normal("eps_st", mu=0.0, sigma=eps_st_sigma, shape=n_state_years)
    # Final state-year effect
    beta_st = pm.Deterministic("beta_st", beta_rt[region_year_idx] + eps_st[state_year_idx])


    # \beta_{s,t}: State-year-specific typing effort random effect
    # sigma_st = pm.HalfNormal("sigma_st", sigma=2)
    # beta_st = pm.Normal("beta_st", mu=0.0, sigma=sigma_st, shape=n_state_years)

    # \beta_s (CAR prior)
    # sigma_car = pm.HalfNormal("sigma_car", sigma=1)
    # beta_s = pm.MvNormal("beta_s", mu=np.zeros(n_states), cov=sigma_car**2 * np.linalg.inv(Q), shape=n_states) #--> CAR prior

    # \beta_t (random walk with drift) 
    # Can be expanded to include piecewise-continuous drift or steps in surveillance efforts
    # sigma_time = pm.HalfNormal("sigma_time", sigma=1)
    # drift = pm.Normal("drift", mu=0.0, sigma=1)
    # beta_t_raw = pm.GaussianRandomWalk("beta_t_raw", sigma=sigma_time, shape=n_months, init_dist=pm.Normal.dist(mu=0,sigma=1))
    # drift_vector = drift * np.arange(n_months)
    # beta_t = pm.Deterministic("beta_t", beta_t_raw + drift_vector)

    # \sum_j \beta_j X_{s,j} (covariates)
    # sigma_cov = pm.HalfNormal("sigma_cov", sigma=1.0, shape=n_covariates)
    # beta_cov = pm.Normal("beta_cov", mu=0.0, sigma=sigma_cov, shape=n_covariates)
    # covariates = pm.math.dot(X[state_idx], beta_cov)

    # # logit(𝜇_{s,t})
    # mu = pm.Deterministic("mu", pm.math.sigmoid(beta + beta_st))
    # # 𝛿_{s,t} ~ 𝐵𝑒𝑡𝑎(𝜇_{s,t}.𝜙, (1 − 𝜇_{s,t}).𝜙)
    # phi = pm.HalfNormal("phi", sigma=100.0)
    # alpha_beta = mu * phi 
    # beta_beta = (1 - mu) * phi 
    # delta_st = pm.Beta("delta_st", alpha=alpha_beta, beta=beta_beta, observed=delta_obs)

    # Alternative: model serotyped fraction as a logit-normal since beta is close to zero
    logit_delta_obs = np.log(delta_obs / (1 - delta_obs)) 
    logit_mu = beta  + beta_st
    # logit_delta_sigma is important because it controls the overall noise levels on the serotyped cases (lower = less noise)
    # it also controls an important trade-off in this model: the relationship between N_total and N_typed is not perfectly linear, i.e. you can't fit both N_total and delta_st perfectly
    # Values of 0.001-0.002 sacrifices delta_st for a better fit to N_total, while a value of 0.01 gives a good fit to delta_st but a poorer fit to N_typed an too much uncertainty
    # Opposed
    logit_delta_sigma = pm.HalfNormal("logit_delta_sigma", sigma=0.002) 
    logit_delta = pm.Normal("logit_delta", mu=logit_mu, sigma=logit_delta_sigma, observed=logit_delta_obs)
    delta_st = pm.Deterministic("delta_st", pm.math.sigmoid(logit_delta))

    # N^*_{s,t} ~ Binomial(N_{total,s,t}, \delta_{s,t})
    N_typed_latent = pm.Binomial("N_typed_latent", n=N_total, p=delta_st, observed=N_typed)

    # --- Subtype Composition Model ---
    # p_{i,s,t} ~ Dirichlet(\theta_{i,s,t})
    # log 𝜃_{i,s,t} = 𝛼 + 𝛼_s + 𝛼_t + 𝛼_i + 𝛼_{i,t} + 𝛼_{s,i}    

    # Global intercept
    # Changes \theta_{i,s,t} identically for every subtype --> Changes overall uncertainty
    alpha = pm.Normal("alpha", mu=0.0, sigma=10.0)

    # α_s: state-specific random effect
    # Make uncertainty on subtype proportions state-dependent
    alpha_s_sigma = pm.HalfNormal("alpha_s_sigma", sigma=1.0)
    alpha_s = pm.Normal("alpha_s", mu=0.0, sigma=alpha_s_sigma, shape=n_states)

    # α_t: global temporal RW(1)
    # Make uncertainty on subtype proportions time-dependent --> shrinkage has little impact on speed of drifting
    # Drift can grow or shrink uncertainty in the subtyping over time --> removed because no drift detected
    # Took this term out while leaving α_{i,t} in and found out it's pretty much redundant
    rw_shrinkage = pm.HalfNormal("rw_shrinkage", sigma=0.001) # Value: 0.0001 --> flat; still need to find the "sweet spot"; try 0.001
    #alpha_t_sigma = pm.HalfNormal("alpha_t_sigma", sigma=rw_shrinkage)
    #alpha_t = pm.GaussianRandomWalk("alpha_t", sigma=alpha_t_sigma, shape=n_months, init_dist=pm.Normal.dist(0, 0.1))

    # α_i: serotype-specific baseline
    # Model the time-independent Brasil-average subtype composition
    alpha_i_sigma = pm.HalfNormal("alpha_i_sigma", sigma=1.0)
    alpha_i = pm.Normal("alpha_i", mu=0.0, sigma=alpha_i_sigma, shape=n_serotypes)

    # α_{i,t}: serotype-specific temporal trends as RW(1) -- unpooled 
    # Allows the serotype distribution to vary over time
    # Per-serotype standard deviations (with shrinkage)
    alpha_it_sigma = pm.HalfNormal("alpha_it_sigma", sigma=rw_shrinkage, shape=n_serotypes)
    # Per-serotype RW(1)
    alpha_it_list = []
    for i in range(n_serotypes):
        a = pm.GaussianRandomWalk(f"alpha_it_{i}", sigma=alpha_it_sigma[i], shape=n_months, init_dist=pm.Normal.dist(mu=0, sigma=0.1))
        alpha_it_list.append(a)
    # Stack to shape (n_months, n_serotypes)
    alpha_it = pm.Deterministic("alpha_it", pt.stack(alpha_it_list, axis=1))

    # α_{s,i}: state-serotype spatial CAR structure with inferred distance kernel
    # Models the spatial correlation between subtype compositions --> Improves fit!
    ## Define variables
    D_shared = pm.MutableData("D_shared", D_matrix)
    zeta_car = pm.Exponential("zeta_car", 1/1000)       # --> Influences how far the serotype composition neighbourhood stretches --> smaller = more local 
    a_car = pm.Beta("a_car", 2, 2)                      # --> Influences the strength of the correlation within the correlated neighbourhood --> 0 = no spatial correlation
    ## Build distance weighted kernel
    W = pt.exp(-D_shared / zeta_car)
    W = pt.set_subtensor(W[pt.arange(W.shape[0]), pt.arange(W.shape[1])], 0.0)
    ## Build degree matrix
    row_sums = W.sum(axis=1)
    D_mat = pt.diag(row_sums)
    ## Build precision matrix Q
    Q = D_mat - a_car * W
    Q = Q + 1e-6 * pt.eye(W.shape[0])  # Stabilization
    ## Use in CAR prior
    sigma_car = pm.HalfNormal("sigma_car", sigma=1.0, shape=n_serotypes) # Weakly informed because I don't want to shrink the spatial correlation too drastically
    ## loop over serotypes
    alpha_si_list = []
    for i in range(n_serotypes):
        alpha_si_list.append(pm.MvNormal(f"alpha_si_{i}", mu=np.zeros(n_states), cov=sigma_car[i]**2 * pt.nlinalg.matrix_inverse(Q), shape=n_states)) #--> CAR prior
    alpha_si = pm.Deterministic("alpha_si", pm.math.stack(alpha_si_list, axis=1))  # shape: (n_states, 4)

    # α_{i,t}: serotype-year-specific baseline + α_{i,r,t}: serotype-region-year specific baseline
    # Final puzzle piece: allow the average serotype composition to change yearly by region
    # Model serotype-by-region-by-year back as a perturbation to serotype-by-year with shrinkage to control degree of overfitting
    # First: serotype by year (with its own shrinkage)
    alpha_i_year_sigma = pm.HalfNormal("alpha_i_year_sigma", sigma=0.0005) # --> 0.001: Medium impact; Controls the degree of overfitting
    alpha_i_year = pm.Normal("alpha_i_year", mu=0.0, sigma=alpha_i_year_sigma, shape=(n_years, n_serotypes))
    # Then: serotype by region by year as deviation from its respective year
    eps_i_region_year_sigma = pm.Deterministic("eps_i_region_year_sigma", alpha_i_year_sigma/2)
    eps_i_region_year = pm.Normal("eps_i_region_year", mu=0.0, sigma=eps_i_region_year_sigma, shape=(n_region_years, n_serotypes))
    # Final serotype-region-year
    alpha_i_region_year = pm.Deterministic("alpha_i_region_year", alpha_i_year[year_idx, :] + eps_i_region_year[region_year_idx, :])

    # Construct log θ_{i,s,t}
    theta_log = (
        alpha
        + alpha_s[state_idx][:, None]               # shape (n_obs, 1)
        #+ alpha_t[month_idx][:, None]               # shape (n_obs, 1)
        + alpha_i[None, :]                          # shape (1, 4)
        + alpha_it[month_idx, :]                    # shape (n_obs, 4)
        + alpha_si[state_idx, :]                    # shape (n_obs, 4)
        + alpha_i_region_year
    )  # Result: shape (n_obs, 4)

    # Dirichlet prior for subtype fractions
    p = pm.Deterministic("p", pm.math.softmax(theta_log, axis=1))

    # --- Observed subtyped incidences ---
    # Y_{i,s,t} ~ Multinomial(N^*_{s,t}, p_{i,s,t})

    Y_obs = pm.Multinomial("Y_obs", n=N_typed_latent, p=p, observed=Y_multinomial)


########################
## Running the model  ##
########################

# NUTS
with dengue_model:
    trace = pm.sample(200, tune=100, target_accept=0.99, chains=4, cores=4, init='auto', progressbar=True)

# Plot posterior predictive checks
with dengue_model:
    ppc = pm.sample_posterior_predictive(trace)
arviz.plot_ppc(ppc)
plt.savefig('ppc.pdf')
plt.close()

# Assume `trace` is the result of pm.sample()
arviz.to_netcdf(trace, "trace.nc")
arviz.to_netcdf(ppc, "ppc.nc")

# Traceplot
variables2plot = ['beta', 'beta_rt', 'beta_rt_shrinkage', 'beta_rt_sigma',
                  'rw_shrinkage', 'alpha_t_sigma', 'zeta_car', 'a_car', 'rho',
                  'alpha', 'alpha_s', 'alpha_i', 'alpha_i_sigma', 'alpha_it_sigma',
                  'zeta_car', 'a_car', 'sigma_car', 'alpha_i_year_sigma', 'alpha_i_year'
                ]

for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'trace-{var}_typing-effort-model.pdf')
    plt.close()

# Print summary
summary_df = arviz.summary(trace, round_to=3)
print(summary_df)









