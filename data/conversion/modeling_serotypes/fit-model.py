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
X = pd.read_csv('../../interim/state_covariates.csv', index_col=0)
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

    # N^*_{s,t} ~ Poisson(N_{total,s,t} * \delta_{s,t}) --> Less brutal likelihood than Binomial
    # lambda_ = pm.Deterministic("lambda_", N_total * delta_st)
    # N_typed_latent = pm.Poisson("N_typed_latent", mu=lambda_, observed=N_typed)


    # --- Subtype Composition Model ---
    # p_{i,s,t} ~ Dirichlet(\theta_{i,s,t})
    # log 𝜃_{i,s,t} = 𝛼 + 𝛼_s + 𝛼_t + 𝛼_i + 𝛼_{i,t} + 𝛼_{s,i}    

    # Try to combine an AR(p) with a CAR prior on every timestep in the past
    # priors for zetas and a per serotype per lag
    p = 1
    rw_shrinkage = pm.HalfNormal("rw_shrinkage", sigma=0.001)
    alpha_t_sigma = pm.HalfNormal("alpha_t_sigma", sigma=rw_shrinkage, shape=n_serotypes)
    zeta_car = pm.HalfNormal("zeta_car", 100, shape=(n_serotypes, p))
    a_car = pm.Beta("a_car", 2, 2, shape=(n_serotypes, p))
    rho = pm.Beta("rho", 2, 2, shape=(n_serotypes, p))
    D_shared = pm.MutableData("D_shared", D_matrix)

    # Pair-wise kernel first
    # D_shared: (n_states, n_states)
    # zeta_car: (n_serotypes, p)
    # We need to broadcast D_shared against zeta
    D_expanded = D_shared[None, None, :, :]
    zeta_expanded = zeta_car[:,:,None, None]
    # kernel = exp(-D_shared / zeta)
    W = pt.exp(-D_expanded / zeta_expanded)
    # Construct degree tensor (matrix equivalent: row sums of weighted distance matrix on diagonal of eye(n_states))
    degree = pt.sum(W, axis=-1)[:, :, :, None]
    I = pt.eye(n_states)[None, None, :, :]
    D = I * degree
    # Q = D - a * W + jitter
    jitter = 1e-6 * pt.diag(pt.ones(n_states))
    jitter = jitter[None, None, :, :]
    Q = D - a_car[:,:,None, None] * W + jitter
    # Q shape == (n_serotypes, p, n_states, n_states)

    # Compute the Cholesky of Q
    chol = pt.slinalg.cholesky(Q)
    # solve_triangular to 
    L_inv = pytensor.tensor.slinalg.solve_triangular(chol, pt.eye(n_states), lower=True)  # (n_serotypes, p, n_states, n_states)
    # Transpose to upper bidiagonal
    L_inv_T = L_inv.transpose((0, 1, 3, 2))  # (n_serotypes, p, n_states, n_states)
    # Now apply scaling by standard deviations and reconstruct chol
    alpha_scale = alpha_t_sigma[:,None,None,None]  # adding three dimensions
    chol = alpha_scale * L_inv_T

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    chol_matrix_lag = chol.transpose((1, 0, 2, 3)) # shape == (p, n_serotypes, n_states, n_states) --> makes more sense

    # initialise initial condition
    alpha_init = pm.Normal("alpha_init", mu=0, sigma=1, shape=(p, n_serotypes, n_states))

    # --- FIX: epsilon now includes innovations for each lag at each timestep ---
    epsilon = pm.Normal("epsilon", 0, 1, shape=(n_months - p, p, n_serotypes, n_states))

    def arp_step(previous_vals, epsilon_t, rho, chol_matrix_lag):
        """
        previous_vals: (p, n_serotypes, n_states)
        epsilon_t: (p, n_serotypes, n_states)  # innovations for each lag at current timestep
        """
        # Compute mean AR contribution across lags
        mean = sum(rho[:, lag][:, None] * previous_vals[lag] for lag in range(p))  # shape (n_serotypes, n_states)

        # Compute shocks per lag using the correct lag-specific chol matrix
        shocks = []
        for lag in range(p):
            # epsilon_t[lag]: (n_serotypes, n_states)
            # chol_matrix_lag[lag]: (n_serotypes, n_states, n_states)
            # We want: for each serotype, epsilon_t[lag][serotype] @ chol_matrix_lag[lag][serotype]
            # Use batched dot product per serotype
            # pytensor batched_dot expects shape (batch, n) @ (batch, n, m) -> (batch, m)
            shock_lag = pt.batched_dot(epsilon_t[lag], chol_matrix_lag[lag])  # shape (n_serotypes, n_states)
            shocks.append(shock_lag)
        shock = sum(shocks)  # sum over lags (n_serotypes, n_states)
        new_vals = mean + shock  # (n_serotypes, n_states)

        # Shift lags down by one, discard oldest lag, prepend new_vals as newest lag
        # previous_vals[0] is lag 0 (most recent), previous_vals[1] is lag 1, etc.
        updated_vals = pt.concatenate(
            [new_vals[None, :, :], previous_vals[:-1]], axis=0
        )  # shape (p, n_serotypes, n_states)
        
        return updated_vals

    sequences, _ = pytensor.scan(
        fn=arp_step,
        sequences=epsilon,
        outputs_info=alpha_init,
        non_sequences=[rho, chol_matrix_lag],
    )

    # sequences: (n_months - p, p, n_serotypes, n_states)
    # alpha_init: (p, n_serotypes, n_states)
    theta_log_final = pt.concatenate([pt.repeat(alpha_init[None, :, :, :], p, axis=0), sequences], axis=0)
    # Step 3: slice lag zero (p=0) over full time axis
    theta_log_final = theta_log_final[:, 0, :, :]  # shape (n_months, n_serotypes, n_states)
    # Step 4: convert to flat format
    theta_log_final_flat = theta_log_final.reshape((len(df), n_serotypes))

    # combine all terms
    theta_log =  theta_log_final_flat

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  
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
    trace = pm.sample(200, tune=100, target_accept=0.95, chains=4, cores=4, init='auto', progressbar=True)

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
                  'rw_shrinkage', 'alpha_t_sigma', 'zeta_car', 'a_car', 'rho'
                ]

for var in variables2plot:
    arviz.plot_trace(trace, var_names=[var]) 
    plt.savefig(f'trace-{var}_typing-effort-model.pdf')
    plt.close()

# Print summary
summary_df = arviz.summary(trace, round_to=3)
print(summary_df)









