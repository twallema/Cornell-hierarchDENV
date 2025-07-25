# Cornell-hierarchDENV

Here we list a description of all datasets, raw or interim and their conversion scripts.

## Raw

### Sprint 2025

Downloaded using the instructions under '2 - Using FTPWeb' on https://sprint.mosqlimate.org/data/.

+ `climate.csv`: Reanalysis of hourly data from ERA5, summarized by week by the Mosqlimate project. Period: Epiweek 201001 to epiweek 2025174. Aggregation: Temperature, humidity, and precipitation, originally by hour, were first aggregated by day (min, max, mean), and these daily measures were aggregated by epidemiological week (mean). Added to .gitignore.

+ `datasus_population_2001_2024.csv`: Population data (source: SVS). Files with population by Brazilian municipality and year (2001 - 2024). Source: http://tabnet.datasus.gov.br/cgi/deftohtm.exe?ibge/cnv/popsvs2024br.def 

+ `map_regional_health.csv`: Link between each city and its regional and macroregional health center (source = IBGE).

+ `shape_muni.gpkg`: Geometry of Brazilian municipalities in `shape_muni.gpkg` (source = IBGE).

### DENV datasus

These data are partly confidential and can be found on the Bento lab box.

## Interim

+ `BR_hospital-beds-per-capita_2005-2023.csv`: Contains the total number of hospital beds per 1000 inhabitants, the number of hospital beds available to sus per 1000 inhabitants. Raw data were available from 2005-2023, but data were padded from 1996-2025. Data have a monthly frequency, indexed at the of the month in the `date` column. Data are available per Brazilian state. 

+ `state_covariates.csv`: Contains, per Brazilian state: "population", "population_density", "gini_population", "hospital_beds_per_1000", "rel_humid_med", "precip_med", "temp_med", "population_weighted_latitude". 

+ `weighted_distance_matrix.csv`: Contains a square origin-destination-type distance matrix with the population-weighted distance between Brazil's 27 states.

+ `population_weighted_centroid.png`: Illustrates the difference between the geometric and population-weighted centroid of the Brazilian states.

+ `adjacency_matrix.csv`: Contains a square origin-destination-type adjacency matrix of the Brasilian states.

### DENV datasus

+ `DENV-serotypes_1996-2025_monthly/weekly.csv`: Weekly or monthly total confirmed (not discarded) DENV cases, as well as number of serotyped cases per DENV serotype. Generated using `DENV_datasus_conversion.py`. 

### Imputed DENV datasus

+ `DENV-serotypes-imputed_1996-2025_monthly.csv`: Monthly total confirmed DENV cases in every Brazilian UF (column: `DENV_total`), identical to `~/data/interim/DENV_datasus/DENV-serotypes_1996-2025_monthly.csv`. Contains the serotype fractions in columns `p_1`, `p_2`, `p_3` and `p_4`, as generated using the Bayesian serotype imputation model in `~/data/conversion/modeling_serotypes/fit-model.py`.

## Conversion scripts

+ `access_to_healthcare.R`: Downloads raw data on the number of available hospital beds in Brazil from DataSus and formats it into `BR_hospital-beds-per-capita_2005-2023.csv`. Data are not saved locally, and downloading them took about 12 hours.

+ `build_distance-matrix_covariates.ipynb`: Notebook used to build a demographically-weighted distance matrix between Brazilian states `weighted_distance_matrix.csv`, as well as a dataset of Brazilian state-level covariates relevant to DENV transmission in `state_covariates.csv`. Recommend working in the environment `GEOPANDAS_ENV.yml`.

+ `DENV_datasus_conversion.py`: Script used to convert the (partly confidential) raw linelisted datasus DENV data (`~/data/raw/DENV_datasus/`) into a more pleasant interim format.

### Modeling serotypes

This folder contains a standalone Bayesian model used to smooth and impute the serotyped data.

+ `fit-model.py`: Implementation of a Bayesian model (pyMC) to impute the DENV datasus serotype data (`~/data/interim/DENV-serotypes_1996-2025_monthly/weekly.csv`).
+ `SEROTYPE_MODEL_ENV.yml`: Recommended environment to run the serotype model in.
+ `visualise-model.ipynb`: Can be used to visualise the output of a run.
