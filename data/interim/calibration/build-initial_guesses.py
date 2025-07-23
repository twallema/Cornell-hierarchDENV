import pandas as pd

# model names
model_names = ['SIR-1S', 'SIR-4S']
# get ufs
ufs = pd.read_csv('../adjacency_matrix.csv', index_col=0).index.to_list()
# seasons
seasons = ['2023-2024', '2022-2023', '2021-2022', '2020-2021', '2019-2020', '2018-2019', '2017-2018', '2016-2017', '2015-2016', '2014-2015']
# parameter names per model
parameter_names = [
    ['rho_report', 'f_R', 'f_I', 'beta'] + [f'delta_beta_temporal_{i}' for i in range(12)],
    ['rho_report', 'f_R_0', 'f_R_1', 'f_R_2', 'f_R_3', 'f_I_0', 'f_I_1', 'f_I_2', 'f_I_3', 'beta_0', 'beta_1', 'beta_2', 'beta_3'] + [f'delta_beta_temporal_{i}' for i in range(12)],
]
# initial values per model
parameter_values = [
    [0.5, 0.5, 1E-04, 0.3] + 11*[0.01,],
    [0.5,] + 4*[0.5] + 4*[1E-03,] + 4*[0.3,] + 11*[0.01,]
]

# Build the dataframe
dfs = []
for model, params, values in zip(model_names, parameter_names, parameter_values):
    for uf in ufs:
        for param, val in zip(params, values):
            row = pd.DataFrame(
                [val] * len(seasons),
                index=seasons,
                columns=['value']
            ).T
            row.index = pd.MultiIndex.from_tuples([(model, uf, param)], names=['model', 'uf', 'parameter'])
            dfs.append(row)

# Concatenate all rows
final_df = pd.concat(dfs)

# Show result
final_df.to_csv('initial_guesses.csv')