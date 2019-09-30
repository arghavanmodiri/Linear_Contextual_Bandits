# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

#%%
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportions_ztest
import statsmodels.api as sm
from statsmodels.formula.api import ols
import itertools
import math
import subprocess
import os

#%%
def print_mean_stdev(metric: dict, user_range: tuple):
    print('User range: ' + str(user_range))
    rows = []
    for policy, df in metric.items():
        row_dict = {}
        policy_name = policy.replace('_', ' ').title()
        print('Policy: ' + policy_name)
        row_dict['Policy'] = policy_name
        mean = df[user_range[0]:user_range[1]].values.mean()
        stdev = df[user_range[0]:user_range[1]].values.std(ddof=1)
        print('Mean: ' + str(mean))
        print('Standard deviation: ' + str(stdev))
        row_dict['Mean'] = mean
        row_dict['Standard Deviation'] = stdev
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.set_index('Policy')

#%%
def ttest(metric: dict, user_range: tuple):
    rows = []
    for c in itertools.combinations(list(metric.keys()), 2):
        row_dict = {}
        print(c)
        row_dict['Policies'] = '\makecell{' + c[0].replace(
            '_', ' ').title() + ' \\\\and\\\\ ' + c[1].replace(
                '_', ' ').title() + '}'
        test_result = stats.ttest_ind(
            metric[c[0]][user_range[0]:user_range[1]], metric[c[1]][user_range[0]:user_range[1]], 
                axis=None, equal_var = False)
        print(test_result)
        row_dict['P-value'] = test_result[1]
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.set_index('Policies')


#%%
def cohens_d(metric: dict, user_range: tuple):
    rows = []
    for c in itertools.combinations(list(metric.keys()), 2):
        row_dict = {}
        print(c)
        row_dict['Policies'] = '\makecell{' + c[0].replace(
            '_', ' ').title() + ' \\\\and\\\\ ' + c[1].replace(
                '_', ' ').title() + '}'
        d = (metric[c[0]][user_range[0]:user_range[1]].values.mean() - 
        metric[c[1]][user_range[0]:user_range[1]].values.mean()) / math.sqrt(
            (metric[c[0]][user_range[0]:user_range[1]].values.std(ddof=1)**2
                 + metric[c[1]][user_range[0]:user_range[1]].values.std(ddof=1)**2) / 2)
        print(d)
        row_dict["Cohen's d"] = d
        rows.append(row_dict)

    df = pd.DataFrame(rows)
    return df.set_index('Policies')

#%%
def export_tex(df: pd.DataFrame, file_name: str, file_loc: str):
    loc = file_loc + 'tables/'
    if not os.path.exists(loc):
        os.makedirs(loc)
        
    with open(loc + file_name, 'wb') as f:
        f.write(bytes(template.format(df.to_latex(column_format='ccc', escape=False)),'UTF-8'))
    subprocess.call(['pdflatex', file_name], cwd=loc)

#%%
file_loc = './saved_output/Aug_6/crossover_against_two/raw_data/2019-08-06_192359/'
#file_loc = './saved_output/Aug_6/crossover_for_two/raw_data/2019-08-06_202047/'
#file_loc = './saved_output/Aug_6/partial_crossover_against_one/raw_data/2019-08-07_8345/'
#file_loc = './saved_output/Aug_6/noncrossover_two_main_one_optimum/raw_data/2019-08-07_101150/'
user_range = (1000, 1500)
regrets = {}
optimal_ratio = {}
policies = ['thompson_with_interaction', 'thompson_without_interaction', 'independent_bandit']
template = r'''\documentclass[preview]{{standalone}}
\usepackage{{booktabs}}
\usepackage{{makecell}}
\begin{{document}}
{}\end{{document}}
'''
pd.set_option('display.max_colwidth', 1000)

#%%
for policy in policies:
    regrets[policy] = pd.read_csv(
        file_loc + policy + '_regrets.csv').drop(columns=['iteration'])
    optimal_ratio[policy] = pd.read_csv(
        file_loc + policy + '_optimal_action_ratio.csv').drop(columns=['iteration'])


#%%
for policy, df in optimal_ratio.items():
    optimal_ratio[policy] = df.mean(axis = 1, skipna = True)


#%%
df = print_mean_stdev(regrets, user_range)
export_tex(df, 'regret_mean.tex', file_loc)

df_test = ttest(regrets, user_range)
df_d = cohens_d(regrets, user_range)
df_combined = pd.concat([df_test, df_d], axis=1)
export_tex(df_combined, 'regret_effect.tex', file_loc)

#%%
df = print_mean_stdev(optimal_ratio, user_range)
export_tex(df, 'optimal_ratio_mean.tex', file_loc)

df_test = ttest(optimal_ratio, user_range)
df_d = cohens_d(optimal_ratio, user_range)
df_combined = pd.concat([df_test, df_d], axis=1)
export_tex(df_combined, 'optimal_ratio_effect.tex', file_loc)

#%%



#%%