'''
Purpose: Process simulation data and make Tables

Date started: March 13, 2019

Input Data
- thompson_bias_in_coeff.csv
- thompson_context_action.csv
- thompson_regrets.csv

Notes:

'''

import os
import pandas as pd
import numpy as np


# Change directory
os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/raw_data2019-03-12_204526')

# Define parameters
n_sim = 500
n_user = 1000
n_q1 = 250
n_q4 = 750

# Load data of MAB bias
df_bias_thompson = pd.read_csv('thompson_bias_in_coeff.csv',skiprows=1)

## Compute mean bias (across simulation) for each itteration
# Subgroup effect d1
df_bias_thompson["d1.0"] = df_bias_thompson["d1"] 
d1_bias = [[np.mean([df_bias_thompson["d1." + str(sim)][user] for sim in range(n_sim)])][0] for user in range(n_user)]
df_bias_thompson["d1_bias"] = d1_bias

# Hetregenous effect d1*x1
df_bias_thompson["d1*x1.0"] = df_bias_thompson["d1*x1"] 
d1x1_bias = [[np.mean([df_bias_thompson["d1*x1." + str(sim)][user] for sim in range(n_sim)])][0] for user in range(n_user)]
df_bias_thompson["d1x1_bias"] = d1x1_bias


# Compute mean 

# Compute bias for [0,250]
d1_bias_q1 = np.mean(d1_bias[0:(n_q1+1)])
print("[0,250] bias(d1) TS correct spec:" + str(d1_bias_q1))

d1x1_bias_q1 = np.mean(d1x1_bias[0:(n_q1+1)])
print("[0,250] bias(d1*x1) TS correct spec:" + str(d1x1_bias_q1))

# Compute bias for [750,1000]
d1_bias_q4 = np.mean(d1_bias[n_q4:(n_user+1)])
print("[750,1000] bias(d1) TS correct spec:" + str(d1_bias_q4))

d1x1_bias_q4 = np.mean(d1x1_bias[n_q4:(n_user+1)])
print("[750,1000] bias(d1*x1) TS correct spec:" + str(d1x1_bias_q4))




