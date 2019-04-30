'''
Purpose: Process thompson_ols simulation data and compute biases

Date started: March 15, 2019

Input Data
-thompson_ols_d1.csv
-thompson_ols_d1x1.csv

Notes:

'''

import os
import pandas as pd
import numpy as np

# List of simulations
#sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect"]
#sim_names = ["_SCC", "_SUC", "_LCC", "_LUC"]
sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect",
            "SmallMABUnderFitUnder","LargeMABUnderFitUnder","SmallRandomCorrect", "LargeRandomCorrect",
            "SmallRandomUnder", "LargeRandomUnder", "SmallUniformPolicy", "LargeUniformPolicy"]
sim_names = ["_SCC", "_SUC", "_LCC", "_LUC", "_SUU", "_LUU", "_SRC", "_LRC", "_SRU", "_LRU", "_SR","_LR"]
true_fit_d1 = [0.25, 0.25, 0.25, 0.25]
true_fit_d1x1 = [-0.4, -0.4, -0.8, -0.8]
sim_count = 0

os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/' + sim_list[0])

# Load d1 random OLS fit
d1_rand_ols = pd.read_csv("random_ols_d1.csv")
d1_rand_ols = d1_rand_ols.drop(columns = ["iteration"])

# Compute mean coeff across simulations
d1_bias_rand_ols = d1_rand_ols.mean(axis=1) - true_fit_d1[0]

# Compute Q1 for d1 random ols fit
d1_bias_rand_ols_q1 = np.round(np.mean(d1_bias_rand_ols[0:250]),4)
print("[0,250] bias(d1) random fit model " + str(d1_bias_rand_ols_q1))

# Compute Q4 for d1 random ols fit
d1_bias_rand_ols_q4 = np.round(np.mean(d1_bias_rand_ols[750:1000]),4)
print("[750,1000] bias(d1) random fit model " + str(d1_bias_rand_ols_q4))

# Load d1 random OLS fit
d1x1_rand_ols = pd.read_csv("random_ols_d1x1.csv")
d1x1_rand_ols = d1x1_rand_ols.drop(columns = ["iteration"])

# Compute mean coeff across simulations
d1x1_bias_rand_ols = d1x1_rand_ols.mean(axis=1) - true_fit_d1x1[0]

# Compute Q1 for d1 random ols fit
d1x1_bias_rand_ols_q1 = np.round(np.mean(d1x1_bias_rand_ols[0:250]),4)
print("[0,250] bias(d1x1) random fit model " + str(d1x1_bias_rand_ols_q1))

# Compute Q4 for d1 random ols fit
d1x1_bias_rand_ols_q4 = np.round(np.mean(d1x1_bias_rand_ols[750:1000]),4)
print("[750,1000] bias(d1x1) random fit model " + str(d1x1_bias_rand_ols_q4))
                             



'''
# Load d1 thompson sampling ols bias data
d1_ts_ols = pd.read_csv("thompson_ols_d1.csv")
d1_ts_ols = d1_ts_ols.drop(columns = ["iteration"])

# Compute mean coeff across simulations
d1_bias_ts_ols = d1_ts_ols.mean(axis=1) - true_fit_d1[0]

# Compute 
d1_bias_ts_ols_q1 = np.mean(d1_bias_ts_ols[0:250])
print("[0,250] bias(d1) fit model " + str(d1_bias_ts_ols_q1))

d1_bias_ts_ols_q4 = np.mean(d1_bias_ts_ols[750:1000])
print("[750,1000] bias(d1) fit model " + str(d1_bias_ts_ols_q4))


# Load d1*x1 thompson sampling ols bias data
d1x1_ts_ols = pd.read_csv("thompson_ols_d1x1.csv")
d1x1_ts_ols = d1x1_ts_ols.drop(columns = ["iteration"])

# Compute mean coeff across simulations
d1x1_bias_ts_ols = d1x1_ts_ols.mean(axis=1) - true_fit_d1[0]

# Compute 
d1x1_bias_ts_ols_q1 = np.mean(d1x1_bias_ts_ols[0:250])
print("[0,250] bias(d1) fit model " + str(d1x1_bias_ts_ols_q1))

d1x1_bias_ts_ols_q4 = np.mean(d1x1_bias_ts_ols[750:1000])
print("[750,1000] bias(d1) fit model " + str(d1x1_bias_ts_ols_q4))

'''

'''
# Loop over simulations
for sim_type in sim_list:

    # Change directory
    os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/' + sim_list[sim_count])

    # Define parameters
    n_sim = 500
    n_user = 1000
    n_q1 = 250
    n_q4 = 750
'''
