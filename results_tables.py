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

# List of simulations
sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect", "SmallUniformPolicy", "LargeUniformPolicy"]
#sim_list = ["SmallUniformPolicy", "LargeUniformPolicy"]
#sim_names = ["_SR", "_LR"]
sim_names = ["_SCC", "_SUC", "_LCC", "_LUC", "_SR", "_LR"]
sim_count = 0

# Loop over simulations
for sim_type in sim_list:

    # Change directory
    os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/' + sim_list[sim_count])

    # Define parameters
    n_sim = 500
    n_user = 1000
    n_q1 = 250
    n_q4 = 750
    
    # None value variables (underspecified and uniform)
    d1x1_bias = None
    d1_bias_q1 = None
    d1x1_bias_q1 = None
    d1_bias_q4 = None
    d1x1_bias_q4 = None

    ### MAB Policy ###

    if sim_names[sim_count][2] != "R":
        # Load data of MAB bias
        df_bias_thompson = pd.read_csv('thompson_bias_in_coeff.csv',skiprows=1)

        ## Compute mean bias (across simulation) for each itteration
        # Subgroup effect d1
        df_bias_thompson["d1.0"] = df_bias_thompson["d1"] 
        d1_bias = [[np.mean([df_bias_thompson["d1." + str(sim)][user] for sim in range(n_sim)])][0] for user in range(n_user)]
        df_bias_thompson["d1_bias"] = d1_bias

        if sim_names[sim_count][2] != "U":
            # Hetregenous effect d1*x1
            df_bias_thompson["d1*x1.0"] = df_bias_thompson["d1*x1"] 
            d1x1_bias = [[np.mean([df_bias_thompson["d1*x1." + str(sim)][user] for sim in range(n_sim)])][0] for user in range(n_user)]
            df_bias_thompson["d1x1_bias"] = d1x1_bias

        # Compute bias for [0,250]
        d1_bias_q1 = np.round(np.mean(d1_bias[0:(n_q1+1)]),4)
        print("[0,250] bias(d1) " + sim_names[sim_count][1:4] + " " + str(d1_bias_q1))

        if sim_names[sim_count][2] != "U":
            d1x1_bias_q1 = np.round(np.mean(d1x1_bias[0:(n_q1+1)]),4)
            print("[0,250] bias(d1*x1) " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_q1))

        # Compute bias for [750,1000]
        d1_bias_q4 = np.round(np.mean(d1_bias[n_q4:(n_user+1)]),4)
        print("[750,1000] bias(d1) " + sim_names[sim_count][1:4] + " " + str(d1_bias_q4))

        if sim_names[sim_count][2] != "U":
            d1x1_bias_q4 = np.round(np.mean(d1x1_bias[n_q4:(n_user+1)]),4)
            print("[750,1000] bias(d1*x1) "+ sim_names[sim_count][1:4] + " " + str(d1x1_bias_q4))

    # Load regret data
    df_regret_thompson = pd.read_csv('thompson_regrets.csv')
    df_regret_thompson = df_regret_thompson.drop(columns = ["iteration"])

    # Compute mean regret (across simulation) for each itteration
    df_regret = df_regret_thompson.mean(axis=1)

    # Compute regret for [0,250]
    df_regret_q1 = np.round_(np.mean(df_regret[0:(n_q1+1)]),2)
    print("[0,250] regret TS " + sim_names[sim_count][1:4] + " "+ str(df_regret_q1))

    # Compute regret for [750,1000]
    df_regret_q4 = np.round(np.mean(df_regret[n_q4:(n_user+1)]),2)
    print("[750,1000] regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_q4))

    # Load action choosen given context data
    df_action_context_thompson = pd.read_csv("thompson_context_action.csv",skiprows=1)
    df_action_context_thompson["x0_d1.0"] = df_action_context_thompson["x0_d1"] 
    df_action_context_thompson["x0_d0.0"] = df_action_context_thompson["x0_d0"]
    df_action_context_thompson["x1_d1.0"] = df_action_context_thompson["x1_d1"] 
    df_action_context_thompson["x1_d0.0"] = df_action_context_thompson["x1_d0"] 

    # Compute prop. of optimal action choosen (X = 0)
    count_x0_optimal = [sum([df_action_context_thompson["x0_d1." + str(sim)][user] for sim in range(n_sim)]) for user in range(n_user)]
    count_x0_total = [sum([(df_action_context_thompson["x0_d1." + str(sim)][user] + df_action_context_thompson["x0_d0." + str(sim)][user]) for sim in range(n_sim)]) for user in range(n_user)]
    prop_x0_optimal = [count_x0_optimal[user]/count_x0_total[user] for user in range(n_user)]
    df_action_context_thompson["prop_x0_optimal"] = prop_x0_optimal

    # Compute prop. of optimal action choosen (X = 1)
    count_x1_optimal = [sum([df_action_context_thompson["x1_d0." + str(sim)][user] for sim in range(n_sim)]) for user in range(n_user)]
    count_x1_total = [sum([(df_action_context_thompson["x1_d1." + str(sim)][user] + df_action_context_thompson["x1_d0." + str(sim)][user]) for sim in range(n_sim)]) for user in range(n_user)]
    prop_x1_optimal = [count_x1_optimal[user]/count_x1_total[user] for user in range(n_user)]
    df_action_context_thompson["prop_x1_optimal"] = prop_x1_optimal


    # Compute prop. of optimal action choosen (X = 0) for [0,250]
    prop_x0_optimal_q1 = np.round(np.mean(prop_x0_optimal[0:(n_q1+1)]), 2)
    print("[0,250] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q1))

    # Compute prop. of optimal action choosen (X = 0) for [750,1000]
    prop_x0_optimal_q4 = np.round(np.mean(prop_x0_optimal[n_q4:(n_user+1)]),2)
    print("[750,1000] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q4))

    # Compute prop. of optimal action choosen (X = 1) for [0,250]
    prop_x1_optimal_q1 = np.round(np.mean(prop_x1_optimal[0:(n_q1+1)]),2)
    print("[0,250] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q1))

    # Compute prop. of optimal action choosen (X = 1) for [750,1000]
    prop_x1_optimal_q4 = np.round(np.mean(prop_x1_optimal[n_q4:(n_user+1)]),2)
    print("[750,1000] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q4))

    # End of a simulation
    print("Simulation results for " +  sim_list[sim_count] + " are finished")

    # Save data frame
    data_list = [d1_bias_q1, d1x1_bias_q1, d1_bias_q4, d1x1_bias_q4, df_regret_q1, df_regret_q4,
                 prop_x0_optimal_q1, prop_x0_optimal_q4, prop_x1_optimal_q1, prop_x1_optimal_q4]
    cols = ["d1_bias_q1", "d1x1_bias_q1", "d1_bias_q4", "d1x1_bias_q4", "df_regret_q1", "df_regret_q4",
                 "prop_x0_optimal_q1", "prop_x0_optimal_q4", "prop_x1_optimal_q1", "prop_x1_optimal_q4"]
    cols = [cols[i] + sim_names[sim_count] for i in range(len(cols))]
    results_df = pd.DataFrame([data_list], columns=cols)
    results_df.to_csv("BanditSimResults" + sim_names[sim_count] +".csv")

    # Increase simulation counter
    sim_count += 1
    







