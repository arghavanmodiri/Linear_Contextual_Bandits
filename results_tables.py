'''
Purpose: Process simulation data and store in CSV files. The CSV files can be used to make Tables.

Date started: March 13, 2019

Input Data
- thompson_bias_in_coeff.csv
- thompson_context_action.csv
- thompson_regrets.csv
- thompson_ols_d1.csv
- thompson_ols_d1x1.csv
- random_ols_d1.csv
- random_ols_d1x1.csv
- random_context_action.csv
- random_regrets.csv

Notes:

'''

import os
import pandas as pd
import numpy as np

# List of simulations
#sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect", "SmallUniformPolicy", "LargeUniformPolicy"]
'''sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect",
            "SmallMABUnderFitUnder","LargeMABUnderFitUnder","SmallRandomCorrect", "LargeRandomCorrect",
            "SmallRandomUnder", "LargeRandomUnder", "SmallUniformPolicy", "LargeUniformPolicy"]'''
#sim_list = ["SmallUniformPolicy", "LargeUniformPolicy"]
sim_list = ["SmallMABCorrectFitCorrectJ"]
#sim_list = ["SmallRandomCorrectJ"]
#sim_list = ["SmallUniformPolicyJ"]
#sim_names = ["_SR", "_LR"]
#sim_names = ["_SCC", "_SUC", "_LCC", "_LUC", "_SUU", "_LUU", "_SRC", "_LRC", "_SRU", "_LRU", "_SR","_LR"]
#sim_names = ["_SCC", "_SUC", "_LCC", "_LUC"]
sim_names = ["_SCC"]
#sim_names = ["_SRC"]
#sim_names = ["_SR"]
sim_count = 0
#true_fit_d1 = 0.25
true_fit_d1 = 0.6
true_fit_d1x1 = [-1.2, -1.2, -1.2, -1.2, -1.2]
#true_fit_d1x1 = [-0.4, -0.4, -0.8, -0.8, -0.4, -0.8, -0.4, -0.8, -0.4, -0.8]

# Loop over simulations
for sim_type in sim_list:

    # Change directory
    os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/' + sim_list[sim_count])

    # Define parameters
    n_sim = 500
    n_user = 1000
    n_q1 = 250
    n_q2 = 500
    n_q3 = 750

    # Not random policy and OLS fit
    if sim_list[sim_count][5] != "R" and len(sim_names[sim_count]) != 3:
        
        # None value variables (underspecified and uniform)
        d1_bias_ts_ols_q1 = None
        d1x1_bias_ts_ols_q1 = None
        d1_bias_ts_ols_q4 = None
        d1x1_bias_ts_ols_q4 = None

        ### MAB Policy ###

        # Load d1 thompson sampling ols bias data
        d1_ts_ols = pd.read_csv("thompson_ols_d1.csv")
        d1_ts_ols = d1_ts_ols.drop(columns = ["iteration"])

        # Compute mean coeff across simulations
        if sim_names[sim_count][3] != "U":
            d1_bias_ts_ols = d1_ts_ols.mean(axis=1) - true_fit_d1
        # Underspecified
        else:
            d1_bias_ts_ols = d1_ts_ols.mean(axis=1) - true_fit_d1 - true_fit_d1x1[sim_count]/2


        # d1_bias_rand_ols_se_q1 = np.round(np.std(d1_ts_ols[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)

        # Compute OLS fitted bias Q1
        d1_bias_ts_ols_q1 = np.round(np.mean(d1_bias_ts_ols[0:(n_q1+1)]),4)
        print("[0,250] bias(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_q1))

        # Compute OLS SE(d1) fitted bias Q1
        d1_bias_ts_ols_se_q1 = np.round(np.std(d1_ts_ols[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[0,250] SE(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_se_q1))

        # Compute OLS fitted bias Q2
        d1_bias_ts_ols_q2 = np.round(np.mean(d1_bias_ts_ols[n_q1:n_q2]),4)
        print("[250,500] bias(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_q2))

        # Compute OLS SE(d2) fitted bias Q2
        d1_bias_ts_ols_se_q2 = np.round(np.std(d1_ts_ols[n_q1:n_q2].mean(axis=0))/np.sqrt(n_sim),5)
        print("[250,500] SE(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_se_q2))

        # Compute OLS fitted bias Q3
        d1_bias_ts_ols_q3 = np.round(np.mean(d1_bias_ts_ols[n_q2:n_q3]),4)
        print("[500,750] bias(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_q3))

        # Compute OLS SE(d2) fitted bias Q3
        d1_bias_ts_ols_se_q3 = np.round(np.std(d1_ts_ols[n_q2:n_q3].mean(axis=0))/np.sqrt(n_sim),5)
        print("[500,750] SE(d1) fit model "  + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_se_q3))

        # Compute OLS fitted bias Q4
        d1_bias_ts_ols_q4 = np.round(np.mean(d1_bias_ts_ols[n_q3:(n_user+1)]),4)
        print("[750,1000] bias(d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_q4))

        # Compute OLS SE(d4) fitted bias Q4
        d1_bias_ts_ols_se_q4 = np.round(np.std(d1_ts_ols[n_q3:(n_user+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[750,1000] SE(d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1_bias_ts_ols_se_q4))


        # SE for d1 
        #se_ts_ols = d1_ts_ols.var(axis=1)
        #se_ts_ols = 

        # SE for d1 at Q1
        #se_ts_ols_q1 = np.sqrt(np.sum(se_ts_ols))/n_q1

        # Compute bias for d1x1 if not underspecified
        if sim_names[sim_count][3] != "U":
        
            # Load d1*x1 thompson sampling ols bias data
            d1x1_ts_ols = pd.read_csv("thompson_ols_d1x1.csv")
            d1x1_ts_ols = d1x1_ts_ols.drop(columns = ["iteration"])

            # Compute mean coeff across simulations
            d1x1_bias_ts_ols = d1x1_ts_ols.mean(axis=1) - true_fit_d1x1[sim_count]

            # Compute OLS fitted bias for interaction at Q1
            d1x1_bias_ts_ols_q1 = np.round(np.mean(d1x1_bias_ts_ols[0:(n_q1+1)]),4)
            print("[0,250] bias(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_q1))

            # Compute OLS SE(d1x1) fitted bias for interaction at Q1
            d1x1_bias_ts_ols_se_q1 = np.round(np.std(d1x1_ts_ols[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)
            print("[0,250] SE(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_se_q1))

            # Compute OLS fitted bias for interaction at Q2
            d1x1_bias_ts_ols_q2 = np.round(np.mean(d1x1_bias_ts_ols[n_q1:n_q2]),4)
            print("[250,500] bias(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_q2))

            # Compute OLS SE(d1x1) fitted bias for interaction at Q2
            d1x1_bias_ts_ols_se_q2 = np.round(np.std(d1x1_ts_ols[n_q1:n_q2].mean(axis=0))/np.sqrt(n_sim),5)
            print("[250,500] SE(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_se_q2))

            # Compute OLS fitted bias for interaction at Q3
            d1x1_bias_ts_ols_q3 = np.round(np.mean(d1x1_bias_ts_ols[n_q2:n_q3]),4)
            print("[500,750] bias(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_q3))

            # Compute OLS SE(d1x1) fitted bias for interaction at Q3
            d1x1_bias_ts_ols_se_q3 = np.round(np.std(d1x1_ts_ols[n_q2:n_q3].mean(axis=0))/np.sqrt(n_sim),5)
            print("[500,750] SE(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_se_q3))

            # Compute OLS fitted bias for interaction at Q4
            d1x1_bias_ts_ols_q4 = np.round(np.mean(d1x1_bias_ts_ols[n_q3:(n_user+1)]),4)
            print("[750,1000] bias(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_q4))

            # Compute OLS SE(d1) fitted bias for interaction at Q4
            d1x1_bias_ts_ols_se_q4 = np.round(np.std(d1x1_ts_ols[n_q3:(n_user+1)].mean(axis=0))/np.sqrt(n_sim),5)
            print("[750,1000] SE(d1x1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_bias_ts_ols_se_q4))

            # Compute OLS fitted bias for d1 + d1x1 from Q1 - Q4
            d1x1_x1_bias_ts_ols = d1_bias_ts_ols + d1x1_bias_ts_ols

            # Compute OLS Bias(d1x1 + d1) fitted bias for interaction at Q1
            d1x1_x1_bias_ts_ols_q1 = np.round(np.mean(d1x1_x1_bias_ts_ols[0:(n_q1+1)]),4)
            print("[0,250] bias(d1x1 + d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_x1_bias_ts_ols_q1))


            # Compute OLS SE(d1x1 + d1) fitted bias for interaction at Q1
            

            # Compute OLS Bias(d1x1 + d1) fitted bias for interaction at Q2
            d1x1_x1_bias_ts_ols_q2 = np.round(np.mean(d1x1_x1_bias_ts_ols[n_q1:n_q2]),4)
            print("[250,500] bias(d1x1 + d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_x1_bias_ts_ols_q2))


            # Compute OLS SE(d1x1 +d1) fitted bias for interaction at Q2
            

            # Compute OLS Bias(d1x1 +d1) fitted bias for interaction at Q3
            d1x1_x1_bias_ts_ols_q3 = np.round(np.mean(d1x1_x1_bias_ts_ols[n_q2:n_q3]),4)
            print("[500,750] bias(d1x1 + d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_x1_bias_ts_ols_q3))


            # Compute OLS SE(d1x1 +d1) fitted bias for interaction at Q3


            # Compute OLS Bias(d1x1 +d1) fitted bias for interaction at Q4
            d1x1_x1_bias_ts_ols_q4 = np.round(np.mean(d1x1_x1_bias_ts_ols[n_q3:(n_user+1)]),4)
            print("[750,1000] bias(d1x1 + d1) fit model " + sim_names[sim_count][1:4] + " " + str(d1x1_x1_bias_ts_ols_q4))


            # Compute OLS SE(d1x1 +d1) fitted bias for interaction at Q4

        # Load thompson sampling slope data
            
        # Load regret data
        df_regret_thompson = pd.read_csv('thompson_regrets.csv')
        df_regret_thompson = df_regret_thompson.drop(columns = ["iteration"])

        # Compute mean regret (across simulation) for each itteration
        df_regret = df_regret_thompson.mean(axis=1)

        # Variance of regret across simulations for each user
        # df_regret_var = df_regret_thompson.var(axis=1)

        # Compute regret for Q1
        df_regret_q1 = np.round_(np.mean(df_regret[0:(n_q1+1)]),4)
        print("[0,250] overall regret TS " + sim_names[sim_count][1:4] + " "+ str(df_regret_q1))

        # Compute SE(regret) for Q1
        df_regret_se_q1 = np.round(np.std(df_regret_thompson[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[0, 250] SE of regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_se_q1))

        # Compute regret for Q2
        df_regret_q2 = np.round_(np.mean(df_regret[n_q1:n_q2]),4)
        print("[250,500] overall regret TS " + sim_names[sim_count][1:4] + " "+ str(df_regret_q2))

        # Compute SE(regret) for Q2
        df_regret_se_q2 = np.round(np.std(df_regret_thompson[n_q1:n_q2].mean(axis=0))/np.sqrt(n_sim),5)
        print("[250, 500] SE of regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_se_q2))

        # Compute regret for Q3
        df_regret_q3 = np.round_(np.mean(df_regret[n_q2:n_q3]),4)
        print("[500,750] overall regret TS " + sim_names[sim_count][1:4] + " "+ str(df_regret_q3))

        # Compute SE(regret) for Q3
        df_regret_se_q3 = np.round(np.std(df_regret_thompson[n_q2:n_q3].mean(axis=0))/np.sqrt(n_sim),5)
        print("[500, 750] SE of regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_se_q3))

        # Compute regret for Q4
        df_regret_q4 = np.round(np.mean(df_regret[n_q3:(n_user+1)]),4)
        print("[750,1000] overall regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_q4))

        # Compute SE(regret) for Q4
        df_regret_se_q4 = np.round(np.std(df_regret_thompson[n_q3:(n_user+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[750, 1000] SE of regret TS "+ sim_names[sim_count][1:4] + " "+ str(df_regret_se_q4))

        # Load action choosen given context data
        df_action_context_thompson = pd.read_csv("thompson_context_action.csv",skiprows=1)
        df_action_context_thompson["x0_d1.0"] = df_action_context_thompson["x0_d1"] 
        df_action_context_thompson["x0_d0.0"] = df_action_context_thompson["x0_d0"]
        df_action_context_thompson["x1_d1.0"] = df_action_context_thompson["x1_d1"] 
        df_action_context_thompson["x1_d0.0"] = df_action_context_thompson["x1_d0"]

        # Compute prop. of optimal action choosen (X = 0)
        # Optimal action for X = 0 is D = 1
        count_x0_optimal = [sum([df_action_context_thompson["x0_d1." + str(sim)][user] for sim in range(n_sim)]) for user in range(n_user)]
        count_x0_total = [sum([(df_action_context_thompson["x0_d1." + str(sim)][user] + df_action_context_thompson["x0_d0." + str(sim)][user]) for sim in range(n_sim)]) for user in range(n_user)]
        prop_x0_optimal = [count_x0_optimal[user]/count_x0_total[user] for user in range(n_user)]
        df_action_context_thompson["prop_x0_optimal"] = prop_x0_optimal

        # Compute prop. of optimal action choosen (X = 1)
        # Optimal action for X = 1 is D = 0
        count_x1_optimal = [sum([df_action_context_thompson["x1_d0." + str(sim)][user] for sim in range(n_sim)]) for user in range(n_user)]
        count_x1_total = [sum([(df_action_context_thompson["x1_d1." + str(sim)][user] + df_action_context_thompson["x1_d0." + str(sim)][user]) for sim in range(n_sim)]) for user in range(n_user)]
        prop_x1_optimal = [count_x1_optimal[user]/count_x1_total[user] for user in range(n_user)]
        df_action_context_thompson["prop_x1_optimal"] = prop_x1_optimal

        # Compute overall prop. of optimal action chosen
        prop_optimal = [(count_x0_optimal[user] + count_x1_optimal[user])/(count_x0_total[user] + count_x1_total[user]) for user in range(n_user)]

        # Compute overall optimal for Q1
        prop_optimal_q1 = np.round(np.mean(prop_optimal[0:(n_q1+1)]),2)
        print("[0,250] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q1))

        # Compute overall optimal for Q2
        prop_optimal_q2 = np.round(np.mean(prop_optimal[n_q1:n_q2]),2)
        print("[250,500] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q2))

        # Compute overall optimal for Q3
        prop_optimal_q3 = np.round(np.mean(prop_optimal[n_q2:n_q3]),2)
        print("[500,750] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q3))

        # Compute overall optimal for Q4
        prop_optimal_q4 = np.round(np.mean(prop_optimal[n_q3:(n_user+1)]),2)
        print("[750,1000] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q4))

        # Compute prop. of optimal action choosen (X = 0) for Q1
        prop_x0_optimal_q1 = np.round(np.mean(prop_x0_optimal[0:(n_q1+1)]), 2)
        print("[0,250] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q1))

        # Compute prop. of optimal action choosen (X = 0) for Q2
        prop_x0_optimal_q2 = np.round(np.mean(prop_x0_optimal[n_q1:n_q2]), 2)
        print("[250,500] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q2))

        # Compute prop. of optimal action choosen (X = 0) for Q3
        prop_x0_optimal_q3 = np.round(np.mean(prop_x0_optimal[n_q2:n_q3]), 2)
        print("[500,750] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q3))

        # Compute prop. of optimal action choosen (X = 0) for Q4
        prop_x0_optimal_q4 = np.round(np.mean(prop_x0_optimal[n_q3:(n_user+1)]),2)
        print("[750,1000] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q4))

        # Compute prop. of optimal action choosen (X = 1) for Q1
        prop_x1_optimal_q1 = np.round(np.mean(prop_x1_optimal[0:(n_q1+1)]),2)
        print("[0,250] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q1))

        # Compute prop. of optimal action choosen (X = 1) for Q2
        prop_x1_optimal_q2 = np.round(np.mean(prop_x1_optimal[n_q1:n_q2]),2)
        print("[250,500] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q2))

        # Compute prop. of optimal action choosen (X = 1) for Q3
        prop_x1_optimal_q3 = np.round(np.mean(prop_x1_optimal[n_q2:n_q3]),2)
        print("[500,750] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q3))

        # Compute prop. of optimal action choosen (X = 1) for Q4
        prop_x1_optimal_q4 = np.round(np.mean(prop_x1_optimal[n_q3:(n_user+1)]),2)
        print("[750,1000] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q4))

        # End of a simulation
        print("Simulation results for " +  sim_list[sim_count] + " are finished")

        # Save data frame with column names as place holder variables
        data_list = [d1_bias_ts_ols_q1, d1_bias_ts_ols_q2, d1_bias_ts_ols_q3, d1_bias_ts_ols_q4,
                     d1x1_bias_ts_ols_q1, d1x1_bias_ts_ols_q2, d1x1_bias_ts_ols_q3, d1x1_bias_ts_ols_q4,
                     df_regret_q1, df_regret_q2, df_regret_q3, df_regret_q4, prop_x0_optimal_q1, prop_x0_optimal_q2, prop_x0_optimal_q3,
                     prop_x0_optimal_q4, prop_x1_optimal_q1, prop_x1_optimal_q2, prop_x1_optimal_q3, prop_x1_optimal_q4,
                     d1x1_x1_bias_ts_ols_q1, d1x1_x1_bias_ts_ols_q2, d1x1_x1_bias_ts_ols_q3, d1x1_x1_bias_ts_ols_q4]
        cols = ["d1_bias_ts_ols_q1", "d1_bias_ts_ols_q2", "d1_bias_ts_ols_q3", "d1_bias_ts_ols_q4",
                "d1x1_bias_ts_ols_q1", "d1x1_bias_ts_ols_q2", "d1x1_bias_ts_ols_q3", "d1x1_bias_ts_ols_q4",
                "df_regret_q1", "df_regret_q2", "df_regret_q3", "df_regret_q4", "prop_x0_optimal_q1", "prop_x0_optimal_q2",
                "prop_x0_optimal_q3", "prop_x0_optimal_q4", "prop_x1_optimal_q1", "prop_x1_optimal_q2", "prop_x1_optimal_q3",
                "prop_x1_optimal_q4", "d1x1_x1_bias_ts_ols_q1", "d1x1_x1_bias_ts_ols_q2", "d1x1_x1_bias_ts_ols_q3", "d1x1_x1_bias_ts_ols_q4"]
        cols = [cols[i] + sim_names[sim_count] for i in range(len(cols))]
        results_df = pd.DataFrame([data_list], columns=cols)
        results_df.to_csv("BanditSimResults" + sim_names[sim_count] +".csv")

    # Random policy and OLS fit
    if sim_list[sim_count][5] == "R" and len(sim_names[sim_count]) != 3:

        # None value variables (underspecified and uniform)
        d1x1_bias_rand_ols_q1 = None
        d1x1_bias_rand_ols_q4 = None
        
        # Load d1 random OLS fit
        d1_rand_ols = pd.read_csv("random_ols_d1.csv")
        d1_rand_ols = d1_rand_ols.drop(columns = ["iteration"])

        # Compute mean coeff across simulations
        if sim_names[sim_count][3] != "U":
            d1_bias_rand_ols = d1_rand_ols.mean(axis=1) - true_fit_d1
        else:
            d1_bias_rand_ols = d1_rand_ols.mean(axis=1) - true_fit_d1 - true_fit_d1x1[sim_count]/2

        # Variance of OLS across simulation for each user
        d1_bias_rand_ols_var = d1_rand_ols.var(axis=1)

        # Compute Q1 for d1 random ols fit
        d1_bias_rand_ols_q1 = np.round(np.mean(d1_bias_rand_ols[0:(n_q1+1)]),4)
        print("[0,250] bias(d1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1_bias_rand_ols_q1))

        # Compute standard errors for Q1
        #d1_bias_rand_ols_se_q1 = np.roundnp.std(d1_bias_rand_ols[0:(n_q1+1)].mean(axis=0)
        d1_bias_rand_ols_se_q1 = np.round(np.std(d1_rand_ols[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[0,250] SE(d1) random fit model " + str(d1_bias_rand_ols_se_q1))

        # Compute Q2 for d1 random ols fit
        d1_bias_rand_ols_q2 = np.round(np.mean(d1_bias_rand_ols[n_q1:n_q2]),4)
        print("[250,500] bias(d1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1_bias_rand_ols_q2))

        # Compute standard errors for Q2
        # d1_bias_rand_ols_se_q2 = np.round(np.sqrt(np.sum(d1_bias_rand_ols_var[n_q1:n_q2])/n_sim)/n_q1,5)
        d1_bias_rand_ols_se_q2 = np.round(np.std(d1_rand_ols[n_q1:n_q2].mean(axis=0))/np.sqrt(n_sim),5)
        print("[250,500] SE(d1) random fit model " + str(d1_bias_rand_ols_se_q2))

        # Compute Q3 for d1 random ols fit
        d1_bias_rand_ols_q3 = np.round(np.mean(d1_bias_rand_ols[n_q2:n_q3]),4)
        print("[500,750] bias(d1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1_bias_rand_ols_q3))

        # Compute standard errors for Q3
        # d1_bias_rand_ols_se_q3 = np.round(np.sqrt(np.sum(d1_bias_rand_ols_var[n_q2:n_q3])/n_sim)/n_q1,5)
        d1_bias_rand_ols_se_q3 = np.round(np.std(d1_rand_ols[n_q2:n_q3].mean(axis=0))/np.sqrt(n_sim),5)
        print("[500,750] SE(d1) random fit model " + str(d1_bias_rand_ols_se_q3))
                                       
        # Compute Q4 for d1 random ols fit
        d1_bias_rand_ols_q4 = np.round(np.mean(d1_bias_rand_ols[n_q3:(n_user+1)]),4)
        print("[750,1000] bias(d1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1_bias_rand_ols_q4))

        # Compute standard errors for Q4
        d1_bias_rand_ols_se_q4 = np.round(np.std(d1_rand_ols[n_q3:(n_user+1)].mean(axis=0))/np.sqrt(n_sim),5)
        print("[750,1000] SE(d1) random fit model " + str(d1_bias_rand_ols_se_q4))

        if sim_names[sim_count][3] != "U":
            
            # Load d1 random OLS fit
            d1x1_rand_ols = pd.read_csv("random_ols_d1x1.csv")
            d1x1_rand_ols = d1x1_rand_ols.drop(columns = ["iteration"])

            # Compute mean coeff across simulations
            d1x1_bias_rand_ols = d1x1_rand_ols.mean(axis=1) - true_fit_d1x1[sim_count]

            # Compute Q1 for d1x1 random ols fit
            d1x1_bias_rand_ols_q1 = np.round(np.mean(d1x1_bias_rand_ols[0:(n_q1+1)]),4)
            print("[0,250] bias(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_q1))

            # Compute Q1 for SE(d1x1) random ols fit
            d1x1_bias_rand_ols_se_q1 = np.round(np.std(d1x1_rand_ols[0:(n_q1+1)].mean(axis=0))/np.sqrt(n_sim),5)
            print("[0,250] SE(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_se_q1))

            # Compute Q2 for d1x1 random ols fit
            d1x1_bias_rand_ols_q2 = np.round(np.mean(d1x1_bias_rand_ols[n_q1:n_q2]),4)
            print("[250,500] bias(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_q2))

            # Compute Q2 for SE(d1x1) random ols fit
            d1x1_bias_rand_ols_se_q2 = np.round(np.std(d1x1_rand_ols[n_q1:n_q2].mean(axis=0))/np.sqrt(n_sim),5)
            print("[250,500] SE(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_se_q2))

            # Compute Q3 for d1x1 random ols fit
            d1x1_bias_rand_ols_q3 = np.round(np.mean(d1x1_bias_rand_ols[n_q2:n_q3]),4)
            print("[500,750] bias(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_q3))

            # Compute Q3 for SE(d1x1) random ols fit
            d1x1_bias_rand_ols_se_q3 = np.round(np.std(d1x1_rand_ols[n_q2:n_q3].mean(axis=0))/np.sqrt(n_sim),5)
            print("[500,750] SE(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_se_q3))

            # Compute Q4 for d1x1 random ols fit
            d1x1_bias_rand_ols_q4 = np.round(np.mean(d1x1_bias_rand_ols[n_q3:(n_user+1)]),4)
            print("[750,1000] bias(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_q4))

            # Compute Q4 for SE(d1x1) random ols fit
            d1x1_bias_rand_ols_se_q4 = np.round(np.std(d1x1_rand_ols[n_q3:(n_user+1)].mean(axis=0))/np.sqrt(n_sim),5)
            print("[750,1000] SE(d1x1) random fit model " + sim_names[sim_count][1:4] + " "+ str(d1x1_bias_rand_ols_se_q4))

            # End of a simulation
            print("Simulation results for " +  sim_list[sim_count] + " are finished")

            # Save data frame user column names for place holder variables
            data_list = [d1_bias_rand_ols_q1, d1_bias_rand_ols_q2, d1_bias_rand_ols_q3, d1_bias_rand_ols_q4,
                         d1x1_bias_rand_ols_q1, d1x1_bias_rand_ols_q2, d1x1_bias_rand_ols_q3, d1x1_bias_rand_ols_q4]
            cols = ["d1_bias_rand_ols_q1", "d1_bias_rand_ols_q2", "d1_bias_rand_ols_q3", "d1_bias_rand_ols_q4",
                    "d1x1_bias_rand_ols_q1", "d1x1_bias_rand_ols_q2", "d1x1_bias_rand_ols_q3", "d1x1_bias_rand_ols_q4"]
            cols = [cols[i] + sim_names[sim_count] for i in range(len(cols))]

        results_df = pd.DataFrame([data_list], columns=cols)
        results_df.to_csv("BanditSimResults" + sim_names[sim_count] +".csv")

    # Uniform sampling and no OLS fit
    if len(sim_names[sim_count]) == 3:

        # Load regret data
        df_regret_thompson = pd.read_csv('random_regrets.csv')
        df_regret_thompson = df_regret_thompson.drop(columns = ["iteration"])

        # Compute mean regret (across simulation) for each itteration
        df_regret = df_regret_thompson.mean(axis=1)

        # Compute regret for Q1
        df_regret_q1 = np.round_(np.mean(df_regret[0:(n_q1+1)]),2)
        print("[0,250] regret UP " + sim_names[sim_count][1:4] + " "+ str(df_regret_q1))

        # Compute regret for Q2
        df_regret_q2 = np.round_(np.mean(df_regret[n_q1:n_q2]),2)
        print("[250,500] regret UP " + sim_names[sim_count][1:4] + " "+ str(df_regret_q2))

        # Compute regret for Q3
        df_regret_q3 = np.round_(np.mean(df_regret[n_q2:n_q3]),2)
        print("[500,750] regret UP " + sim_names[sim_count][1:4] + " "+ str(df_regret_q3))

        # Compute regret for Q4
        df_regret_q4 = np.round(np.mean(df_regret[n_q3:(n_user+1)]),2)
        print("[750,1000] regret UP "+ sim_names[sim_count][1:4] + " "+ str(df_regret_q4))

        # Load action choosen given context data
        df_action_context_thompson = pd.read_csv("random_context_action.csv",skiprows=1)
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

        # Compute overall prop. of optimal action chosen
        prop_optimal = [(count_x0_optimal[user] + count_x1_optimal[user])/(count_x0_total[user] + count_x1_total[user]) for user in range(n_user)]

        # Compute overall optimal for Q1
        prop_optimal_q1 = np.round(np.mean(prop_optimal[0:(n_q1+1)]),2)
        print("[0,250] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q1))

        # Compute overall optimal for Q2
        prop_optimal_q2 = np.round(np.mean(prop_optimal[n_q1:n_q2]),2)
        print("[250,500] prop. optimal overall " + sim_names[sim_count][1:4] + " " + str(prop_optimal_q2))

        # Compute prop. of optimal action choosen (X = 0) for Q1
        prop_x0_optimal_q1 = np.round(np.mean(prop_x0_optimal[0:(n_q1+1)]), 2)
        print("[0,250] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q1))

        # Compute prop. of optimal action choosen (X = 0) for Q4
        prop_x0_optimal_q4 = np.round(np.mean(prop_x0_optimal[n_q3:(n_user+1)]),2)
        print("[750,1000] prop. optimal (X = 0) " + sim_names[sim_count][1:4] + " "+ str(prop_x0_optimal_q4))

        # Compute prop. of optimal action choosen (X = 1) for Q1
        prop_x1_optimal_q1 = np.round(np.mean(prop_x1_optimal[0:(n_q1+1)]),2)
        print("[0,250] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q1))

        # Compute prop. of optimal action choosen (X = 1) for Q4
        prop_x1_optimal_q4 = np.round(np.mean(prop_x1_optimal[n_q3:(n_user+1)]),2)
        print("[750,1000] prop. optimal (X = 1) " + sim_names[sim_count][1:4] + " "+ str(prop_x1_optimal_q4))

        # Save uniform policy data
        data_list = [prop_x0_optimal_q1, prop_x0_optimal_q2, prop_x0_optimal_q3,  prop_x0_optimal_q4, prop_x1_optimal_q1,
                     prop_x1_optimal_q2, prop_x1_optimal_q3, prop_x1_optimal_q4,
                     df_regret_q1, df_regret_q2, df_regret_q3, df_regret_q4]
        cols = ["prop_x0_optimal_q1", "prop_x0_optimal_q4", "prop_x1_optimal_q1", "prop_x1_optimal_q4",
                "df_regret_q1", "df_regret_q4"]
        cols = [cols[i] + sim_names[sim_count] for i in range(len(cols))]
        results_df = pd.DataFrame([data_list], columns=cols)
        results_df.to_csv("BanditSimResults" + sim_names[sim_count] +".csv")

        # End of a simulation
        print("Simulation results for " +  sim_list[sim_count] + " are finished")

    # Increase simulation counter
    sim_count += 1


'''
    ##### Code for MAB posterior mean bias
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

'''
