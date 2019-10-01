import os
import numpy as np
import pandas as pd
import true_hypo_models as models
import making_decision
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nprnd

# Simulations
#sim_list = ["SimCCorrectMABFitCorrect", "SimCUnderMABFitCorrect", "SimCRandomCorrect",
            #"SimDCorrectMABFitCorrect", "SimDUnderMABFitCorrect", "SimDRandomCorrect"]
#sim_names = ["_CSCC", "_CSUC", "_CSRC", "_DLCC", "_DLUC", "_DLRC"]
sim_list = ["SimDCorrectMABFitCorrectRep15"]
sim_names = ["_DLCCR15"]
#sim_names = ["_CSCCR15"]
sim_count = 0

# Itterate over simulation
for simulation in sim_names:

    # Change directory
    os.chdir('/Users/hammadshaikh/linear_contextual_bandits/saved_output/' + sim_list[sim_count])

    # Bandit
    if simulation[3] != "R":
        # Load OLS fitted coefficients data
        intercept_csv = pd.read_csv("thompson_ols_intercept.csv")
        #intercept_csv = intercept_csv.drop(columns = ["iteration"])
        d1_csv = pd.read_csv("thompson_ols_d1.csv")
        #d1_csv = d1_csv.drop(columns = ["iteration"])
        d1x1_csv = pd.read_csv("thompson_ols_d1x1.csv")
        #d1x1_csv = d1x1_csv.drop(columns = ["iteration"])
    # RCT
    else:
        # Load OLS fitted coefficients data
        intercept_csv = pd.read_csv("random_ols_intercept.csv")
        d1_csv = pd.read_csv("random_ols_d1.csv")
        d1x1_csv = pd.read_csv("random_ols_d1x1.csv")

    # Number of simulation
    n_sim = d1_csv.shape[1]-1

    # Quarters of users
    n_q1 = 249
    n_q4 = 999

    # Experimental variables
    true_model_params = models.read_true_model()
    bandit_arms = models.find_possible_actions()
    experiment_vars = true_model_params['experiment_vars']

    # Keep track of optimal action choosen
    ols_action_optimal_250_sim = []
    ols_action_optimal_1000_sim = []

    # Itterate over all simulations
    for sim in range(n_sim):

        # Retrieve OLS coefficient at itteration 250
        intercept_250 = intercept_csv.iloc[n_q1][str(sim)]
        d1_250 = d1_csv.iloc[n_q1][str(sim)]
        d1x1_250 = d1x1_csv.iloc[n_q1][str(sim)]

        # Retrieve OLS coefficient at itteration 1000
        intercept_1000 = intercept_csv.iloc[n_q4][str(sim)]
        d1_1000 = d1_csv.iloc[n_q4][str(sim)]
        d1x1_1000 = d1x1_csv.iloc[n_q4][str(sim)]

        # Compute OLS implies policy
        # Dictonary of OLS coefficients
        ols_coeff_dict_250 = {"intercept": intercept_250, "d1":d1_250, "d1*x1":d1x1_250}
        ols_coeff_dict_1000 = {"intercept": intercept_1000, "d1":d1_1000, "d1*x1":d1x1_1000}

        # Contextual values
        context_value0 = {"x1": 0}
        context_value1 = {"x1": 1}

        # Compute optimal arm for X1 = 0
        ols_optimal_x0_250 = making_decision.pick_true_optimal_arm(ols_coeff_dict_250, context_value0, experiment_vars,bandit_arms)[0][0][0]
        ols_optimal_x0_1000 = making_decision.pick_true_optimal_arm(ols_coeff_dict_1000, context_value0, experiment_vars,bandit_arms)[0][0][0]

        # Compute optimal arm for X1 = 1
        ols_optimal_x1_250 = making_decision.pick_true_optimal_arm(ols_coeff_dict_250, context_value1, experiment_vars,bandit_arms)[0][0][0]
        ols_optimal_x1_1000 = making_decision.pick_true_optimal_arm(ols_coeff_dict_1000, context_value1, experiment_vars,bandit_arms)[0][0][0]

        # Determine whether OLS picked optimal action
        ols_action_optimal_250 = (ols_optimal_x0_250 == 1 and ols_optimal_x1_250 == 0)
        ols_action_optimal_1000 = (ols_optimal_x0_1000 == 1 and ols_optimal_x1_1000 == 0)

        # Simulations in which optimal policy was choosen by OLS
        ols_action_optimal_250_sim.append(ols_action_optimal_250)
        ols_action_optimal_1000_sim.append(ols_action_optimal_1000)

    # Proportion of simulations in which OLS implies optimal policy
    prop_ols_optimal_250 = np.mean(ols_action_optimal_250_sim)
    prop_ols_optimal_1000 = np.mean(ols_action_optimal_1000_sim)

    # Save data
    data_list = [prop_ols_optimal_250, prop_ols_optimal_1000]
    cols = ["prop_ols_optimal_q1", "prop_ols_optimal_q4"]
    cols = [cols[i] + sim_names[sim_count] for i in range(len(cols))]
    results_df = pd.DataFrame([data_list], columns=cols)
    results_df.to_csv("PropOLSOptimal" + sim_names[sim_count] +".csv")

    # Increment
    sim_count += 1

    print("Proportion OLS selects optimal policy at itteration 250 " + str(simulation) + " " + str(np.mean(ols_action_optimal_250_sim)))
    print("Proportion OLS selects optimal policy at itteration 1000 " + str(simulation) + " " + str(np.mean(ols_action_optimal_1000_sim)))
        
