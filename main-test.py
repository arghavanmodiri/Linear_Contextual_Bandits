from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import logging
import pandas as pd


TODAY = date.today()

def plot_optimal_action_ratio(user_count, policy_names,
                                optimal_action_ratio_all_policies,
                                simulation_count, batch_size, mode='per_user',
                                save_fig=True):
    plt.figure()
    colors = ['teal', 'indigo', 'palevioletred']
    if(mode == 'per_user'):
        UserItter = range(1,user_count+1)
        #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")
        for idx, policy in enumerate(policy_names):
            plt.plot(UserItter, optimal_action_ratio_all_policies[idx], label =
                policy, color = colors[idx])
            mean_ratio = sum(optimal_action_ratio_all_policies[idx])/len(
                            optimal_action_ratio_all_policies[idx])
            #plt.plot(UserItter, [mean_ratio]*user_count,label =
            #    "mean of ratios for {} ".format(policy), linestyle=':')
        plt.xlabel('User Iterations', fontsize = 18)
    elif(mode == 'per_batch'):
        batchItter = range(1,int(user_count/batch_size)+1)
        #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")
        for idx, policy in enumerate(policy_names):
            optimal_action_ratio_batch = list(mean(
                optimal_action_ratio_all_policies[idx][i:i+batch_size])
                for i in range(0,
                len(optimal_action_ratio_all_policies[idx]),
                batch_size))
            plt.plot(batchItter, optimal_action_ratio_batch, label = policy)
        plt.xlabel('Batch Iterations', fontsize = 18)

    plt.legend(fontsize = 12, loc='upper center', bbox_to_anchor=(0.5, -0.07),
          fancybox=True, shadow=True, ncol=5)
    plt.grid()
    plt.ylabel('Proportion of Optimal Action Assignment', fontsize = 12)
    plt.title('Proportion of Optimal Action at Each Itteration\n '
                '(Simulations = {sims}, Batch Size={batches})'.format(sims=
                simulation_count, batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_optimal_action_ratio.png'.format(date=TODAY,
                    i=user_count, sim=simulation_count))

def main(mode=None):
    misspecified_model = pd.read_csv("saved_output/raw_data_2019-11-15_17-9-6/optimal_action_ratio_thompson_all main effect of action var.csv",index_col="iteration")
    correctly_specified_model = pd.read_csv("saved_output/raw_data_2019-11-15_17-9-6/optimal_action_ratio_thompson_correctly specified model.csv",index_col="iteration") 
    over_specified_model = pd.read_csv("saved_output/raw_data_2019-11-15_17-9-6/optimal_action_ratio_thompson_main effects of context vars, action vars & action-action, contex-action interaction.csv",index_col="iteration") 

    misspecified_model = misspecified_model.mean(axis=1)
    correctly_specified_model = correctly_specified_model.mean(axis=1)
    over_specified_model = over_specified_model.mean(axis=1)

    data = [correctly_specified_model, misspecified_model, over_specified_model]
    policies = ["Correct Model", "Model with Left-out Variables", "Model with Irrelevant Variables"]

    plot_optimal_action_ratio(3000, policies, data, 300, 10, mode='per_user')
    plt.show()

if __name__ == "__main__":
    print("hi")
    main()
    