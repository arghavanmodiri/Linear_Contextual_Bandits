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
    colors = ['indigo', 'darkmagenta', 'palevioletred',
                'black', 'dimgray', 'lightgray',
                'darkred', 'red', 'salmon',
                'darkolivegreen', 'yellowgreen', 'palegreen']
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

    plt.legend(fontsize = 9, loc='upper center', bbox_to_anchor=(0.5, -0.06),
          fancybox=True, shadow=True, ncol=4)
    plt.grid()
    plt.ylabel('Proportion of Optimal Action Assignment', fontsize = 12)
    plt.title('Proportion of Optimal Action at Each Itteration\n '
                '(Simulations = {sims}, Batch Size={batches})'.format(sims=
                simulation_count, batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_optimal_action_ratio.png'.format(date=TODAY,
                    i=user_count, sim=simulation_count))

def plot_regret(user_count, policy_names, regrets_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    plt.grid()
    #plt.xlim(0, user_count)
    plt.xlim(0, user_count)
    #plt.ylim(0, 100 )
    #plt.set_ylim(0, 300)
    colors = ['indigo', 'darkmagenta', 'palevioletred',
                'black', 'dimgray', 'lightgray',
                'darkred', 'red', 'salmon',
                'darkolivegreen', 'yellowgreen', 'palegreen']
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")
    #colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, np.cumsum(regrets_all_policies[idx]),
                label = policy, color = colors[idx])#, color=colors[idx])


    plt.legend(fontsize = 9, loc='upper center', bbox_to_anchor=(0.5, -0.06),
          fancybox=True, shadow=True, ncol=4)
    #plt.xlabel('User Iterations', fontsize = 12)
    plt.xlabel('User Iterations', fontsize = 18)
    #plt.ylabel('Regret', fontsize = 12)
    plt.ylabel('Cumulative Regret', fontsize = 18)
    #plt.title('Calculated Regret for Each User\n(Simulations = {sims}, Batch '
    #            'Size={batches})'.format(sims=simulation_count,
    #            batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_regrets.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))



def main(mode=None):
    base_folder = "C:/Users/modir/OneDrive/Documents/simulations_results/"
    model1_hypo1 = pd.read_csv("{}Weekdays_TrueModel1_Hypo1,2,3/optimal_action_ratio_thompson_1-everything-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo2 = pd.read_csv("{}Weekdays_TrueModel1_Hypo1,2,3/optimal_action_ratio_thompson_2-everything-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo3 = pd.read_csv("{}Weekdays_TrueModel1_Hypo1,2,3/optimal_action_ratio_thompson_3-everything-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo4 = pd.read_csv("{}Weekdays_TrueModel1_Hypo4,5,6/optimal_action_ratio_thompson_4-no-context-interaction-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo5 = pd.read_csv("{}Weekdays_TrueModel1_Hypo4,5,6/optimal_action_ratio_thompson_5-no-context-interaction-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo6 = pd.read_csv("{}Weekdays_TrueModel1_Hypo4,5,6/optimal_action_ratio_thompson_6-no-context-interaction-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo7 = pd.read_csv("{}Weekdays_TrueModel1_Hypo7,8,9/optimal_action_ratio_thompson_7-only-day-context-no-context-interaction-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo8 = pd.read_csv("{}Weekdays_TrueModel1_Hypo7,8,9/optimal_action_ratio_thompson_8-only-day-context-no-context-interaction-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo9 = pd.read_csv("{}Weekdays_TrueModel1_Hypo7,8,9/optimal_action_ratio_thompson_9-only-day-context-no-context-interaction-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo10 = pd.read_csv("{}Weekdays_TrueModel1_Hypo10,11,12/optimal_action_ratio_thompson_10-only-day-context-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo11 = pd.read_csv("{}Weekdays_TrueModel1_Hypo10,11,12/optimal_action_ratio_thompson_11-only-day-context-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo12 = pd.read_csv("{}Weekdays_TrueModel1_Hypo10,11,12/optimal_action_ratio_thompson_12-only-day-context-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo1 = model1_hypo1.mean(axis=1)
    model1_hypo2 = model1_hypo2.mean(axis=1)
    model1_hypo3 = model1_hypo3.mean(axis=1)
    model1_hypo4 = model1_hypo4.mean(axis=1)
    model1_hypo5 = model1_hypo5.mean(axis=1)
    model1_hypo6 = model1_hypo6.mean(axis=1)
    model1_hypo7 = model1_hypo7.mean(axis=1)
    model1_hypo8 = model1_hypo8.mean(axis=1)
    model1_hypo9 = model1_hypo9.mean(axis=1)
    model1_hypo10 = model1_hypo10.mean(axis=1)
    model1_hypo11 = model1_hypo11.mean(axis=1)
    model1_hypo12 = model1_hypo12.mean(axis=1)
    
    data = [model1_hypo1,
            model1_hypo2,
            model1_hypo3,
            model1_hypo4,
            model1_hypo5,
            model1_hypo6,
            model1_hypo7,
            model1_hypo8,
            model1_hypo9,
            model1_hypo10,
            model1_hypo11,
            model1_hypo12]

    policies = ["1-everything-daily",
            "2-everything-is-weekend",
            "3-everything-both",
            "4-no-context-interaction-daily",
            "5-no-context-interaction-is-weekend",
            "6-no-context-interaction-both",
            "7-only-day-context-no-context-interaction-daily",
            "8-only-day-context-no-context-interaction-is-weekend",
            "9-only-day-context-no-context-interaction-both",
            "10-only-day-context-daily",
            "11-only-day-context-is-weekend",
            "12-only-day-context-both"]

    #plot_optimal_action_ratio(2400, policies, data, 100, 30, mode='per_user')

    ##############################################################
    model1_hypo1_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo1,2,3/regrets_thompson_1-everything-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo2_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo1,2,3/regrets_thompson_2-everything-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo3_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo1,2,3/regrets_thompson_3-everything-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo4_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo4,5,6/regrets_thompson_4-no-context-interaction-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo5_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo4,5,6/regrets_thompson_5-no-context-interaction-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo6_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo4,5,6/regrets_thompson_6-no-context-interaction-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo7_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo7,8,9/regrets_thompson_7-only-day-context-no-context-interaction-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo8_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo7,8,9/regrets_thompson_8-only-day-context-no-context-interaction-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo9_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo7,8,9/regrets_thompson_9-only-day-context-no-context-interaction-daily-and-is-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo10_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo10,11,12/regrets_thompson_10-only-day-context-daily.csv".format(base_folder),index_col="iteration")
    model1_hypo11_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo10,11,12/regrets_thompson_11-only-day-context-is-weekend.csv".format(base_folder),index_col="iteration")
    model1_hypo12_regret = pd.read_csv("{}Weekdays_Truemodel1_Hypo10,11,12/regrets_thompson_12-only-day-context-weekend.csv".format(base_folder),index_col="iteration")

    model1_hypo1_regret = model1_hypo1_regret.mean(axis=1)
    model1_hypo2_regret = model1_hypo2_regret.mean(axis=1)
    model1_hypo3_regret = model1_hypo3_regret.mean(axis=1)
    model1_hypo4_regret = model1_hypo4_regret.mean(axis=1)
    model1_hypo5_regret = model1_hypo5_regret.mean(axis=1)
    model1_hypo6_regret = model1_hypo6_regret.mean(axis=1)
    model1_hypo7_regret = model1_hypo7_regret.mean(axis=1)
    model1_hypo8_regret = model1_hypo8_regret.mean(axis=1)
    model1_hypo9_regret = model1_hypo9_regret.mean(axis=1)
    model1_hypo10_regret = model1_hypo10_regret.mean(axis=1)
    model1_hypo11_regret = model1_hypo11_regret.mean(axis=1)
    model1_hypo12_regret = model1_hypo12_regret.mean(axis=1)
    
    data_regret = [model1_hypo1_regret,
            model1_hypo2_regret,
            model1_hypo3_regret,
            model1_hypo4_regret,
            model1_hypo5_regret,
            model1_hypo6_regret,
            model1_hypo7_regret,
            model1_hypo8_regret,
            model1_hypo9_regret,
            model1_hypo10_regret,
            model1_hypo11_regret,
            model1_hypo12_regret]

    plot_optimal_action_ratio(2400, policies, data, 100, 30, mode='per_user')
    plot_regret(2400, policies, data_regret, 100,30)
    plt.show()


if __name__ == "__main__":
    print("hi")
    main()
    