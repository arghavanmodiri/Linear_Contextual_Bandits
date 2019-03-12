from datetime import date
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean


TODAY = date.today()

def plot_regret(user_count, policy_names, regrets_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, np.cumsum(regrets_all_policies[idx]),
                label = policy)


    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Regret', fontsize = 12)
    plt.title('Calculated Regret for Each User\n(Simulations = {sims}, Batch '
                'Size={batches})'.format(sims=simulation_count,
                batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_regrets.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))



def plot_regression(user_count, regression_params_dict, regression_params_std_dict, true_coeffs,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    plt.gca()
    plt.grid()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    colors = ['b', 'g', 'c', 'm', 'y', 'k']
    color_idx = 0
    for key,value in regression_params_dict.items():
        #plt.plot(UserItter, value, label = key)
        mean = sum(value)/len(value)
        #y_min = [value_i - std_i for value_i,std_i in zip(value,regression_stderr_dict[key])]
        #y_max = [value_i + std_i for value_i,std_i in zip(value,regression_stderr_dict[key])]
        #y_max = [value_i + regression_stderr_dict[key] for value_i in value]
        y_min = value - regression_params_std_dict[key]
        y_max = value + regression_params_std_dict[key]
        if key == 'd1x1':
            true_value = [true_coeffs['d1*x1']]*user_count
        else:
            true_value = [true_coeffs[key]]*user_count
        plt.plot(UserItter, value, label ="OLS fit - Coeff. of {}".format(key), color=colors[color_idx])
        plt.plot(UserItter, true_value, label ="True Model - Coeff. of {}".format(key), color=colors[color_idx],alpha=0.8,  linestyle=':')
        plt.fill_between(UserItter, y_max, y_min,color=colors[color_idx],alpha=0.3)
        color_idx += 1

    plt.legend(loc='upper right', fontsize = 10, prop={'size': 8})
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Regression Params', fontsize = 12)
    plt.title('OLS Regression Params - Thompson Sampling\n(Simulations = {sims}, '
                'Batch Size={batches})'.format(sims=simulation_count,
                batches=batch_size), fontsize = 12)

    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_regression_params.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))


def plot_optimal_action_ratio(user_count, policy_names,
                                optimal_action_ratio_all_policies,
                                simulation_count, batch_size, mode='per_user',
                                save_fig=True):
    plt.figure()

    if(mode == 'per_user'):
        UserItter = range(1,user_count+1)
        #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")
        for idx, policy in enumerate(policy_names):
            plt.plot(UserItter, optimal_action_ratio_all_policies[idx], label =
                policy)
            mean_ratio = sum(optimal_action_ratio_all_policies[idx])/len(
                            optimal_action_ratio_all_policies[idx])
            plt.plot(UserItter, [mean_ratio]*user_count,label =
                "mean of ratios for {} ".format(policy), linestyle=':')
        plt.xlabel('User Itterations', fontsize = 18)
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
        plt.xlabel('Batch Itterations', fontsize = 18)

    plt.legend(loc='upper left', fontsize = 12)
    plt.ylabel('Proportion of Optimal Action Assignment', fontsize = 12)
    plt.title('Proportion of Optimal Action at Each Itteration\n '
                '(Simulations = {sims}, Batch Size={batches})'.format(sims=
                simulation_count, batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_optimal_action_ratio.png'.format(date=TODAY,
                    i=user_count, sim=simulation_count))


def plot_suboptimal_action_ratio(user_count, policy_names,
                                optimal_action_ratio_all_policies,
                                simulation_count, batch_size, mode='per_user',
                                save_fig=True):
    plt.figure()

    if(mode == 'per_user'):
        UserItter = range(1,user_count+1)
        #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")
        for idx, policy in enumerate(policy_names):
            plt.plot(UserItter, optimal_action_ratio_all_policies[idx], label =
                policy)
        plt.xlabel('User Itterations', fontsize = 18)
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
        plt.xlabel('Batch Itterations', fontsize = 18)

    plt.legend(loc='upper left', fontsize = 12)
    plt.ylabel('Proportion of Sub-optimal Action Assignment', fontsize = 12)
    plt.title('Proportion of Sub-optimal Action at Each Itteration per User Context\n '
                '(Simulations = {sims}, Batch Size={batches})'.format(sims=
                simulation_count, batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_suboptimal_action_ratio.png'.format(date=TODAY,
                    i=user_count, sim=simulation_count))


def plot_mse(user_count, policy_names, mse_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, mse_all_policies[idx], label = policy)

    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('MSE', fontsize = 12)
    plt.title('MSE at Each Itteration\n(Simulations = {sims}, '
                'Batch Size={batches})'.format(sims=simulation_count,
                batches=batch_size), fontsize = 12)

    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_mse.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))



def plot_coeff_ranking(user_count, policy_name, beta_thompson_coeffs,
                    hypo_params, simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)

    for param in range(0,len(hypo_params)):
        plt.plot(UserItter, beta_thompson_coeffs[:,param],
                label = hypo_params[param])

    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Hypo Coeff', fontsize = 12)
    plt.title('Coeff at Each Itteration for {policy}\n(Simulations = {sims},'
                'Batch Size={batches})'.format(policy=policy_name, sims=
                simulation_count, batches=batch_size),fontsize = 12)

    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_coeff_ranking.png'.format(date=TODAY, i=user_count,
                    sim=simulation_count))


def plot_coeff_sign_error(user_count, policy_name, hypo_params,
                        coeff_sign_error, simulation_count, batch_size,
                        save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for param in range(0,len(hypo_params)):
        plt.plot(UserItter, coeff_sign_error[:,param], label = hypo_params[
                param])

    plt.legend(loc='upper right', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Coeff Sign Error', fontsize = 12)
    plt.title('Coeff Sign Error at Each Itteration for {policy}\n '
                '(Simulations= {sims}, Batch Size={batches})'.
                format(policy=policy_name,sims= simulation_count,
                batches=batch_size),fontsize = 12)
    #plt.show()
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_coeff_sign_err.png'.format(date=TODAY, i=user_count,
                    sim=simulation_count))



def plot_bias_in_coeff(user_count, policy_name, hypo_params,
                        bias_in_coeff, simulation_count, batch_size,
                        save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for param in range(0,len(hypo_params)):
        plt.plot(UserItter, bias_in_coeff[:,param], label = hypo_params[
                param])

    plt.legend(loc='upper right', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Bias in Coeff', fontsize = 12)
    plt.title('Bias in Coeff at Each Itteration for {policy}\n '
                '(Simulations= {sims}, Batch Size={batches})'.
                format(policy=policy_name,sims= simulation_count,
                batches=batch_size),fontsize = 12)
    #plt.show()
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim} '
                    'sims_bias_in_coeff.png'.format(date=TODAY, i=user_count,
                    sim=simulation_count))
