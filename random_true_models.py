#!/usr/bin/env python
import os
import sys
import json
import multiprocessing
import argparse
import logging
import numpy as np
import pandas as pd
import simulator
import policies.thompson_sampling_nig as thompson
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from shutil import copy2
from datetime import date
from datetime import datetime
from training_models import training_bandit_model as tbm

TODAY = date.today()
NOW = datetime.now()


pd.set_option('display.max_columns', 30)
def simulatingDiffModels(pool, input_dict, sim_name='test.json', mode=None):
    """start the model"""
    '''
    The simulator will start by passing the json file. The number of users to
    be simulated needs to be passed with "user_count" variable.
    '''

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s')

    ## Setting the true model parameters and the enviromental parameters
    #coeffs = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    true_coeff = dict(zip(true_model_terms, input_dict['true_coeff']))


    save_optimal_action_ratio_thompson_df = pd.DataFrame()
    save_mse_thompson_df = pd.DataFrame()
    save_coeff_sign_err_thompson_df = pd.DataFrame()
    save_bias_in_coeff_thompson_df = pd.DataFrame()
    save_regret_thompson_df = pd.DataFrame()
    save_suboptimal_action_ratio_group_thompson_df = pd.DataFrame()
    save_optimal_action_ratio_random_df = pd.DataFrame()
    save_mse_random_df = pd.DataFrame()
    save_coeff_sign_err_random_df = pd.DataFrame()
    save_bias_in_coeff_random_df = pd.DataFrame()
    save_regret_random_df = pd.DataFrame()
    save_suboptimal_action_ratio_group_random_df = pd.DataFrame()
    bandit_models = []
    for idx in range(len(hypo_params_all_models)):
        bandit_models.append(
                    tbm(pool, user_count,
                    batch_size, experiment_vars, bandit_arms, true_coeff,
                    extensive, hypo_params_all_models[idx]))
    if rand_sampling_applied==True:
        random_model = tbm(pool, user_count,
                    batch_size, experiment_vars, bandit_arms, true_coeff,
                    extensive)

    if input_context_type == 'simulated':
        users_context = simulator.generate_true_dataset(context_vars,
            user_count, input_dict['context_dist'], batch_size)
    else:
        users_context = data


    for sim in range(0, simulation_count):
        logging.info("{}, sim: {}   - Time: {}".format(sim_name, sim,
                datetime.now()))
        a_pre = input_dict['NIG_priors']['a']
        b_pre = input_dict['NIG_priors']['b']
        #Hammad: Bias Correction
        #mean_pre = np.zeros(len(hypo_params))
        #cov_pre = np.identity(len(hypo_params))

        #Step 3: Calls the sampling policy to select action for each user
        for idx in range(len(hypo_params_all_models)):
            bandit_models[idx].apply_thompson(users_context, a_pre, b_pre, noise_stats)

        if rand_sampling_applied==True:
            random_model.apply_random(users_context, noise_stats)

        #regret
        #optimal_action_ratio
        #beta_thompson_coeffs
        #mse
        #coeff_sign_error_per
        #Bias Correction

        # Add Random Policy

    all_hypo_params_all_models = true_coeff
    regrets_all_users = []
    regrets_std_all_users = []

    regression_params_avg_quarters = []
    regression_params_std_quarters = []

    regrets_avg_quarters = []
    regrets_std_avg_quarters = []
    optimal_action_ratio = []
    optimal_action_ratio_avg_quarters = []
    beta_thompson_coeffs = []
    
    for idx in range(len(bandit_models)):
        all_hypo_params_all_models = set().union(all_hypo_params_all_models, bandit_models[idx].hypo_params)
        regrets_all_users.append(bandit_models[idx].get_regret_average())
        regrets_std_all_users.append(bandit_models[idx].get_regret_std())
        users_quarter = int(len(regrets_all_users[idx]) / 4)
        regrets_avg_quarters.append([round(sum(regrets_all_users[idx][:users_quarter]) / users_quarter, 2) ,
                                    round(sum(regrets_all_users[idx][users_quarter:2*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_all_users[idx][2*users_quarter:3*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_all_users[idx][3*users_quarter:4*users_quarter]) / users_quarter, 2)])
        regrets_std_avg_quarters.append([round(sum(regrets_std_all_users[idx][:users_quarter]) / users_quarter, 2) ,
                                    round(sum(regrets_std_all_users[idx][users_quarter:2*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_std_all_users[idx][2*users_quarter:3*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_std_all_users[idx][3*users_quarter:4*users_quarter]) / users_quarter, 2)])

        optimal_action_ratio.append(bandit_models[idx].get_optimal_action_ratio_average())
        optimal_action_ratio_avg_quarters.append([round(sum(optimal_action_ratio[idx][:users_quarter]) / users_quarter, 2) ,
                                    round(sum(optimal_action_ratio[idx][users_quarter:2*users_quarter]) / users_quarter, 2),
                                    round(sum(optimal_action_ratio[idx][2*users_quarter:3*users_quarter]) / users_quarter, 2),
                                    round(sum(optimal_action_ratio[idx][3*users_quarter:4*users_quarter]) / users_quarter, 2)])
    
        #beta_thompson_coeffs: lists of dataframes!
        beta_thompson_coeffs.append(
            bandit_models[idx].get_beta_thompson_coeffs_average())

    regrets_avg = [round(sum(el)/len(el), 2) for el in regrets_avg_quarters]
    regrets_std = [round(sum(el)/len(el), 2) for el in regrets_std_avg_quarters]
    optimal_avg = [round(sum(el)/len(el), 2) for el in optimal_action_ratio_avg_quarters]

    sorted_regret = ''
    policy_regret_dict = dict(zip(policies, regrets_avg))
    for k in sorted( ((v,k) for k,v in policy_regret_dict.items()), reverse=True):
        sorted_regret += k[1]+' '

    sorted_regret += '      '
    for k in sorted( ((v,k) for k,v in policy_regret_dict.items()), reverse=True):
        sorted_regret += str(k[0])+' '

    result_rank.write("{}\n".format(sorted_regret))
    result_coeffs.write("{}\n".format(input_dict['true_coeff']))
    result_quarter.write("{}    -   {}  -   {}\n".format(regrets_avg_quarters, regrets_std_avg_quarters, optimal_action_ratio_avg_quarters))
    result_avg.write("{}    -   {} -   {}\n".format(regrets_avg, regrets_std, optimal_avg))
    result_quarter.flush()
    result_avg.flush()
    result_coeffs.flush()
    result_rank.flush()

    if rand_sampling_applied==True:
        regrets_all_users.append(random_model.get_regret_average())
        regrets_std_all_users.append(random_model.get_regret_std())
        regrets_avg_quarters.append([round(sum(regrets_all_users[-1][:users_quarter]) / users_quarter, 2) ,
                                    round(sum(regrets_all_users[-1][users_quarter:2*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_all_users[-1][2*users_quarter:3*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_all_users[-1][3*users_quarter:4*users_quarter]) / users_quarter, 2)])
        regrets_std_avg_quarters.append([round(sum(regrets_std_all_users[-1][:users_quarter]) / users_quarter, 2) ,
                                    round(sum(regrets_std_all_users[-1][users_quarter:2*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_std_all_users[-1][2*users_quarter:3*users_quarter]) / users_quarter, 2),
                                    round(sum(regrets_std_all_users[-1][3*users_quarter:4*users_quarter]) / users_quarter, 2)])

    
    #############################################################
    #                           Plots                           #
    #############################################################
    '''
    fig, ax = plt.subplots(1,1,sharey=False)
    fig, ax1 = plt.subplots(1,1,sharey=False)

    bplots.plot_regret(ax,user_count, policies, regrets_all_users,
                        simulation_count, batch_size)
    bplots.plot_regret(ax1,user_count, policies[:len(bandit_models)], regrets_all_users,
                        simulation_count, batch_size)
    bplots.plot_optimal_action_ratio(user_count, policies[:len(bandit_models)],
            optimal_action_ratio, simulation_count, batch_size,
            mode='per_user')
    '''
    #Plotting each coeff separetly for comparison
    '''
    for param_name in list(all_hypo_params_all_models):
        param = []
        policy_names = []
        for idx in range(len(beta_thompson_coeffs)):
            if param_name in beta_thompson_coeffs[idx].columns:
                param.append(beta_thompson_coeffs[idx][param_name])
                policy_names.append(policies[idx])

        bplots.plot_hypo_regression_param(user_count, policy_names, param_name,
                    param, true_coeff[param_name], simulation_count, batch_size)
    '''
    #if show_fig:
    #    plt.show()


    #############################################################
    #                 Saving results in csv files               #
    #############################################################
    '''
    for idx in range(len(bandit_models)):
        bandit_models[idx].get_selected_action().to_csv('{}context_action_thompson_{}.csv'.format(
                                save_output_folder,policies[idx]), index_label='iteration')
        bandit_models[idx].get_regret().to_csv('{}regrets_thompson_{}.csv'.format(
                                save_output_folder,policies[idx]), index_label='iteration')
        bandit_models[idx].get_optimal_action_ratio().to_csv('{}optimal_action_ratio_thompson_{}.csv'.format(
                                save_output_folder,policies[idx]), index_label='iteration')

    if rand_sampling_applied==True:
        random_model.get_regret().to_csv('{}regrets_random_sampling.csv'.format(
                                save_output_folder), index_label='iteration')
        random_model.get_optimal_action_ratio().to_csv('{}optimal_action_ratio_random_sampling.csv'.format(
                                save_output_folder), index_label='iteration')
    '''


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('input_file', metavar='input_file', type=str, nargs=1,
                        help='Name of the json config file')
    args = parser.parse_args()

    if (len(args.input_file) != 1) or (not args.input_file[0].endswith(".json")):
        print( "Error: Function should have only one input, name of the JSON config file." )
        sys.exit(1)

    save_output_folder = 'saved_output/raw_data_'+str(TODAY)+'_'+str(NOW.hour)+'-'+str(NOW.minute)+'-'+str(NOW.second)+"/"
    if not os.path.exists(save_output_folder):
        os.mkdir(save_output_folder)
    copy2('{}'.format(args.input_file[0]), '{}/'.format(save_output_folder))


    logging.info('cpu_count: {}'.format(multiprocessing.cpu_count()))
    pool = multiprocessing.Pool()

    input_data = args.input_file[0]
    input_data = json.load(open(input_data))
    noise_stats = input_data['true_model_params']['noise']
    true_model_terms = input_data['true_model_params']['true_model_terms']
    coeff_sets = [0, 1000, 2000, 6000]
    model_terms_count = len(true_model_terms)
    #true_coeff = dict(zip(true_model_terms, coeffs))

    try:
        data = pd.read_csv(input_data['true_model_params']['context_csv']) 
        user_count = data.shape[0]
        context_vars = data.columns
        logging.info("context_vars : {}".format(context_vars))
        input_context_type = "from_csv"
    except KeyError:
        logging.info("input_csv is not available. Trying to generate data.")
        context_vars =np.array(input_data['true_model_params']['context_vars'])
        context_dist = input_data['dist_of_context']
        user_count = input_data['user_count']
        input_context_type = "simulated"

    bandit_arms = input_data['possible_actions']
    experiment_vars=np.array(input_data['true_model_params']['experiment_vars'])

    ## Setting the training mode (hypo model) parameters
    policies = input_data['hypo_model_names']
    rand_sampling_applied = input_data['rand_sampling_applied']
    if rand_sampling_applied==True:
        policies.append('Random Sampling')
    hypo_params_all_models = input_data['hypo_model_params']
    NIG_priors = input_data['NIG_priors']
    ## Setting the simulation parameters
    batch_size = input_data['batch_size'] # 10
    simulation_count = input_data['simulation_count']  # 2500
    extensive = input_data['extensive']
    show_fig = input_data['show_fig']

    input_fixed_vars = {'noise_stats': noise_stats,
                        'true_model_terms': true_model_terms,
                        'context_vars': context_vars,
                        'context_dist': context_dist,
                        'user_count': user_count,
                        'input_context_type': input_context_type,
                        'bandit_arms': bandit_arms,
                        'experiment_vars': experiment_vars,
                        'policies': policies,
                        'hypo_params_all_models': hypo_params_all_models,
                        'NIG_priors': NIG_priors,
                        'batch_size': batch_size,
                        'simulation_count': simulation_count,
                        'extensive': extensive,
                        'rand_sampling_applied': rand_sampling_applied,
                        'show_fig': show_fig
                        }

    # Storing data
    result_rank = open("result_rank.txt","w")
    result_avg = open("result_avg.txt","w")
    result_quarter = open("result_quarter.txt","w")
    result_coeffs = open("result_coeffs.txt","w")
    result_rank.write("sorted Policies and their corresponding regret values\n")
    result_quarter.write("Average Regret per Quarter    -   Average Std per Quarter  -   Average Optimal Action Ratio per Quarter\n")
    result_avg.write("Average Regret    -   Average Std -   Average Optimal Action Ratio\n")
    result_coeffs.write("{}\n".format(true_model_terms))

    idxs = np.load('utils/random_generated_coeffs.npy')
    coeffs_all = np.zeros(model_terms_count)
    for rep in range(10):
        logging.info("*******set: {}   - Time: {}".format(rep, datetime.now()))
        #random_idxs = np.random.randint(len(coeff_sets), size=model_terms_count)
        true_coeff = [coeff_sets[i] for i in idxs[rep]]
        input_fixed_vars['true_coeff'] = true_coeff

        simulatingDiffModels(pool, input_fixed_vars,
            sim_name=str(args.input_file[0]))


    result_rank.close()
    result_quarter.close()
    result_avg.close()
    result_coeffs.close()
