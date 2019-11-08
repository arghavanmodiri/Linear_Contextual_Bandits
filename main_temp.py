#!/usr/bin/env python
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import true_hypo_models as models
import policies.thompson_sampling_nig as thompson
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from datetime import date
from datetime import datetime
from training_models import training_bandit_model as tbm

TODAY = date.today()
NOW = datetime.now()



def main(input_dict, mode=None):
    """start the model"""

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s')

    ## Setting the true model parameters and the enviromental parameters
    true_model_params = input_dict['true_model_params']
    bandit_arms = input_dict['possible_actions']
    noise_stats = true_model_params['noise']
    true_coeff = true_model_params['true_coeff']
    true_coeff_list = list(true_coeff.values())
    context_vars = np.array(true_model_params['context_vars'])
    experiment_vars = np.array(true_model_params['experiment_vars'])
    logging.info(true_coeff_list)

    ## Setting the training mode (hypo model) parameters
    policies = input_dict['hypo_model_names']
    hypo_params_all_models = input_dict['hypo_model_params']
    print(len(hypo_params_all_models))
    ## Setting the simulation parameters
    user_count = input_dict['user_count']
    batch_size = input_dict['batch_size'] # 10
    simulation_count = input_dict['simulation_count']  # 2500
    extensive = input_dict['extensive']
    rand_sampling_applied = input_dict['rand_sampling_applied']
    show_fig = input_dict['show_fig']


    '''
    regrets = np.zeros(user_count)
    regrets_rand = np.zeros(user_count)
    optimal_action_ratio = np.zeros(user_count)
    optimal_action_ratio_rand = np.zeros(user_count)
    mse = np.zeros((user_count, len(hypo_params)))
    beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
    coeff_sign_error = np.zeros((user_count, len(hypo_params)))
    bias_in_coeff = np.zeros((user_count, len(hypo_params)))
    policies = []
    regression_intercept_all_sim = []
    regression_d1_all_sim = []
    regression_d1x1_all_sim = []
    regression_intercept_all_sim_random = []
    regression_d1_all_sim_random = []
    regression_d1x1_all_sim_random = []
    policies.append(['Thompson Sampling'])
    '''

    save_output_folder = 'saved_output/raw_data_'+str(TODAY)+'_'+str(NOW.hour)+str(NOW.minute)+str(NOW.second)+"/"
    if not os.path.exists(save_output_folder):
        os.mkdir(save_output_folder)
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
    save_context_action_thompson_df = pd.DataFrame()
    save_context_action_random_df = pd.DataFrame()

    bandit_models = []
    for idx in range(len(hypo_params_all_models)):
        bandit_models.append(
                    tbm(hypo_params_all_models[idx], user_count,
                    batch_size, experiment_vars, bandit_arms, true_coeff,
                    extensive))

    for sim in range(0, simulation_count):
        logging.info("sim: {}".format(sim))
        a_pre = input_dict['NIG_priors']['a']
        b_pre = input_dict['NIG_priors']['b']
        #Hammad: Bias Correction
        #mean_pre = np.zeros(len(hypo_params))
        #cov_pre = np.identity(len(hypo_params))

        users_context = models.generate_true_dataset(context_vars, user_count, input_dict['dist_of_context'])

        #Step 3: Calls the sampling policy to select action for each user
        for idx in range(len(hypo_params_all_models)):
            bandit_models[idx].apply_thompson(users_context, a_pre, b_pre, noise_stats)


        #regret
        #optimal_action_ratio
        #beta_thompson_coeffs
        #mse
        #coeff_sign_error_per
        #Bias Correction

        # Add Random Policy

    #Step 4: Saving the results in csv files and plotting metrics of interest
    #also save the hypo model and true model and other metrics in text file
    save_regret_thompson_df.to_csv('{}thompson_regrets.csv'.format(
                                save_output_folder), index_label='iteration')
    save_optimal_action_ratio_thompson_df.to_csv(
                                '{}thompson_optimal_action_ratio.csv'.format(
                                save_output_folder), index_label='iteration')
    save_mse_thompson_df.to_csv('{}thompson_mse.csv'.format(
                                save_output_folder), index_label='iteration')
    save_bias_in_coeff_thompson_df.to_csv(
                                '{}thompson_bias_in_coeff.csv'.format(
                                save_output_folder))
    save_coeff_sign_err_thompson_df.to_csv(
                                '{}thompson_coeff_sign_err.csv'.format(
                                save_output_folder))
    save_context_action_thompson_df.to_csv(
                                '{}thompson_context_action.csv'.format(
                                save_output_folder))


    #Plots
    '''
    Plots some basic figures. In "Extensive Mode", details will be saved so
    user can plots more figures if desired.
    '''
    all_hypo_params_all_models = {}
    regrets_all_users = []
    regrets_std_all_users = []

    regression_params_avg_quarters = []
    regression_params_std_quarters = []

    regrets_avg_quarters = []
    regrets_std_avg_quarters = []
    optimal_action_ratio = []
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
    
        #beta_thompson_coeffs: lists of dataframes!
        beta_thompson_coeffs.append(
            bandit_models[idx].get_beta_thompson_coeffs_average())

    strTable = "<html><link rel='stylesheet' type='text/css' href='mystyle.css'>"
    strTable = strTable+"<h1>Regret</h1>"
    strTable = strTable+ "<table>\
            <tr>\
            <th>policy</th><th>Regret (1st quarter)</th><th>Regret (2nd quarter)</th><th>Regret (3rd quarter)</th><th>Regret (4th quarter)</th>\
            </tr>"
    for idx in range(len(policies)):
        strRW = "<tr>"
        strRW = strRW + "<td>" + str(policies[idx]) + "</td>"
        strRW = strRW + "<td>" + str(regrets_avg_quarters[idx][0]) +" (" + str(regrets_std_avg_quarters[idx][0]) + ")" + "</td>"
        strRW = strRW + "<td>" + str(regrets_avg_quarters[idx][1]) +" (" + str(regrets_std_avg_quarters[idx][1]) + ")" +"</td>"
        strRW = strRW + "<td>" + str(regrets_avg_quarters[idx][2]) +" (" + str(regrets_std_avg_quarters[idx][2]) + ")" +"</td>"
        strRW = strRW + "<td>" + str(regrets_avg_quarters[idx][3]) +" (" + str(regrets_std_avg_quarters[idx][3]) + ")" +"</td>"
        strRW = strRW + "</tr>"
        strTable = strTable+strRW
    
    strTable = strTable+"The values in parentheses is standard deviation."
    strTable = strTable+"</table>"
    strTable = strTable+"Calculating std: the std for each user is calculated over all simulations. Then, the average of the calculated\
                values for each quarter is taken and reported in table above."

    strTable = strTable+"<br><br><br>"
    strTable = strTable+"The below tables shows the average coeff of each term for different policies."
    strTable = strTable+"<h1>Regression Coeff for different policies</h1>"
    strTable = strTable+"<h1>First Quarter (user 1 to "+ str(int(user_count/4)) +")</h1>"
    strTable = strTable+"<table><tr><th>policy</th>"
    for param_name in list(all_hypo_params_all_models):
        strTable = strTable + "<th>"+str(param_name)+"</th>"
    strTable = strTable + "</tr>"

    for param_name in list(all_hypo_params_all_models):
        strRW = "<tr>"
        strRW = strRW + "<td>True Policy</td>"
        if param_name in true_coeff.keys():
            strRW = strRW + "<td>" + str(true_coeff[param_name]) +"</td>"
        else:
            strRW = strRW + "<td>-</td>"
    strRW = strRW + "</tr>"
    strTable = strTable+strRW

    for idx in range(len(policies)):
        strRW = "<tr>"
        print(param_name) 
        strRW = strRW + "<td>"+str(policies[idx])+"</td>"
        for param_name in list(all_hypo_params_all_models):
            if param_name in beta_thompson_coeffs[idx].columns:
                #param.append(beta_thompson_coeffs[idx][param_name])
                users_quarter = int(len(beta_thompson_coeffs[idx][param_name]) / 4)
                '''regression_params_avg_quarters.append([
                                                                                                round(beta_thompson_coeffs[idx][param_name][:users_quarter].mean(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][users_quarter:2*users_quarter].mean(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][2*users_quarter:3*users_quarter].mean(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][3*users_quarter:].mean(), 2)])
                                                                regression_params_std_quarters.append([
                                                                                                round(beta_thompson_coeffs[idx][param_name][:users_quarter].std(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][users_quarter:2*users_quarter].std(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][2*users_quarter:3*users_quarter].std(), 2),
                                                                                                round(beta_thompson_coeffs[idx][param_name][3*users_quarter:].std(), 2)])'''
                strRW = strRW + "<td>" + str(round(beta_thompson_coeffs[idx][param_name][:users_quarter].mean(), 2)) +" (" + str(round(beta_thompson_coeffs[idx][param_name][:users_quarter].std(), 2)) + ")" +"</td>"
            else:
                '''regression_params_avg_quarters.append(["NA", "NA", "NA", "NA"])
                                                                regression_params_std_quarters.append(["NA", "NA", "NA", "NA"])'''
                strRW = strRW + "<td>-</td>"
        strRW = strRW + "</tr>"
        strTable = strTable+strRW

            
    '''for policy in policies:
                    strRW = "<tr>"
                    strRW = strRW + "<td>" + str(policies[idx]) + "</td>"
                    for param_name in list(all_hypo_params_all_models):
                        strRW = strRW + "<td>" + str(regrets_avg_quarters[idx][1]) +" (" + str(regrets_std_avg_quarters[idx][1]) + ")" +"</td>"
            '''


    strTable = strTable + "</tr>"  
    strTable = strTable+"The values in parentheses is standard deviation."
    strTable = strTable+"</table>"            
    strTable = strTable+"</html>"
         
    hs = open("simulation.html", 'w')
    hs.write(strTable)
    
    fig, ax = plt.subplots(1,1,sharey=False)
    bplots.plot_regret(ax,user_count, policies, regrets_all_users,
                        simulation_count, batch_size)
    bplots.plot_optimal_action_ratio(user_count, policies,
            optimal_action_ratio, simulation_count, batch_size,
            mode='per_user')

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
    #plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('input_file', metavar='input_file', type=str, nargs=1,
                        help='Name of the json config file')
    args = parser.parse_args()

    if (len(args.input_file) != 1) or (not args.input_file[0].endswith(".json")):
        print( "Error: Function should have only one input, name of the JSON config file." )
        sys.exit(1)

    input_data = args.input_file[0]
    input_data = json.load(open(input_data))
    main(input_data)

