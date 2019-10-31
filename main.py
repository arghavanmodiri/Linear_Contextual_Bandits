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
    hypo_params = input_dict['hypo_model_params']

    ## Setting the simulation parameters
    user_count = input_dict['user_count']
    batch_size = input_dict['batch_size'] # 10
    simulation_count = input_dict['simulation_count']  # 2500
    extensive = input_dict['extensive']
    rand_sampling_applied = input_dict['rand_sampling_applied']
    show_fig = input_dict['show_fig']


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

    bandit_model1 = tbm(hypo_params, user_count, batch_size,
                        experiment_vars, bandit_arms, true_coeff, extensive)

    for sim in range(0, simulation_count):
        logging.info("sim: {}".format(sim))
        a_pre = input_dict['NIG_priors']['a']
        b_pre = input_dict['NIG_priors']['b']
        #Hammad: Bias Correction
        mean_pre = np.zeros(len(hypo_params))
        cov_pre = np.identity(len(hypo_params))

        regression_intercept = []
        regression_d1 = []
        regression_d1x1 = []
        regression_intercept_random = []
        regression_d1_random = []
        regression_d1x1_random = []


        #Step 3: Calls the right policy
        '''
        Calls thompson sampling, random sampling, or any others which is
        specified
        by the user in command line (default: Calls thompson sampling)
            default priors: 0 unless it is specified by the user
        '''
        users_context = models.generate_true_dataset(context_vars, user_count, input_dict['dist_of_context'])

        #bandit_model1.apply_policy(users_context, mean_pre, cov_pre, a_pre, b_pre, noise_stats)


        thompson_output = thompson.apply_thompson_sampling(users_context,
                                                    experiment_vars,
                                                    bandit_arms,
                                                    hypo_params,
                                                    true_coeff,
                                                    batch_size,
                                                    extensive,
                                                    mean_pre,
                                                    cov_pre,
                                                    a_pre,
                                                    b_pre,
                                                    noise_stats)
        #policies.append(['Thompson Sampling'])
        regrets += thompson_output[2]
        
        save_regret_thompson_df = pd.concat([save_regret_thompson_df,
                                    pd.DataFrame(thompson_output[2])],
                                    ignore_index=True, axis=1)

        optimal_action_ratio_per_sim = np.array(list((thompson_output[1][i] in thompson_output[0][i]) for i in range(0,user_count))).astype(int)
        optimal_action_ratio += optimal_action_ratio_per_sim
        save_optimal_action_ratio_thompson_df = pd.concat([
                                save_optimal_action_ratio_thompson_df,
                                pd.DataFrame(optimal_action_ratio_per_sim)],
                                ignore_index=True, axis=1)

        beta_thompson_coeffs += np.array(thompson_output[5])

        #list is the true coefficients of parameters that exist in both hypo and true models
        true_params_in_hypo = []
        for  idx, hypo_param_name in enumerate(hypo_params):
            if(hypo_param_name in true_coeff):
                true_params_in_hypo.append(true_coeff[hypo_param_name])

        mse_per_sim = np.power(np.array(np.array(thompson_output[5]) -
                                np.array(true_params_in_hypo)),2)
        print(len(mse_per_sim))
        print("-------------------------------------")
        mse += mse_per_sim
        mse_per_sim_df = pd.DataFrame(mse_per_sim)
        mse_per_sim_df.columns = pd.MultiIndex.from_product([
                [sim], hypo_params])
        save_mse_thompson_df = pd.concat([save_mse_thompson_df,
                                mse_per_sim_df], axis=1)

        coeff_sign_error_per_sim = np.sign(np.array(true_params_in_hypo)) - np.sign(np.array(thompson_output[5])) == np.zeros(len(hypo_params))
        coeff_sign_error +=coeff_sign_error_per_sim
        coeff_sign_error_per_sim_df = pd.DataFrame(
            coeff_sign_error_per_sim.astype(int))
        coeff_sign_error_per_sim_df.columns = pd.MultiIndex.from_product([
                [sim], hypo_params])
        save_coeff_sign_err_thompson_df = pd.concat([
                                save_coeff_sign_err_thompson_df,
                                coeff_sign_error_per_sim_df], axis=1)


        #Hammad: Bias Correction
        # Changed == 3 to == len(true_coeff), meaning the model is accurately specified
        if len(true_params_in_hypo) == len(true_coeff):
            bias_in_coeff_per_sim = np.array(np.array(thompson_output[5]) - np.array(true_params_in_hypo))

        # Under specified model bias (Y = A0 + A1D)
        else:
            # Bias(A1) = E(A1) - (B1 + B2/2)
            # shouldn't this just be done once? same with true_params_in_hypo?
            #true_coeff_list_main = make_true_coeff_list(true_params_in_hypo, true_coeff, expected_vals)
            #the next line should be fixed and generalized! NOTEEEE
            bias_in_coeff_per_sim = np.array(np.array(thompson_output[5]) - np.array(true_params_in_hypo))
            true_coeff_list_main = [true_coeff_list[0], true_coeff_list[1] + true_coeff_list[2]/2]


        bias_in_coeff += bias_in_coeff_per_sim
        bias_in_coeff_per_sim_df = pd.DataFrame(bias_in_coeff_per_sim)
        bias_in_coeff_per_sim_df.columns = pd.MultiIndex.from_product([
                [sim], hypo_params])
        save_bias_in_coeff_thompson_df = pd.concat([
                                save_bias_in_coeff_thompson_df,
                                bias_in_coeff_per_sim_df], axis=1)

        if(rand_sampling_applied):
            rand_outputs= random.apply_random_sampling(users_context,
                                                        experiment_vars,
                                                        bandit_arms,
                                                        hypo_params,
                                                        true_coeff,
                                                        batch_size,
                                                        extensive,
                                                        noise_stats)
            regrets_rand += rand_outputs[2]
            save_regret_random_df = pd.concat([save_regret_random_df,
                                    pd.DataFrame(rand_outputs[2])],
                                    ignore_index=True, axis=1)

            optimal_action_ratio_rand_per_sim = np.array(list((
                    rand_outputs[1][i] in
                    thompson_output[0][i]) for i in 
                    range(0,user_count))).astype(int)
            optimal_action_ratio_rand += optimal_action_ratio_rand_per_sim
            save_optimal_action_ratio_random_df = pd.concat([
                            save_optimal_action_ratio_random_df,
                            pd.DataFrame(optimal_action_ratio_rand_per_sim)],
                            ignore_index=True, axis=1)

        ################# OLS REGRESSION STARTS ########################
        '''
        x1 = np.empty((0,len(users_context[0].keys())))
        for i in range(0,len(users_context)):
            user_context_list = np.array([])
            for key,value in users_context[i].items():
                user_context_list = np.append(user_context_list,value)
            x1 = np.append(x1,[user_context_list], axis=0)
        x1 = [x1[i][0] for i in range(0,len(x1))]
        d1 = [thompson_output[1][i][0] for i in range(0,len(thompson_output[1]))]
        #d1 = [rand_outputs[1][i][0] for i in range(0,len(rand_outputs[1]))]
        d1_x1 = [a*b for a,b in zip(d1,x1)]
        df = pd.DataFrame({'d1':d1, 'd1x1':d1_x1, 'y':thompson_output[3]})



        for iteration in range(1, user_count+1):
            regression = sm.ols(formula="y ~ d1 + d1x1",
                                data=df.iloc[:iteration]).fit()
            regression_intercept.append(regression.params['Intercept'])
            regression_d1.append(regression.params['d1'])
            regression_d1x1.append(regression.params['d1x1'])

        regression_intercept_all_sim.append(regression_intercept)
        regression_d1_all_sim.append(regression_d1)
        regression_d1x1_all_sim.append(regression_d1x1)

        #Only saving the output of OLS Random Policy, and not plotting it

        if(rand_sampling_applied):
            d1_r = [rand_outputs[1][i][0] for i in range(0,
                        len(rand_outputs[1]))]
            d1_x1_r = [a*b for a,b in zip(d1_r,x1)]
            df_r = pd.DataFrame({'d1':d1_r, 'd1x1':d1_x1_r,
                                    'y':rand_outputs[0]})
            for iteration in range(1, user_count+1):
                regression_r = sm.ols(formula="y ~ d1 + d1x1",
                                    data=df_r.iloc[:iteration]).fit()
                regression_intercept_random.append(
                                            regression_r.params['Intercept'])
                regression_d1_random.append(regression_r.params['d1'])
                regression_d1x1_random.append(regression_r.params['d1x1'])

            regression_intercept_all_sim_random.append(
                                                regression_intercept_random)
            regression_d1_all_sim_random.append(regression_d1_random)
            regression_d1x1_all_sim_random.append(regression_d1x1_random)

    regression_intercept_all_sim_df=pd.DataFrame(regression_intercept_all_sim)
    regression_d1_all_sim_df=pd.DataFrame(regression_d1_all_sim)
    regression_d1x1_all_sim_df=pd.DataFrame(regression_d1x1_all_sim)

    regression_intercept_all_sim_df.T.to_csv(
                                '{}thompson_ols_intercept.csv'.format(
                                save_output_folder), index_label='iteration')
    regression_d1_all_sim_df.T.to_csv(
                                '{}thompson_ols_d1.csv'.format(
                                save_output_folder), index_label='iteration')
    regression_d1x1_all_sim_df.T.to_csv(
                                '{}thompson_ols_d1x1.csv'.format(
                                save_output_folder), index_label='iteration')

    if(rand_sampling_applied):
        regression_intercept_all_sim_random_df=pd.DataFrame(
                                        regression_intercept_all_sim_random)
        regression_d1_all_sim_random_df=pd.DataFrame(
                                        regression_d1_all_sim_random)
        regression_d1x1_all_sim_random_df=pd.DataFrame(
                                        regression_d1x1_all_sim_random)

        regression_intercept_all_sim_random_df.T.to_csv(
                                    '{}random_ols_intercept.csv'.format(
                                    save_output_folder), index_label='iteration')
        regression_d1_all_sim_random_df.T.to_csv(
                                    '{}random_ols_d1.csv'.format(
                                    save_output_folder), index_label='iteration')
        regression_d1x1_all_sim_random_df.T.to_csv(
                                    '{}random_ols_d1x1.csv'.format(
                                    save_output_folder),
                                    index_label='iteration')

    regression_intercept_all_sim_mean = np.mean(
            regression_intercept_all_sim_df, axis=0)

    regression_d1_all_sim_mean = np.mean(regression_d1_all_sim_df, axis=0)
    regression_d1x1_all_sim_mean = np.mean(regression_d1x1_all_sim_df, axis=0)



    regression_intercept_all_sim_std = np.std(regression_intercept_all_sim_df, axis=0)

    regression_d1_all_sim_std = np.std(regression_d1_all_sim_df, axis=0)
    regression_d1x1_all_sim_std = np.std(regression_d1x1_all_sim_df, axis=0)

    regression_params_dict = {"intercept" : regression_intercept_all_sim_mean,
                           "d1" : regression_d1_all_sim_mean,
                            "d1x1": regression_d1x1_all_sim_mean}

    regression_params_std_dict = {"intercept":regression_intercept_all_sim_std,
                            "d1" : regression_d1_all_sim_std,
                            "d1x1": regression_d1x1_all_sim_std}


    bplots.plot_regression(user_count, regression_params_dict, regression_params_std_dict, true_coeff,
                simulation_count, batch_size, save_fig=True)'''
    ################# OLS REGRESSION ENDS ########################

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

    if(rand_sampling_applied):
        save_regret_random_df.to_csv('{}random_regrets.csv'.format(
                                    save_output_folder),
                                    index_label='iteration')
        save_optimal_action_ratio_random_df.to_csv(
                                    '{}random_optimal_action_ratio.csv'.format(
                                    save_output_folder),
                                    index_label='iteration')
        save_context_action_random_df.to_csv(
                                    '{}random_context_action.csv'.format(
                                    save_output_folder))

    regrets = regrets / simulation_count
    optimal_action_ratio = optimal_action_ratio /simulation_count
    mse = mse / simulation_count
    beta_thompson_coeffs = beta_thompson_coeffs / simulation_count
    bias_in_coeff = bias_in_coeff / simulation_count
    coeff_sign_error = coeff_sign_error / simulation_count

    if(rand_sampling_applied):
        policies.append(['Random Sampling'])
        regrets_rand = regrets_rand / simulation_count
        optimal_action_ratio_rand = optimal_action_ratio_rand/simulation_count

    #Step 4: Plots
    '''
    Plots some basic figures. In "Extensive Mode", details will be saved so
    user can plots more figures if desired.
    '''
    if(rand_sampling_applied):
        regrets_all_policies = np.stack((regrets, regrets_rand))
        optimal_action_ratio_all_policies = np.stack((optimal_action_ratio,
            optimal_action_ratio_rand))
        mse_all_policies = np.array([mse])

    else:
        regrets_all_policies = np.array([regrets])
        optimal_action_ratio_all_policies = np.array([optimal_action_ratio])
        mse_all_policies = np.array([mse])


    fig, ax = plt.subplots(1,1,sharey=False)
    bplots.plot_regret(ax,user_count, policies, regrets_all_policies,
                        simulation_count, batch_size)


    bplots.plot_optimal_action_ratio(user_count, policies,
            optimal_action_ratio_all_policies, simulation_count, batch_size,
            mode='per_user')

    bplots.plot_mse(user_count, ['Thompson Sampling'], mse_all_policies,
                    simulation_count, batch_size)
    bplots.plot_coeff_ranking(user_count, 'Thompson Sampling',
                beta_thompson_coeffs, hypo_params, simulation_count,
                batch_size, save_fig=True)

    bplots.plot_coeff_sign_error(user_count, 'Thompson Sampling', hypo_params,
                coeff_sign_error, simulation_count, batch_size, save_fig=True)


    bplots.plot_bias_in_coeff(user_count, 'Thompson Sampling', hypo_params,
                bias_in_coeff, simulation_count, batch_size, save_fig=True)

    
    if(show_fig):
        #plt.show(block=False)
        plt.show()

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

