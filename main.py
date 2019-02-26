import numpy as np
import pandas as pd
import true_hypo_models as models
import policies.thompson_sampling_nig as thompson
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm


np.set_printoptions(threshold=np.nan)

def main(mode=None):
    """start the model"""


    #Step 1: Configuration (to be added in a seprate function)
    '''
    Parse the script command
    Check the csv files to make sure they are not empty
        True_Model_Coefficients.csv: will have true model coefficients
        Hypo_Model_Design.csv: will have the list of parameters to be used
    Create output folder if not existed and if in "Extensive Mode"
        Extensive Mode: details of the simulation will be saved in output files
        Test Mode: Just the rewards and basics will be printed on the screen
    Set the output files prefix. E.g. dec_01_1000users_
    and any other configuration to be set
    '''


    #Step 2: Setting the models
    '''
    Calls true_hypo_models.read_true_model to parse True_Model_Coefficients.csv
    Calls true_hypo_models.read_hypo_model to parse Hypo_Model_Design.csv
    Based on the variables, find the list of all bandit arms
    '''
 
    #models.generate_true_dataset(np.array(['var1', 'var2']), user_count)
    true_model_params = models.read_true_model()
    hypo_params = models.read_hypo_model()
    bandit_arms = models.find_possible_actions()
    noise_stats = true_model_params['noise']
    true_coeff = true_model_params['true_coeff']
    context_vars = true_model_params['context_vars']
    experiment_vars = true_model_params['experiment_vars']

    user_count = 1000
    batch_size = 10
    simulation_count = 500
    extensive = True
    rand_sampling_applied = True
    show_fig=True

    regrets = np.zeros(user_count)
    regrets_rand = np.zeros(user_count)
    optimal_action_ratio = np.zeros(user_count)
    optimal_action_ratio_rand = np.zeros(user_count)
    mse = np.zeros(user_count)
    beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
    coeff_sign_error = np.zeros((user_count, len(hypo_params)))
    bias_in_coeff = np.zeros((user_count, len(hypo_params)))
    policies = []
    regression_intercept_all_sim = []
    regression_d1_all_sim = []
    regression_d1x1_all_sim = []
    policies.append(['Thompson Sampling'])

    for sim in range(0, simulation_count):
        print("sim: ",sim)
        a_pre = 0
        b_pre = 0
        mean_pre = np.zeros(len(hypo_params))
        cov_pre = np.identity(len(hypo_params))

        regression_intercept = []
        regression_d1 = []
        regression_d1x1 = []


        #Step 3: Calls the right policy
        '''
        Calls thompson sampling, random sampling, or any others which is
        specified
        by the user in command line (default: Calls thompson sampling)
            default priors: 0 unless it is specified by the user
        '''
        users_context = models.generate_true_dataset(context_vars, user_count)

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
        optimal_action_ratio += np.array(list((thompson_output[1][i] in thompson_output[0][i]) for i in range(0,user_count))).astype(int)
        mse += np.power(np.array(thompson_output[3]) - np.array(thompson_output
            [4]),2)
        beta_thompson_coeffs += np.array(thompson_output[5])

        #for coeff_name, coeff_value in true_coeff.items():
        true_params_in_hypo = []
        for  idx, hypo_param_name in enumerate(hypo_params):
            if(hypo_param_name in true_coeff):
                true_params_in_hypo.append(true_coeff[hypo_param_name])

        #coeff_sign_error += np.sign(np.array(true_params_in_hypo) * np.array(
        #    thompson_output[5]))
        coeff_sign_error += np.sign(np.array(true_params_in_hypo)) - np.sign(np.array(thompson_output[5])) == np.zeros(len(hypo_params))
        bias_in_coeff += np.array(np.array(true_params_in_hypo) - np.array(
            thompson_output[5]))

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
            optimal_action_ratio_rand += np.array(list((rand_outputs[1][i] in
                    thompson_output[0][i]) for i in 
                    range(0,user_count))).astype(int)


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
    
    regression_intercept_all_sim_df=pd.DataFrame(regression_intercept_all_sim)
    regression_d1_all_sim_df=pd.DataFrame(regression_d1_all_sim)
    regression_d1x1_all_sim_df=pd.DataFrame(regression_d1x1_all_sim)

    regression_intercept_all_sim_mean = np.mean(regression_intercept_all_sim_df, axis=0)

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
                simulation_count, batch_size, save_fig=True)
    '''
    ################# OLS REGRESSION ENDS ########################

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
        #optimal_action_ratio_all_policies = np.stack((optimal_action_ratio,
        #regrets_rand))
        optimal_action_ratio_all_policies = np.stack((optimal_action_ratio,
            optimal_action_ratio_rand))
        mse_all_policies = np.array([mse])
    else:
        regrets_all_policies = np.array([regrets])
        optimal_action_ratio_all_policies = np.array([optimal_action_ratio])
        mse_all_policies = np.array([mse])


    bplots.plot_regret(user_count, policies, regrets_all_policies,
                        simulation_count, batch_size)

    bplots.plot_optimal_action_ratio(user_count, policies,
            optimal_action_ratio_all_policies, simulation_count, batch_size,
            mode='per_batch')

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
    main()