import os
import numpy as np
import pandas as pd
import true_hypo_models as models
import making_decision
import policies.thompson_sampling_nig as thompson
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as nprnd
import statsmodels.formula.api as sm
from datetime import date
from datetime import datetime

TODAY = date.today()
NOW = datetime.now()

#np.set_printoptions(threshold=np.nan)

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
    # Hammad update: list of coefficinets
    true_coeff_list = list(true_coeff.values())
    print(true_coeff_list)
    context_vars = true_model_params['context_vars']
    experiment_vars = true_model_params['experiment_vars']

    # Simulation parameters
    user_count = 1000 # 1000
    batch_size = 10 # 10
    simulation_count = 500 #2500
    extensive = True
    rand_sampling_applied = False
    show_fig=True

    regrets = np.zeros(user_count)
    regrets_rand = np.zeros(user_count)
    optimal_action_ratio = np.zeros(user_count)
    optimal_action_ratio_rand = np.zeros(user_count)
    # Hammad update: MSE is for each parameter
    mse = np.zeros((user_count, len(hypo_params)))
    beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
    coeff_sign_error = np.zeros((user_count, len(hypo_params)))
    bias_in_coeff = np.zeros((user_count, len(hypo_params)))
    policies = []

    # Need to generalize to toher models
    regression_intercept_all_sim = []
    regression_d1_all_sim = []
    regression_d1x1_all_sim = []
    regression_intercept_all_sim_random = []
    regression_d1_all_sim_random = []
    regression_d1x1_all_sim_random = []
    policies.append(['Thompson Sampling'])


    x0_suboptimal_ratio = np.zeros(user_count)
    x1_suboptimal_ratio = np.zeros(user_count)
    x0_suboptimal_ratio_rand = np.zeros(user_count)
    x1_suboptimal_ratio_rand = np.zeros(user_count)
    x0_d0_count_quarter = [0]*4
    x0_d1_count_quarter = [0]*4
    x1_d0_count_quarter = [0]*4
    x1_d1_count_quarter = [0]*4

    # Make raw_data folder to save data
    save_output_folder = 'saved_output/raw_data' +str(TODAY)+'_'+str(NOW.hour)+str(NOW.minute)+str(NOW.second)+"/"
    if not os.path.exists(save_output_folder):
        # Creates folder if it already doesn't exist
        os.mkdir(save_output_folder)

    # Data frames for saving quantities of interest
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


    

    for sim in range(0, simulation_count):

        # Set priors
        print("sim: ",sim)
        a_pre = 2
        b_pre = 1
        mean_pre = np.zeros(len(hypo_params))
        cov_pre = np.identity(len(hypo_params))

        # How to generalize this to other models?
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
        #optimal_action_ratio += np.array(list((thompson_output[1][i] in thompson_output[0][i]) for i in range(0,user_count))).astype(int)
        # mse += np.power(np.array(thompson_output[3]) - np.array(thompson_output[4]),2)


        # Save TS regret
        save_regret_thompson_df = pd.concat([save_regret_thompson_df,
                                    pd.DataFrame(thompson_output[2])],
                                    ignore_index=True, axis=1)

        # Compute optimal prop. action chosen for X = 0 and X = 1
        optimal_action_ratio_per_sim = np.array(list((thompson_output[1][i] in thompson_output[0][i]) for i in range(0,user_count))).astype(int)
        optimal_action_ratio += optimal_action_ratio_per_sim
        save_optimal_action_ratio_thompson_df = pd.concat([
                                save_optimal_action_ratio_thompson_df,
                                pd.DataFrame(optimal_action_ratio_per_sim)],
                                ignore_index=True, axis=1)

       
        #mse_per_sim = np.power(np.array(thompson_output[3]) - np.array(thompson_output[4]),2)

        
        beta_thompson_coeffs += np.array(thompson_output[5])

        #for coeff_name, coeff_value in true_coeff.items():
        true_params_in_hypo = []
        for  idx, hypo_param_name in enumerate(hypo_params):
            if(hypo_param_name in true_coeff):
                true_params_in_hypo.append(true_coeff[hypo_param_name])

        # Compute and save MSE

        # Hammad update: MSE = E((coeff - true_param)^2)
        mse_per_sim = np.power(np.array(np.array(thompson_output[5]) - np.array(true_params_in_hypo)),2)
        mse += mse_per_sim
        mse_per_sim_df = pd.DataFrame(mse_per_sim)
        mse_per_sim_df.columns = pd.MultiIndex.from_product([[sim], hypo_params])
        save_mse_thompson_df = pd.concat([save_mse_thompson_df,mse_per_sim_df], axis=1)
        

        #coeff_sign_error += np.sign(np.array(true_params_in_hypo) * np.array(
        #    thompson_output[5]))

        coeff_sign_error_per_sim = np.sign(np.array(true_params_in_hypo)) - np.sign(np.array(thompson_output[5])) == np.zeros(len(hypo_params))
        coeff_sign_error +=coeff_sign_error_per_sim
        coeff_sign_error_per_sim_df = pd.DataFrame(coeff_sign_error_per_sim.astype(int))
        coeff_sign_error_per_sim_df.columns = pd.MultiIndex.from_product([[sim], hypo_params])
        save_coeff_sign_err_thompson_df = pd.concat([save_coeff_sign_err_thompson_df, coeff_sign_error_per_sim_df], axis=1)



        # Hammad update: bias = E(coeff) - true_param
        # bias_in_coeff is dimensions n_{user} x 3
        
        # Correct specified model bias (Y = B0 + B1X + B2X*D)

        # Need to generalize code
        if len(true_params_in_hypo) >= 3:
            # Bias(B1) = E(B1) - B1
            #bias_in_coeff += np.array(np.array(thompson_output[5]) - np.array(true_params_in_hypo))
            bias_in_coeff_per_sim = np.array(np.array(thompson_output[5]) - np.array(true_params_in_hypo))

        # Under specified model bias (Y = A0 + A1D)
        else:
            # Bias(A1) = E(A1) - (B1 + B2/2)
            true_coeff_list_main = [true_coeff_list[0], true_coeff_list[1] + true_coeff_list[2]/2]
            #bias_in_coeff += np.array(np.array(thompson_output[5]) - np.array(true_coeff_list_main))
            bias_in_coeff_per_sim = np.array(np.array(thompson_output[5]) - np.array(true_coeff_list_main))

        # Compute and save coefficient biases
        bias_in_coeff += bias_in_coeff_per_sim
        bias_in_coeff_per_sim_df = pd.DataFrame(bias_in_coeff_per_sim)
        bias_in_coeff_per_sim_df.columns = pd.MultiIndex.from_product([[sim], hypo_params])
        save_bias_in_coeff_thompson_df = pd.concat([save_bias_in_coeff_thompson_df, bias_in_coeff_per_sim_df], axis=1)

        
        #bias_in_coeff += np.array(np.array(true_params_in_hypo) - np.array(thompson_output[5]))

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
            '''optimal_action_ratio_rand += np.array(list((rand_outputs[1][i] in
                    thompson_output[0][i]) for i in 
                    range(0,user_count))).astype(int)'''
            save_regret_random_df = pd.concat([save_regret_random_df, pd.DataFrame(rand_outputs[2])], ignore_index=True, axis=1)


            optimal_action_ratio_rand_per_sim = np.array(list((
                    rand_outputs[1][i] in
                     thompson_output[0][i]) for i in 
                     range(0,user_count))).astype(int)
            optimal_action_ratio_rand += optimal_action_ratio_rand_per_sim
            save_optimal_action_ratio_random_df = pd.concat([
                            save_optimal_action_ratio_random_df,
                            pd.DataFrame(optimal_action_ratio_rand_per_sim)],
                            ignore_index=True, axis=1)

        quarter = int(user_count/4)
        '''
        first_user = 0
        last_user = quarter
        for j in range(0,4):
            for i in range(first_user, last_user):
                if(users_context[i]['x1']==0 and thompson_output[1][i][0]==0):
                    x0_d0_count_quarter[j] += 1
                elif(users_context[i]['x1']==0 and thompson_output[1][i][0]==1):
                    x0_d1_count_quarter[j] += 1
                elif(users_context[i]['x1']==1 and thompson_output[1][i][0]==0):
                    x1_d0_count_quarter[j] += 1
                elif(users_context[i]['x1']==1 and thompson_output[1][i][0]==1):
                    x1_d1_count_quarter[j] += 1
            first_user += quarter
            last_user += quarter'''

        x0_d0_count = np.zeros(user_count)
        x0_d1_count = np.zeros(user_count)
        x1_d0_count = np.zeros(user_count)
        x1_d1_count = np.zeros(user_count)
        for i in range(0,user_count):
            if(users_context[i]['x1']==0 and thompson_output[1][i][0]==0):
                    x0_d0_count[i] = 1
            elif(users_context[i]['x1']==0 and thompson_output[1][i][0]==1):
                x0_d1_count[i] = 1
            elif(users_context[i]['x1']==1 and thompson_output[1][i][0]==0):
                x1_d0_count[i] = 1
            elif(users_context[i]['x1']==1 and thompson_output[1][i][0]==1):
                x1_d1_count[i] = 1

        context_action_thompson_per_sim_df = pd.DataFrame([x0_d0_count,
                x0_d1_count, x1_d0_count, x1_d1_count]).T
        context_action_thompson_per_sim_df.columns = \
            pd.MultiIndex.from_product([[sim], ['x0_d0','x0_d1','x1_d0','x1_d1']])
        save_context_action_thompson_df = pd.concat([
                                save_context_action_thompson_df,
                                context_action_thompson_per_sim_df], axis=1)

        x0_d0_count = np.cumsum(x0_d0_count)
        x0_d1_count = np.cumsum(x0_d1_count)
        x1_d0_count = np.cumsum(x1_d0_count)
        x1_d1_count = np.cumsum(x1_d1_count)
        x0_suboptimal_ratio += np.divide(x0_d0_count,(x0_d0_count+x0_d1_count),
            out=np.zeros_like(x0_d0_count), where=(x0_d0_count+x0_d1_count)!=0)

        x1_suboptimal_ratio += np.divide(x1_d1_count,(x1_d1_count+x1_d0_count),
            out=np.zeros_like(x1_d1_count), where=(x1_d1_count+x1_d0_count)!=0)
 
        if(rand_sampling_applied):
            x0_d0_count = np.zeros(user_count)
            x0_d1_count = np.zeros(user_count)
            x1_d0_count = np.zeros(user_count)
            x1_d1_count = np.zeros(user_count)
            for i in range(0,user_count):
                if(users_context[i]['x1']==0 and rand_outputs[1][i][0]==0):
                        x0_d0_count[i] = 1
                elif(users_context[i]['x1']==0 and rand_outputs[1][i][0]==1):
                    x0_d1_count[i] = 1
                elif(users_context[i]['x1']==1 and rand_outputs[1][i][0]==0):
                    x1_d0_count[i] = 1
                elif(users_context[i]['x1']==1 and rand_outputs[1][i][0]==1):
                    x1_d1_count[i] = 1

            context_action_random_per_sim_df = pd.DataFrame([x0_d0_count,
                    x0_d1_count, x1_d0_count, x1_d1_count]).T
            context_action_random_per_sim_df.columns = \
                pd.MultiIndex.from_product([[sim], ['x0_d0','x0_d1','x1_d0','x1_d1']])
            save_context_action_random_df = pd.concat([
                                    save_context_action_random_df,
                                    context_action_random_per_sim_df],
                                    axis=1)
            x0_d0_count = np.cumsum(x0_d0_count)
            x0_d1_count = np.cumsum(x0_d1_count)
            x1_d0_count = np.cumsum(x1_d0_count)
            x1_d1_count = np.cumsum(x1_d1_count)
            x0_suboptimal_ratio_rand += np.divide(x0_d0_count,
                    (x0_d0_count+x0_d1_count),out=np.zeros_like(x0_d0_count),
                    where=(x0_d0_count+x0_d1_count)!=0)

            x1_suboptimal_ratio_rand += np.divide(x1_d1_count,
                    (x1_d1_count+x1_d0_count),out=np.zeros_like(x1_d1_count),
                    where=(x1_d1_count+x1_d0_count)!=0)






        ################# OLS REGRESSION STARTS ########################

        
        # Construct context vector (convert to list)
        x1 = np.empty((0,len(users_context[0].keys())))
        for i in range(0,len(users_context)):
            user_context_list = np.array([])
            for key,value in users_context[i].items():
                user_context_list = np.append(user_context_list,value)
            x1 = np.append(x1,[user_context_list], axis=0)
        x1 = [x1[i][0] for i in range(0,len(x1))]

        # Thompson policy assignment
        #d1 = [thompson_output[1][i][0] for i in range(0,len(thompson_output[1]))]

        # Hammad Update: Random policy assignment
        #d1 = [rand_outputs[1][i][0] for i in range(0,len(rand_outputs[1]))]
        
        # Get values for interaction term
        # thomposon_output[1] is hypo. optimal action for a given simulation across all users
        d1 = [thompson_output[1][i][0] for i in range(0,len(thompson_output[1]))]
        d1_x1 = [a*b for a,b in zip(d1,x1)]
        df = pd.DataFrame({'d1':d1, 'd1x1':d1_x1, 'y':thompson_output[3]})
        #df = pd.DataFrame({'d1':d1, 'y':thompson_output[3]})

        # Thompson policy data frame
        #df = pd.DataFrame({'d1':d1, 'd1x1':d1_x1, 'y':thompson_output[3]})

        # Hammad update: Random policy data frame
        #df = pd.DataFrame({'d1':d1, 'd1x1':d1_x1, 'y':rand_outputs[0]})

        # OLS Regression for N = user itteration
        for iteration in range(1, user_count+1):
            regression = sm.ols(formula="y ~ d1 + d1x1",
                                data=df.iloc[:iteration]).fit()
            #regression = sm.ols(formula="y ~ d1",
            #                    data=df.iloc[:iteration]).fit()
            regression_intercept.append(regression.params['Intercept'])
            regression_d1.append(regression.params['d1'])
            regression_d1x1.append(regression.params['d1x1'])

        # Compute OLS implies policy
        # Dictonary of OLS coefficients
        ols_coeff_dict = {"intercept": regression.params['Intercept'], "d1":regression.params['d1'], "d1*x1":regression.params['d1x1']}

        # Contextual values
        context_value0 = {"x1": 0}
        context_value1 = {"x1": 1}

        # Compute optimal arm for X1 = 0
        ols_optimal_x0 = making_decision.pick_true_optimal_arm(ols_coeff_dict, context_value0, experiment_vars,bandit_arms)[0][0][0]
        #print("OLS optimal action x = 0 is " + str(ols_optimal_x0))


        # Compute optimal arm for X1 = 1
        ols_optimal_x1 = making_decision.pick_true_optimal_arm(ols_coeff_dict, context_value1, experiment_vars,bandit_arms)[0][0][0]
        #print("OLS optimal action x = 1 is " + str(ols_optimal_x1))

        # Determine whether OLS picked optimal action
        ols_action_optimal_sim = []
        ols_action_optimal = (ols_optimal_x0 == 1 and ols_optimal_x1 == 0)
        print(ols_action_optimal)

        # Simulations in which optimal policy was choosen by OLS
        ols_action_optimal_sim.append(ols_action_optimal)

        # OLS regression coefficients for each simulations
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
                regression_r = sm.ols(formula="y ~ d1 + d1x1", data=df_r.iloc[:iteration]).fit()
                regression_intercept_random.append(regression_r.params['Intercept'])
                regression_d1_random.append(regression_r.params['d1'])
                regression_d1x1_random.append(regression_r.params['d1x1'])

            regression_intercept_all_sim_random.append(regression_intercept_random)
            regression_d1_all_sim_random.append(regression_d1_random)
            regression_d1x1_all_sim_random.append(regression_d1x1_random)
    
    
    regression_intercept_all_sim_df=pd.DataFrame(regression_intercept_all_sim)
    regression_d1_all_sim_df=pd.DataFrame(regression_d1_all_sim)
    regression_d1x1_all_sim_df=pd.DataFrame(regression_d1x1_all_sim)

    # Average OLS coefficients across simulations
    regression_intercept_all_sim_mean = np.mean(regression_intercept_all_sim_df, axis=0)
    regression_d1_all_sim_mean = np.mean(regression_d1_all_sim_df, axis=0)
    regression_d1x1_all_sim_mean = np.mean(regression_d1x1_all_sim_df, axis=0)

    # Save OLS coefficients

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
                                    save_output_folder), index_label='iteration')
        
    regression_intercept_all_sim_mean = np.mean(
            regression_intercept_all_sim_df, axis=0)









    

    # Standard deviation of coefficients acrosss simulations
    regression_intercept_all_sim_std = np.std(regression_intercept_all_sim_df, axis=0)
    regression_d1_all_sim_std = np.std(regression_d1_all_sim_df, axis=0)
    regression_d1x1_all_sim_std = np.std(regression_d1x1_all_sim_df, axis=0)

    regression_params_dict = {"intercept" : regression_intercept_all_sim_mean,
                           "d1" : regression_d1_all_sim_mean,
                            "d1x1": regression_d1x1_all_sim_mean}

    regression_params_std_dict = {"intercept":regression_intercept_all_sim_std,
                            "d1" : regression_d1_all_sim_std,
                            "d1x1": regression_d1x1_all_sim_std}

    # Plot OLS coefficients for either thompson or random policy
    '''bplots.plot_regression(user_count, regression_params_dict, regression_params_std_dict, true_coeff,
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

    x0_d0_count_quarter = np.array(x0_d0_count_quarter) /simulation_count / quarter
    x0_d1_count_quarter = np.array(x0_d1_count_quarter) / simulation_count / quarter
    x1_d0_count_quarter = np.array(x1_d0_count_quarter) /simulation_count /quarter
    x1_d1_count_quarter = np.array(x1_d1_count_quarter) / simulation_count / quarter
    print("x0_d0_count_quarter : ", x0_d0_count_quarter)
    print("x0_d1_count_quarter : ", x0_d1_count_quarter)
    print("x1_d0_count_quarter : ", x1_d0_count_quarter)
    print("x1_d1_count_quarter : ", x1_d1_count_quarter)

    x0_suboptimal_ratio = x0_suboptimal_ratio / simulation_count
    x1_suboptimal_ratio = x1_suboptimal_ratio / simulation_count
    


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
        x0_suboptimal_ratio_rand = x0_suboptimal_ratio_rand /simulation_count
        x1_suboptimal_ratio_rand = x1_suboptimal_ratio_rand /simulation_count



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
        suboptimal_ratio_all_policies = np.stack((x0_suboptimal_ratio,
                x1_suboptimal_ratio, x0_suboptimal_ratio_rand, x1_suboptimal_ratio_rand))
    else:
        regrets_all_policies = np.array([regrets])
        optimal_action_ratio_all_policies = np.array([optimal_action_ratio])
        mse_all_policies = np.array([mse])
        suboptimal_ratio_all_policies = np.stack((x0_suboptimal_ratio, x1_suboptimal_ratio))

    
    bplots.plot_regret(user_count, policies, regrets_all_policies,
                        simulation_count, batch_size)


    if(rand_sampling_applied):
         bplots.plot_suboptimal_action_ratio(user_count, ['Thompson Sampling X = 0','Thompson Sampling X = 1', 'Random Sampling X=0', 'Random Sampling X=1'],
                suboptimal_ratio_all_policies, simulation_count, batch_size,
                mode='per_user')

    
    
    bplots.plot_optimal_action_ratio(user_count, policies,
            optimal_action_ratio_all_policies, simulation_count, batch_size,
            mode='per_user')
    '''
    bplots.plot_mse(user_count, ['Thompson Sampling'], mse_all_policies,
                    simulation_count, batch_size)'''
    
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
