from subprocess import check_output
import numpy as numpy
import pandas as pd
import plots.plot_basics as bplots
import matplotlib.pyplot as plt

parent_folder_name = "saved_output/raw_data/"
interaction_size = 'D'
d1 = 0.3
d1x1 = -1.2
intercept = 0

# Load data from Small Interaction, MAB Correct, OLS Correct
folder_name_c_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Correct_OLS_Correct/"
thompson_regrets_c_c = pd.read_csv(
        "{}thompson_regrets.csv".format(folder_name_c_c), index_col=0)
thompson_optimal_action_per_group_c_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_c_c),
        index_col=0, header=[0,1])

thompson_ols_intercept_c_c = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_c_c), index_col=0)
thompson_ols_d1_c_c = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_c_c), index_col=0)
thompson_ols_d1x1_c_c = pd.read_csv(
        "{}thompson_ols_d1x1.csv".format(folder_name_c_c), index_col=0)

thompson_bias_in_coeff_c_c = pd.read_csv(
        "{}thompson_bias_in_coeff.csv".format(folder_name_c_c),
        index_col=0, header=[0,1])


# Load data from Small Interaction, MAB UnderSpecified, OLS Correct
folder_name_u_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Under_OLS_Correct/"
thompson_regrets_u_c = pd.read_csv(
        "{}thompson_regrets.csv".format(folder_name_u_c), index_col=0)
thompson_optimal_action_per_group_u_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_u_c),
        index_col=0, header=[0,1])

thompson_ols_intercept_u_c = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_u_c), index_col=0)
thompson_ols_d1_u_c = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_u_c), index_col=0)
thompson_ols_d1x1_u_c = pd.read_csv(
        "{}thompson_ols_d1x1.csv".format(folder_name_u_c), index_col=0)

thompson_bias_in_coeff_u_c = pd.read_csv(
        "{}thompson_bias_in_coeff.csv".format(folder_name_u_c),
        index_col=0, header=[0,1])

# Load data from Small Interaction, MAB UnderSpecified, OLS UnderSpecified
'''
folder_name_u_u = parent_folder_name + "Small_MAB_Under_OLS_Under/"
thompson_ols_intercept_u_u = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_u_u), index_col=0)
thompson_ols_d1_u_u = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_u_u), index_col=0)
'''

# Load data from Small Interaction, Uniform Sampling, OLS Correct
folder_name_c = parent_folder_name + \
                    interaction_size+ \
                    "_Random_OLS_Correct/"
uniform_regrets_c = pd.read_csv(
        "{}random_regrets.csv".format(folder_name_c), index_col=0)




user_count = thompson_regrets_u_c.shape[0]
simulation_count = thompson_regrets_u_c.shape[1]
batch_size = 10

fig, ax = plt.subplots(1,2,sharey=False)


plt.subplots_adjust(wspace=0.27,right = 0.92, left=0.08)
#Processing Regret
regrets = [thompson_regrets_c_c.mean(axis=1).values.reshape(1,user_count),
        thompson_regrets_u_c.mean(axis=1).values.reshape(1,user_count),
        uniform_regrets_c.mean(axis=1).values.reshape(1,user_count)]
bplots.plot_regret(ax[0], user_count,
                    ['Contextual MAB',
                    'Non-Contextual MAB',
                    'Uniform Sampling'], 
                    regrets, simulation_count, batch_size)

#Processing Selected Opimal Action Ratio for Each Group
#print(thompson_optimal_action_per_group_c_c.iloc[:, thompson_optimal_action_per_group_c_c.columns.get_level_values(1)=='x0_d0'])


#opt_act_per_group_c_c_mean = thompson_optimal_action_per_group_c_c.cumsum(axis=0).mean(axis=1, level=1)
#opt_act_per_group_u_c_mean = thompson_optimal_action_per_group_u_c.cumsum(axis=0).mean(axis=1, level=1)

opt_act_per_group_c_c_mean = thompson_optimal_action_per_group_c_c.sum(axis=1, level=1)
opt_act_per_group_u_c_mean = thompson_optimal_action_per_group_u_c.sum(axis=1, level=1)


optimal_action_ratio_per_group_c_c ={
    'x0': opt_act_per_group_c_c_mean['x0_d1']/(opt_act_per_group_c_c_mean['x0_d0']+opt_act_per_group_c_c_mean['x0_d1']),
    'x1': opt_act_per_group_c_c_mean['x1_d0']/(opt_act_per_group_c_c_mean['x1_d0']+opt_act_per_group_c_c_mean['x1_d1'])
    }
optimal_action_ratio_per_group_u_c ={
    'x0': opt_act_per_group_u_c_mean['x0_d1']/(opt_act_per_group_u_c_mean['x0_d0']+opt_act_per_group_u_c_mean['x0_d1']),
    'x1': opt_act_per_group_u_c_mean['x1_d0']/(opt_act_per_group_u_c_mean['x1_d0']+opt_act_per_group_u_c_mean['x1_d1'])
    }
optimal_action_ratio_per_group = [optimal_action_ratio_per_group_c_c,
                                optimal_action_ratio_per_group_u_c]
#bplots.plot_optimal_action_ratio_pre_group_multiple(ax[1], user_count, ['Contextual MAB', 'Non-Contextual MAB'],
#                                optimal_action_ratio_per_group,
#                                simulation_count, batch_size)


#Processing Bias in Hypo Coeff
'''
bias_in_coeff = [
        thompson_bias_in_coeff_c_c.mean(axis=1,level=1),
        thompson_bias_in_coeff_u_c.mean(axis=1,level=1)]
bplots.plot_bias_in_coeff_multuple(ax2[0], user_count,
                        ['Correctly-Specified MAB', 'Under-Specified MAB'],
                        bias_in_coeff, simulation_count, batch_size)
'''

#Processing OLS
ols_params_dict_c_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "d1" : thompson_ols_d1_c_c.mean(axis=1) - d1,
        "d1x1": thompson_ols_d1x1_c_c.mean(axis=1) - d1x1
        }

ols_params_dict_c_c_std= {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "d1" : thompson_ols_d1_c_c.std(axis=1),
        "d1x1": thompson_ols_d1x1_c_c.std(axis=1)
        }
print(ols_params_dict_c_c_std['d1'])
ols_params_dict_u_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "d1" : thompson_ols_d1_u_c.mean(axis=1)- d1,
        "d1x1": thompson_ols_d1x1_u_c.mean(axis=1)- d1x1
        }
ols_params_dict_u_c_std= {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "d1" : thompson_ols_d1_u_c.std(axis=1),
        "d1x1": thompson_ols_d1x1_u_c.std(axis=1)
        }
'''
ols_params_dict_u_u = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "d1" : thompson_ols_d1_u_u.mean(axis=1)
        }
'''
ols_params_list = [ols_params_dict_c_c, ols_params_dict_u_c]
ols_params_list_std = [ols_params_dict_c_c_std,ols_params_dict_u_c_std]
#true_coeffs = {'intercept':intercept, 'd1':d1, 'd1x1':d1x1}
bplots.plot_regression_multiple(ax[1], user_count, ['Contextual MAB',
                    'Non-Contextual MAB'],
    ols_params_list, ols_params_list_std,
            simulation_count, batch_size, save_fig=True)


plt.show()

'''
thompson_regrets = pd.read_csv("{}thompson_regrets.csv".format(folder_name),
        index_col=0)
thompson_optimal_action_ratio = pd.read_csv(
        "{}thompson_optimal_action_ratio.csv".format(folder_name), index_col=0)
thompson_ols_intercept = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name), index_col=0)
thompson_ols_d1 = pd.read_csv("{}thompson_ols_d1.csv".format(folder_name),
        index_col=0)
thompson_ols_d1x1 = pd.read_csv("{}thompson_ols_d1x1.csv".format(folder_name),
        index_col=0)
thompson_mse = pd.read_csv("{}thompson_mse.csv".format(folder_name),
        index_col=0, header=[0,1])
thompson_context_action = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name),
        index_col=0, header=[0,1])
thompson_coeff_sign_err = pd.read_csv(
        "{}thompson_coeff_sign_err.csv".format(folder_name),
        index_col=0, header=[0,1])
thompson_bias_in_coeff = pd.read_csv(
        "{}thompson_bias_in_coeff.csv".format(folder_name),
        index_col=0, header=[0,1])

random_regrets = pd.read_csv("{}random_regrets.csv".format(folder_name),
        index_col=0)
random_ols_intercept = pd.read_csv(
        "{}random_ols_intercept.csv".format(folder_name), index_col=0)
random_ols_d1 = pd.read_csv("{}random_ols_d1.csv".format(folder_name),
        index_col=0)
random_ols_d1x1 = pd.read_csv("{}random_ols_d1x1.csv".format(folder_name),
        index_col=0)
random_optimal_action_ratio = pd.read_csv(
        "{}random_optimal_action_ratio.csv".format(folder_name),
        index_col=0, header=[0,1])
random_context_action = pd.read_csv(
        "{}random_context_action.csv".format(folder_name),
        index_col=0, header=[0,1])
'''
#print(thompson_regrets.index.tolist())
#print(thompson_regrets.mean(axis=1))
#print("***************")
#print(thompson_mse.index.tolist())
#print(thompson_mse.columns)

#print(thompson_mse['1','intercept'].iloc[10])

