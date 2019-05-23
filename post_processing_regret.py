from subprocess import check_output
import numpy as numpy
import pandas as pd
import plots.plot_basics as bplots
import matplotlib.pyplot as plt

parent_folder_name = "saved_output/raw_data/"
interaction_size = 'C'
batch_size = 10



# Load data from Small Interaction, MAB Correct, OLS Correct
folder_name_c_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Correct_OLS_Correct/"
thompson_regrets_c_c = pd.read_csv(
        "{}thompson_regrets.csv".format(folder_name_c_c), index_col=0)
context_c_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_c_c), index_col=0,
        header=[0,1])
x_c_c = context_c_c.drop(['x0_d0','x0_d1'], axis=1, level=1).sum(axis=1,level=0)
regret_x1_c_c = x_c_c * thompson_regrets_c_c
regret_x0_c_c = (x_c_c*-1+1) * thompson_regrets_c_c


# Load data from Small Interaction, MAB UnderSpecified, OLS Correct
folder_name_u_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Under_OLS_Correct/"
thompson_regrets_u_c = pd.read_csv(
        "{}thompson_regrets.csv".format(folder_name_u_c), index_col=0)
context_u_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_u_c), index_col=0,
        header=[0,1])
x_u_c = context_u_c.drop(['x0_d0','x0_d1'], axis=1, level=1).sum(axis=1,level=0)
regret_x1_u_c = x_u_c * thompson_regrets_u_c
regret_x0_u_c = (x_u_c*-1+1) * thompson_regrets_u_c

# Load data from Small Interaction, MAB UnderSpecified, OLS UnderSpecified
#folder_name_u_u = parent_folder_name + \
#                    interaction_size+ \
#                    "_MAB_Under_OLS_Under/"


# Load data from Small Interaction, Uniform Sampling, OLS Correct
folder_name_c = parent_folder_name + \
                    interaction_size+ \
                    "_Random_OLS_Correct/"
random_regrets_c = pd.read_csv(
        "{}random_regrets.csv".format(folder_name_c), index_col=0)

random_regrets_c = pd.read_csv(
        "{}random_regrets.csv".format(folder_name_c), index_col=0)
context_c = pd.read_csv(
        "{}random_context_action.csv".format(folder_name_c), index_col=0,
        header=[0,1])
x_c = context_c.drop(['x0_d0','x0_d1'], axis=1, level=1).sum(axis=1,level=0)
regret_x1_c = x_c * random_regrets_c
regret_x0_c = (x_c*-1+1) * random_regrets_c
# Load data from Small Interaction, Uniform Sampling, OLS UnderSpecified
#folder_name_u = parent_folder_name + \
#                    interaction_size+ \
#                    "_Random_OLS_Under/"


user_count = thompson_regrets_c_c.shape[0]
simulation_count = thompson_regrets_c_c.shape[1]
print(thompson_regrets_c_c.shape)
print(thompson_regrets_u_c.shape)
print(random_regrets_c.shape)
#Processing OLS
regret_dict_c_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "r_1" : (thompson_regrets_c_c.mean(axis=1)).iloc[500:750].mean(),
        "r_2" : (thompson_regrets_c_c.mean(axis=1)).iloc[500:750].mean(),
        "r_x0_1":((regret_x0_c_c.sum(axis=1))/(simulation_count-x_c_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x0_2":((regret_x0_c_c.sum(axis=1))/(simulation_count-x_c_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x1_1":((regret_x1_c_c.sum(axis=1))/x_c_c.sum(axis=1)).iloc[500:750].mean(),
        "r_x1_2":((regret_x1_c_c.sum(axis=1))/x_c_c.sum(axis=1)).iloc[500:750].mean()
        }

regret_dict_u_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "r_1" : (thompson_regrets_u_c.mean(axis=1)).iloc[500:750].mean(),
        "r_2" : (thompson_regrets_u_c.mean(axis=1)).iloc[500:750].mean(),
        "r_x0_1":((regret_x0_u_c.sum(axis=1))/(simulation_count-x_u_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x0_2":((regret_x0_u_c.sum(axis=1))/(simulation_count-x_u_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x1_1":((regret_x1_u_c.sum(axis=1))/x_u_c.sum(axis=1)).iloc[500:750].mean(),
        "r_x1_2":((regret_x1_u_c.sum(axis=1))/x_u_c.sum(axis=1)).iloc[500:750].mean()
        }

regret_dict_c = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "r_1" : (random_regrets_c.mean(axis=1)).iloc[500:750].mean(),
        "r_2" : (random_regrets_c.mean(axis=1)).iloc[500:750].mean(),
        "r_x0_1":((regret_x0_c.sum(axis=1))/(simulation_count-x_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x0_2":((regret_x0_c.sum(axis=1))/(simulation_count-x_c.sum(axis=1))).iloc[500:750].mean(),
        "r_x1_1":((regret_x1_c.sum(axis=1))/x_c.sum(axis=1)).iloc[500:750].mean(),
        "r_x1_2":((regret_x1_c.sum(axis=1))/x_c.sum(axis=1)).iloc[500:750].mean()
        }

#print(interaction_size+', MAB correct, OLS correct, regret, 0-250:',
#        round(regret_dict_c_c['r_1'],2) )
print(interaction_size+', MAB correct, OLS correct, regret, 500:750:',
        round(regret_dict_c_c['r_2'],4))
print(interaction_size+', MAB correct, OLS correct, regret(X=0), 500:750:',
        round(regret_dict_c_c['r_x0_2'],4))
print(interaction_size+', MAB correct, OLS correct, regret(X=1), 500:750:',
        round(regret_dict_c_c['r_x1_2'],4))
print("************************************************")
#print(interaction_size+', MAB UnderSpecified, OLS correct, regret, 0-250:',
#        round(regret_dict_u_c['r_1'],2))
print(interaction_size+', MAB UnderSpecified, OLS correct, regret, 500:750:',
        round(regret_dict_u_c['r_2'],4))
print(interaction_size+', MAB UnderSpecified, OLS correct, regret(X=0), 500:750:',
        round(regret_dict_u_c['r_x0_2'],4))
print(interaction_size+', MAB UnderSpecified, OLS correct, regret(X=1), 500:750:',
        round(regret_dict_u_c['r_x1_2'],4))
print("************************************************")
#print(interaction_size+', Random, OLS correct, regret, 0-250:',
#        round(regret_dict_c['r_1'],2))
print(interaction_size+', Random, OLS correct, regret, 500:750:',
        round(regret_dict_c['r_2'],4))
print(interaction_size+', Random, OLS correct, regret(X=0), 500:750:',
        round(regret_dict_c['r_x0_2'],4))
print(interaction_size+', Random, OLS correct, regret(X=1), 500:750:',
        round(regret_dict_c['r_x1_2'],4))
print("************************************************")
