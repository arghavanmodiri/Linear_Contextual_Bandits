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
context_c_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_c_c), index_col=0,
        header=[0,1])
context_c_c_sum = context_c_c.sum(axis=1, level=1)


# Load data from Small Interaction, MAB UnderSpecified, OLS Correct
folder_name_u_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Under_OLS_Correct/"
context_u_c = pd.read_csv(
        "{}thompson_context_action.csv".format(folder_name_u_c), index_col=0,
        header=[0,1])
context_u_c_sum = context_u_c.sum(axis=1, level=1)

# Load data from Small Interaction, MAB UnderSpecified, OLS UnderSpecified
#folder_name_u_u = parent_folder_name + \
#                    interaction_size+ \
#                    "_MAB_Under_OLS_Under/"


# Load data from Small Interaction, Uniform Sampling, OLS Correct
folder_name_c = parent_folder_name + \
                    interaction_size+ \
                    "_Random_OLS_Correct/"
context_c = pd.read_csv(
        "{}random_context_action.csv".format(folder_name_c), index_col=0,
        header=[0,1])
context_c_sum = context_c.sum(axis=1, level=1)

# Load data from Small Interaction, Uniform Sampling, OLS UnderSpecified
#folder_name_u = parent_folder_name + \
#                    interaction_size+ \
#                    "_Random_OLS_Under/"

user_count = context_c_c.shape[0]
simulation_count = context_c_c.shape[1]

df =pd.DataFrame()
df['x0_d0'] = context_c_c_sum.cumsum().sum(axis=1)
df['x0_d1'] = context_c_c_sum.cumsum().sum(axis=1)
df['x1_d0'] = context_c_c_sum.cumsum().sum(axis=1)
df['x1_d1'] = context_c_c_sum.cumsum().sum(axis=1)
print((context_c_c_sum.cumsum()/df).iloc[500:750].mean())

df1 =pd.DataFrame()
df1['x0_d0'] = context_u_c_sum.cumsum().sum(axis=1)
df1['x0_d1'] = context_u_c_sum.cumsum().sum(axis=1)
df1['x1_d0'] = context_u_c_sum.cumsum().sum(axis=1)
df1['x1_d1'] = context_u_c_sum.cumsum().sum(axis=1)
print((context_u_c_sum.cumsum()/df1).iloc[500:750].mean())

df2 =pd.DataFrame()
df2['x0_d0'] = context_c_sum.cumsum().sum(axis=1)
df2['x0_d1'] = context_c_sum.cumsum().sum(axis=1)
df2['x1_d0'] = context_c_sum.cumsum().sum(axis=1)
df2['x1_d1'] = context_c_sum.cumsum().sum(axis=1)
print((context_c_sum.cumsum()/df2).iloc[500:750].mean())

#print(context_u_c_sum.cumsum()/context_c_c_sum.cumsum().sum(axis=1))
#print(context_c_sum.cumsum()/context_c_c_sum.cumsum().sum(axis=1))

print(context_c_c.shape)
print(context_u_c.shape)
print(context_c.shape)

#Processing OLS
optimal_action_c_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "ratio_1" : ((context_c_c_sum['x0_d1']+context_c_c_sum['x1_d0'])/context_c_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_2" : ((context_c_c_sum['x0_d1']+context_c_c_sum['x1_d0'])/context_c_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_x0_1":((context_c_c_sum['x0_d1'])/(context_c_c_sum['x0_d1']+ context_c_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x0_2":((context_c_c_sum['x0_d1'])/(context_c_c_sum['x0_d1']+ context_c_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x1_1":((context_c_c_sum['x1_d0'])/(context_c_c_sum['x1_d1']+ context_c_c_sum['x1_d0'])).iloc[500:750].mean(),
        "ratio_x1_2":((context_c_c_sum['x1_d0'])/(context_c_c_sum['x1_d1']+ context_c_c_sum['x1_d0'])).iloc[500:750].mean()
        }

optimal_action_u_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "ratio_1" : ((context_u_c_sum['x0_d1']+context_u_c_sum['x1_d0'])/context_u_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_2" : ((context_u_c_sum['x0_d1']+context_u_c_sum['x1_d0'])/context_u_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_x0_1":((context_u_c_sum['x0_d1'])/(context_u_c_sum['x0_d1']+ context_u_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x0_2":((context_u_c_sum['x0_d1'])/(context_u_c_sum['x0_d1']+ context_u_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x1_1":((context_u_c_sum['x1_d0'])/(context_u_c_sum['x1_d1']+ context_u_c_sum['x1_d0'])).iloc[500:750].mean(),
        "ratio_x1_2":((context_u_c_sum['x1_d0'])/(context_u_c_sum['x1_d1']+ context_u_c_sum['x1_d0'])).iloc[500:750].mean()
        }

optimal_action_c = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "ratio_1" : ((context_c_sum['x0_d1']+context_c_sum['x1_d0'])/context_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_2" : ((context_c_sum['x0_d1']+context_c_sum['x1_d0'])/context_c_sum.sum(axis=1)).iloc[500:750].mean(),
        "ratio_x0_1":((context_c_sum['x0_d1'])/(context_c_sum['x0_d1']+ context_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x0_2":((context_c_sum['x0_d1'])/(context_c_sum['x0_d1']+ context_c_sum['x0_d0'])).iloc[500:750].mean(),
        "ratio_x1_1":((context_c_sum['x1_d0'])/(context_c_sum['x1_d1']+ context_c_sum['x1_d0'])).iloc[500:750].mean(),
        "ratio_x1_2":((context_c_sum['x1_d0'])/(context_c_sum['x1_d1']+ context_c_sum['x1_d0'])).iloc[500:750].mean()
        }

#print(interaction_size+', MAB correct, OLS correct, regret, 0-250:',
#        round(regret_dict_c_c['r_1'],2) )
print(interaction_size+', MAB correct, OLS correct, Opt. Action ratio, 500:750:',
        round(optimal_action_c_c['ratio_2'],4))
print(interaction_size+', MAB correct, OLS correct,  Opt. Action ratio(X=0), 500:750:',
        round(optimal_action_c_c['ratio_x0_2'],4))
print(interaction_size+', MAB correct, OLS correct,  Opt. Action ratio(X=1), 500:750:',
        round(optimal_action_c_c['ratio_x1_2'],4))

print("************************************************")
#print(interaction_size+', MAB UnderSpecified, OLS correct, regret, 0-250:',
#        round(regret_dict_u_c['r_1'],2))
print(interaction_size+', MAB UnderSpecified, OLS correct,  Opt. Action ratio, 500:750:',
        round(optimal_action_u_c['ratio_2'],2))
print(interaction_size+', MAB UnderSpecified, OLS correct,  Opt. Action ratio(X=0), 500:750:',
        round(optimal_action_u_c['ratio_x0_2'],2))
print(interaction_size+', MAB UnderSpecified, OLS correct,  Opt. Action ratio(X=1), 500:750:',
        round(optimal_action_u_c['ratio_x1_2'],2))
print("************************************************")
#print(interaction_size+', Random, OLS correct, regret, 0-250:',
#        round(optimal_action_c['r_1'],2))
print(interaction_size+', Random, OLS correct,  Opt. Action ratio, 500:750:',
        round(optimal_action_c['ratio_2'],2))
print(interaction_size+', Random, OLS correct,  Opt. Action ratio(X=0), 500:750:',
        round(optimal_action_c['ratio_x0_2'],2))
print(interaction_size+', Random, OLS correct,  Opt. Action ratio(X=1), 500:750:',
        round(optimal_action_c['ratio_x1_2'],2))
print("************************************************")
