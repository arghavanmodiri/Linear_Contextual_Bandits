from subprocess import check_output
import numpy as numpy
import pandas as pd
import plots.plot_basics as bplots
import matplotlib.pyplot as plt

'''
Each CSV file which is the output of the simulations contain the raw data of
all iterations. This file calculates the value of OLS coefficients averaged over all the simulation runs. Then, the result of each quartile will be printed.
(250th, 500th, 750th, and 1000th)

Note1: If the same simulation has been run for multiple times and the calculation of the overall average is required; first merge all the files with merge_csv.py, then calculate the average using this file.

Note2: In this file, we assume the number of users is 1000.


parent_folder_name: address of the parent folder that contains multiple
subfolders. Each subfolder contains the following folders that we want to
process:
<interaction_size>_MAB_Correct_OLS_Correct
<interaction_size>_MAB_Under_OLS_Correct
<interaction_size>_Random_OLS_Correct
The names of the desired folders are changed manually to the above mentioned
ones. The original names of the folders showed tha date and the time of being
generated.

interaction_size: The parent folder contains the results of all the simulations with different interaction sizes. The subfolders with different interaction sizes are differentiated with the prefix. interaction_size is the 
corresponds to this prefix

batch_size, intercept, d1, d1x1: The same values that used to generate the data

The result will be printed.
'''

parent_folder_name = "saved_output/raw_data/"
interaction_size = 'C'
batch_size = 10
intercept = 0
d1 = 0.3
d1x1=-0.6
drop_list = []
for i in range(1500,2500):
    drop_list.append(i)

# Load data from Small Interaction, MAB Correct, OLS Correct
folder_name_c_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Correct_OLS_Correct/"
thompson_ols_intercept_c_c = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_c_c), index_col=0)
thompson_ols_d1_c_c = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_c_c), index_col=0)
thompson_ols_d1x1_c_c = pd.read_csv(
        "{}thompson_ols_d1x1.csv".format(folder_name_c_c), index_col=0)

thompson_ols_d1_c_c.columns = thompson_ols_d1_c_c.columns.astype(int)
thompson_ols_d1_c_c=thompson_ols_d1_c_c.drop(drop_list, axis=1)
thompson_ols_d1x1_c_c.columns = thompson_ols_d1x1_c_c.columns.astype(int)
thompson_ols_d1x1_c_c=thompson_ols_d1x1_c_c.drop(drop_list, axis=1)


# Load data from Small Interaction, MAB UnderSpecified, OLS Correct
folder_name_u_c = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Under_OLS_Correct/"

thompson_ols_intercept_u_c = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_u_c), index_col=0)
thompson_ols_d1_u_c = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_u_c), index_col=0)
thompson_ols_d1x1_u_c = pd.read_csv(
        "{}thompson_ols_d1x1.csv".format(folder_name_u_c), index_col=0)

thompson_ols_d1_u_c.columns = thompson_ols_d1_u_c.columns.astype(int)
thompson_ols_d1_u_c=thompson_ols_d1_u_c.drop(drop_list, axis=1)
thompson_ols_d1x1_u_c.columns = thompson_ols_d1x1_u_c.columns.astype(int)
thompson_ols_d1x1_u_c=thompson_ols_d1x1_u_c.drop(drop_list, axis=1)

# Load data from Small Interaction, MAB UnderSpecified, OLS UnderSpecified
'''
folder_name_u_u = parent_folder_name + \
                    interaction_size+ \
                    "_MAB_Under_OLS_Under/"
thompson_ols_intercept_u_u = pd.read_csv(
        "{}thompson_ols_intercept.csv".format(folder_name_u_u), index_col=0)
thompson_ols_d1_u_u = pd.read_csv(
        "{}thompson_ols_d1.csv".format(folder_name_u_u), index_col=0)
'''

# Load data from Small Interaction, Uniform Sampling, OLS Correct
folder_name_c = parent_folder_name + \
                    interaction_size+ \
                    "_Random_OLS_Correct/"
random_ols_intercept_c = pd.read_csv(
        "{}random_ols_intercept.csv".format(folder_name_c), index_col=0)
random_ols_d1_c = pd.read_csv(
        "{}random_ols_d1.csv".format(folder_name_c), index_col=0)
random_ols_d1x1_c = pd.read_csv(
        "{}random_ols_d1x1.csv".format(folder_name_c), index_col=0)
thompson_ols_intercept_c_c.columns = thompson_ols_intercept_c_c.columns.astype(int)
random_ols_d1_c.columns = random_ols_d1_c.columns.astype(int)
random_ols_d1_c=random_ols_d1_c.drop(drop_list, axis=1)
random_ols_d1x1_c.columns = random_ols_d1x1_c.columns.astype(int)
random_ols_d1x1_c=random_ols_d1x1_c.drop(drop_list, axis=1)


user_count = thompson_ols_intercept_c_c.shape[0]
simulation_count = thompson_ols_intercept_c_c.shape[1]
print(thompson_ols_intercept_c_c.shape)
print(random_ols_intercept_c.shape)
print(thompson_ols_intercept_u_c.shape)
#Processing OLS
ols_params_dict_c_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "i_1" : (thompson_ols_intercept_c_c.mean(axis=1)).iloc[250-1],
        "i_2" : (thompson_ols_intercept_c_c.mean(axis=1)).iloc[500-1],
        "i_3" : (thompson_ols_intercept_c_c.mean(axis=1)).iloc[750-1],
        "i_4" : (thompson_ols_intercept_c_c.mean(axis=1)).iloc[1000-1],
        "d1_1" : (thompson_ols_d1_c_c.mean(axis=1) - d1).iloc[250-1],
        "d1_2" : (thompson_ols_d1_c_c.mean(axis=1) - d1).iloc[500-1],
        "d1_3" : (thompson_ols_d1_c_c.mean(axis=1) - d1).iloc[750-1],
        "d1_4" : (thompson_ols_d1_c_c.mean(axis=1) - d1).iloc[1000-1],
        "d1x1_1":(thompson_ols_d1x1_c_c.mean(axis=1)- d1x1).iloc[250-1],
        "d1x1_2":(thompson_ols_d1x1_c_c.mean(axis=1)- d1x1).iloc[500-1],
        "d1x1_3":(thompson_ols_d1x1_c_c.mean(axis=1)- d1x1).iloc[750-1],
        "d1x1_4":(thompson_ols_d1x1_c_c.mean(axis=1)- d1x1).iloc[1000-1]
        }

ols_params_dict_u_c = {
        #"intercept" : thompson_ols_intercept_u_c.mean(axis=1),
        "i_1" : (thompson_ols_intercept_u_c.mean(axis=1)).iloc[250-1],
        "i_2" : (thompson_ols_intercept_u_c.mean(axis=1)).iloc[500-1],
        "i_3" : (thompson_ols_intercept_u_c.mean(axis=1)).iloc[750-1],
        "i_4" : (thompson_ols_intercept_u_c.mean(axis=1)).iloc[1000-1],
        "d1_1" : (thompson_ols_d1_u_c.mean(axis=1) - d1).iloc[250-1],
        "d1_2" : (thompson_ols_d1_u_c.mean(axis=1) - d1).iloc[500-1],
        "d1_3" : (thompson_ols_d1_u_c.mean(axis=1) - d1).iloc[750-1],
        "d1_4" : (thompson_ols_d1_u_c.mean(axis=1) - d1).iloc[1000-1],
        "d1x1_1":(thompson_ols_d1x1_u_c.mean(axis=1)- d1x1).iloc[250-1],
        "d1x1_2":(thompson_ols_d1x1_u_c.mean(axis=1)- d1x1).iloc[500-1],
        "d1x1_3":(thompson_ols_d1x1_u_c.mean(axis=1)- d1x1).iloc[750-1],
        "d1x1_4":(thompson_ols_d1x1_u_c.mean(axis=1)- d1x1).iloc[1000-1]
        }
'''
ols_params_dict_u_u = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "d1_1" : (thompson_ols_d1_u_u.mean(axis=1) - (d1+d1x1/2)).iloc[500:750].mean(),
        "d1_2" : (thompson_ols_d1_u_u.mean(axis=1) - (d1+d1x1/2)).iloc[750:1000].mean()
        }
'''

ols_params_dict_c = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "i_1" : (random_ols_intercept_c.mean(axis=1)).iloc[250-1],
        "i_2" : (random_ols_intercept_c.mean(axis=1)).iloc[500-1],
        "i_3" : (random_ols_intercept_c.mean(axis=1)).iloc[750-1],
        "i_4" : (random_ols_intercept_c.mean(axis=1)).iloc[1000-1],
        "d1_1" : (random_ols_d1_c.mean(axis=1) - d1).iloc[250-1],
        "d1_2" : (random_ols_d1_c.mean(axis=1) - d1).iloc[500-1],
        "d1_3" : (random_ols_d1_c.mean(axis=1) - d1).iloc[750-1],
        "d1_4" : (random_ols_d1_c.mean(axis=1) - d1).iloc[1000-1],
        "d1x1_1":(random_ols_d1x1_c.mean(axis=1)- d1x1).iloc[250-1],
        "d1x1_2":(random_ols_d1x1_c.mean(axis=1)- d1x1).iloc[500-1],
        "d1x1_3":(random_ols_d1x1_c.mean(axis=1)- d1x1).iloc[750-1],
        "d1x1_4":(random_ols_d1x1_c.mean(axis=1)- d1x1).iloc[1000-1]
        }
'''
ols_params_dict_u = {
        #"intercept" : thompson_ols_intercept_u_u.mean(axis=1),
        "d1_1" : (random_ols_d1_u.mean(axis=1) - (d1+d1x1/2)).iloc[500:750].mean(),
        "d1_2" : (random_ols_d1_u.mean(axis=1) - (d1+d1x1/2)).iloc[750:1000].mean()
        }
'''
print(interaction_size+', MAB correct, OLS correct, d1, 250:',
        round(ols_params_dict_c_c['i_1'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 500:',
        round(ols_params_dict_c_c['i_2'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 750:',
        round(ols_params_dict_c_c['i_3'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 1000:',
        round(ols_params_dict_c_c['i_4'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 250:',
        round(ols_params_dict_u_c['i_1'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 500:',
        round(ols_params_dict_u_c['i_2'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 750:',
        round(ols_params_dict_u_c['i_3'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 1000:',
        round(ols_params_dict_u_c['i_4'],4 ))
print(interaction_size+', Random, OLS correct, d1, 250:',
        round(ols_params_dict_c['i_1'],4 ))
print(interaction_size+', Random, OLS correct, d1, 500:',
        round(ols_params_dict_c['i_2'],4 ))
print(interaction_size+', Random, OLS correct, d1, 750:',
        round(ols_params_dict_c['i_3'],4 ))
print(interaction_size+', Random, OLS correct, d1, 1000:',
        round(ols_params_dict_c['i_4'],4 ))

print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(interaction_size+', MAB correct, OLS correct, d1, 250:',
        round(ols_params_dict_c_c['d1_1'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 500:',
        round(ols_params_dict_c_c['d1_2'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 750:',
        round(ols_params_dict_c_c['d1_3'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1, 1000:',
        round(ols_params_dict_c_c['d1_4'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1x1, 250:',
        round(ols_params_dict_c_c['d1x1_1'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1x1, 500:',
        round(ols_params_dict_c_c['d1x1_2'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1x1, 750:',
        round(ols_params_dict_c_c['d1x1_3'],4 ))
print(interaction_size+', MAB correct, OLS correct, d1x1, 1000:',
        round(ols_params_dict_c_c['d1x1_4'],4 ))
print("************************************************")
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 250:',
        round(ols_params_dict_u_c['d1_1'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 500:',
        round(ols_params_dict_u_c['d1_2'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 750:',
        round(ols_params_dict_u_c['d1_3'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1, 1000:',
        round(ols_params_dict_u_c['d1_4'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1x1, 250:',
        round(ols_params_dict_u_c['d1x1_1'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1x1, 500:',
        round(ols_params_dict_u_c['d1x1_2'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1x1, 750:',
        round(ols_params_dict_u_c['d1x1_3'],4 ))
print(interaction_size+', MAB UnderSpecified, OLS correct, d1x1, 1000:',
        round(ols_params_dict_u_c['d1x1_4'],4 ))
print("************************************************")


print("************************************************")
print(interaction_size+', Random, OLS correct, d1, 250:',
        round(ols_params_dict_c['d1_1'],4 ))
print(interaction_size+', Random, OLS correct, d1, 500:',
        round(ols_params_dict_c['d1_2'],4 ))
print(interaction_size+', Random, OLS correct, d1, 750:',
        round(ols_params_dict_c['d1_3'],4 ))
print(interaction_size+', Random, OLS correct, d1, 1000:',
        round(ols_params_dict_c['d1_4'],4 ))
print(interaction_size+', Random, OLS correct, d1x1, 250:',
        round(ols_params_dict_c['d1x1_1'],4 ))
print(interaction_size+', Random, OLS correct, d1x1, 500:',
        round(ols_params_dict_c['d1x1_2'],4 ))
print(interaction_size+', Random, OLS correct, d1x1, 750:',
        round(ols_params_dict_c['d1x1_3'],4 ))
print(interaction_size+', Random, OLS correct, d1x1, 1000:',
        round(ols_params_dict_c['d1x1_4'],4 ))
print("************************************************")
