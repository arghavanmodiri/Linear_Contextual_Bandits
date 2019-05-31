import glob
import os
from subprocess import check_output
import numpy as numpy
import pandas as pd

'''
This file merges all the files inside folder1 and folder2 that have the same
name. The merged results will be written into CSV files and stored inside
parent_folder_name. After the results have been saved, they should be
transfered to the desired destination, otherwise they will be replaced by the second run of this file.

'''

parent_folder_name = "saved_output/raw_data/to merge/"
folder1 = parent_folder_name + 'first/'
folder2 = parent_folder_name + 'second/'
all_files_1 = os.listdir(folder1)
#all_files_1 = glob.glob(folder1 + "/*.csv")
all_files_2 = os.listdir(folder2)
print(all_files_2)

for filename in all_files_2:
    if filename.endswith(".csv"):
        print("*******************")
        if( ('bias_in_coeff' in filename) or ('coeff_sign_err' in filename) or
            ('context_action' in filename) or ('mse' in filename)):
            #for two header files
            print("2 header : ",filename)
            file1 = pd.read_csv(
                "{}{}".format(folder1,filename), index_col=0,header=[0,1])
            file2 = pd.read_csv(
                "{}{}".format(folder2,filename), index_col=0,header=[0,1])
            file1.shape[1]
            
            #file2.columns = [file2.columns.get_level_values(0) + 500, file2.columns.get_level_values(1)]
            add_column_index = file1.columns.get_level_values(0).nunique()
            file2.columns = [file2.columns.get_level_values(0).astype(int), file2.columns.get_level_values(1)]
            file2.columns = [file2.columns.get_level_values(0) + add_column_index,
                            file2.columns.get_level_values(1)]
        else:
            #for one header files
            print("1 header : ",filename)
            file1 = pd.read_csv(
            "{}{}".format(folder1,filename), index_col=0)
            file2 = pd.read_csv(
                "{}{}".format(folder2,filename), index_col=0)
            add_column_index = file1.shape[1]
            file2.columns = file2.columns.astype(int)
            file2.columns = file2.columns + add_column_index
        
        print(file1.shape)
        print(file2.shape)
        file2_merged = pd.concat([file1,file2], axis=1)
        file2_merged.to_csv('{}{}'.format(
                                parent_folder_name,filename))
        print("******** ",file2_merged.shape)
        print("*******************")

    #with open(filename, 'a') as f:
    #    df.to_csv(f, header=False)