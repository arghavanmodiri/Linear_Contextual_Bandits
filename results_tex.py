'''
Purpose: Replace placeholds in TeX tables from a provided data file

Date Started: March 13, 2019

Data Inputs:
- CSV file containing placeholders as column names with the
values being the result that gets subtituted into TeX file

Notes:
cols = ["d1_bias_q1", "d1x1_bias_q1", "d1_bias_q4", "d1x1_bias_q4", "df_regret_q1", "df_regret_q4",
                 "prop_x0_optimal_q1", "prop_x0_optimal_q4", "prop_x1_optimal_q1", "prop_x1_optimal_q4"]
                 
sim_names = ["_SCC", "_LCC", "_LUC"]

['d1_bias_q1_SCC', 'd1x1_bias_q1_SCC', 'd1_bias_q4_SCC', 'd1x1_bias_q4_SCC', 'df_regret_q1_SCC', 'df_regret_q4_SCC',
'prop_x0_optimal_q1_SCC', 'prop_x0_optimal_q4_SCC', 'prop_x1_optimal_q1_SCC', 'prop_x1_optimal_q4_SCC']

['d1_bias_q1_LCC', 'd1x1_bias_q1_LCC', 'd1_bias_q4_LCC', 'd1x1_bias_q4_LCC', 'df_regret_q1_LCC', 'df_regret_q4_LCC',
'prop_x0_optimal_q1_LCC', 'prop_x0_optimal_q4_LCC', 'prop_x1_optimal_q1_LCC', 'prop_x1_optimal_q4_LCC']

['d1_bias_q1_LUC', 'd1_bias_q4_LUC', 'df_regret_q1_LUC', 'df_regret_q4_LUC', 'prop_x0_optimal_q1_LUC',
'prop_x0_optimal_q4_LUC', 'prop_x1_optimal_q1_LUC', 'prop_x1_optimal_q4_LUC']

'''

import os
import sys
import pandas as pd
import numpy as np

# Load results data set

# Read TeX template file into a string
temp_path = "/Users/hammadshaikh/Documents/University of Toronto/2018 RA/Hetroscedastic model/No Theta Hetro model/MMP2VarCompTemplate.tex"

# Simulations
'''sim_list = ["SmallMABCorrectFitCorrect", "SmallMABUnderFitCorrect", "LargeMABCorrectFitCorrect", "LargeMABUnderFitCorrect",
            "SmallMABUnderFitUnder","LargeMABUnderFitUnder","SmallRandomCorrect", "LargeRandomCorrect",
            "SmallRandomUnder", "LargeRandomUnder", "SmallUniformPolicy", "LargeUniformPolicy"]
sim_names = ["_SCC", "_SUC", "_LCC", "_LUC", "_SUU", "_LUU", "_SRC", "_LRC", "_SRU", "_LRU", "_SR","_LR"]'''

#sim_list = ["SimCCorrectMABFitCorrect", "SimCUnderMABFitCorrect", "SimCRandomCorrect",  "SimCUniformPolicy"]
#sim_names = ["_CSCC", "_CSUC", "_CSRC", "_CSR"]

# Simulations C and D
'''
sim_list = ["SimCCorrectMABFitCorrect", "SimCUnderMABFitCorrect", "SimCRandomCorrect",  "SimCUniformPolicy",
            "SimDCorrectMABFitCorrect", "SimDUnderMABFitCorrect", "SimDRandomCorrect", "SimDUniformPolicy",
            "SimCCorrectMABFitCorrect", "SimCUnderMABFitCorrect", "SimCRandomCorrect",
            "SimDCorrectMABFitCorrect", "SimDUnderMABFitCorrect", "SimDRandomCorrect"]
sim_names = ["_CSCC", "_CSUC", "_CSRC", "_CSR","_DLCC", "_DLUC", "_DLRC", "_DLR",
             "_CSCC", "_CSUC", "_CSRC", "_DLCC", "_DLUC", "_DLRC"]'''

sim_list = ["MMP2VC_(-1,0,1,9)", "MMP2VC_(-1,0,2,9)", "MMP2VC_(-1,0,1,11)", "MMP2VC_(-1,0,9,11)", "MMP2VC_(-1,0,1,2,9)", "MMP2VC_(-1,0,1,9,11)", "MMP2VC_(-1,1,9,11,13)",
            "MMP2VC_(-1,0,1,2,3)", "MMP2VC_(-1,9,11,13)", "MMP2VC_(1,9,11,13)"]
#sim_list = ["MMP2VC_(-1,0,1,9)"]
count = 0

# Loop over simulations
for sim_count in range(len(sim_list)):

    # Read template as string
    with open (temp_path, "r") as tpl_file:
        tpl_lines = tpl_file.read()

    # Open data set with results
    result_path = "/Users/hammadshaikh/Documents/University of Toronto/2018 RA/Hetroscedastic model/No Theta Hetro model/" + str(sim_list[sim_count]) 

    '''if count < 8:
        df_results = pd.read_csv(result_path + "BanditSimResults" + str(sim_names[sim_count]) + ".csv")
    else:
        df_results = pd.read_csv(result_path + "PropOLSOptimal" + str(sim_names[sim_count]) + ".csv")'''

    df_results = pd.read_csv(result_path + ".csv")

    # Replace TeX parameters with data
    col_names = list(df_results)

    # Loop over variables to replace
    for col in col_names:
        tpl_lines = tpl_lines.replace(col, " " + str(np.round(df_results[col][0], 4)))

    # Output TeX file with results
    write_path = "/Users/hammadshaikh/Documents/University of Toronto/2018 RA/Hetroscedastic model/No Theta Hetro model/MMP2VarCompResults.tex"
    with open (write_path, "w") as output_file:
        output_file.write(tpl_lines)

    # Update template path
    temp_path = write_path

    # Increment counter
    count += 1

    

    
