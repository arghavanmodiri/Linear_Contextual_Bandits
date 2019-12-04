#!/usr/bin/env python
import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
import true_hypo_models as models
import policies.thompson_sampling_lasso as thompson_lasso
import policies.thompson_sampling_nig as thompson_nig
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
from datetime import date
from datetime import datetime


def main(input_dict, mode=None):
    hypo_params = input_dict['hypo_model_params']
    a_pre = input_dict['NIG_priors']['a']
    b_pre = input_dict['NIG_priors']['b']
    #Hammad: Bias Correction
    mean_pre = np.zeros(len(hypo_params[1]))
    cov_pre = np.identity(len(hypo_params[1]))

    thompson_output = thompson_lasso.draw_posterior_sample_lasso(hypo_params[1], mean_pre, cov_pre, a_pre, b_pre)
    print(thompson_output)


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