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
import diamante.diamante_true_hypo_models as dmodels
import diamante.diamante_thompson_sampling_nig as dthompson
from diamante.diamante_training_models import training_bandit_model as dtbm
from shutil import copy2
from datetime import date
from datetime import datetime
from training_models import training_bandit_model as tbm

TODAY = date.today()
NOW = datetime.now()

np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_columns', 30)
def main(input_dict, sim_name='test.json', mode=None):
    """start the model"""
    '''
    The simulator will start by passing the json file. If in the json file,
    a path for "context_csv" is included, all contextual variables and their
    values will be read from csv files (the number of users will be equal
    to the number of rows in csv file). If "context_csv" is not defined in the
    json file, the contextual variables will be automatically created according
    the variables defined in "true_coeff". In this mode, the number of users to
    be simulated needs to be passed with "user_count" variable.
    '''

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s')


    data = pd.read_csv("input_files/merged.csv") 
    context_vars = data.columns
    logging.info("context_vars type : {}".format(type(context_vars)))
    logging.info("context_vars : {}".format(context_vars))

    bandit_arms = input_dict['action_space']
    experiment_vars = input_dict['experiment_vars']

    ## Setting the training mode (hypo model) parameters
    hypo_params = input_dict['hypo_params']
    ## Setting the simulation parameters



    #bandit_model = tbm(user_count, 10, experiment_vars, bandit_arms, '',
    #                False, hypo_params)
    a_pre = input_dict['NIG_priors']['a']
    b_pre = input_dict['NIG_priors']['b']
    #Hammad: Bias Correction
    mean_pre = np.zeros(len(hypo_params))
    cov_pre = np.identity(len(hypo_params))
    lst = ['prediction', 'diamante-feedback-steps-encourage',
        'diamante-feedback-steps',  'diamante-feedback-relative',
        'diamante-feedback-relative-steps', 'diamante-adaptive-opportunity',
        'diamante-adaptive-benefit',    'diamante-adaptive-self-efficacy',
        'time-16:30:00-19:00:00', 'time-14:00:00-16:30:00', 'time-11:30:00-14:00:00',
        'ind']
    users_context = data.drop(lst, axis=1)
    users_context = users_context.set_index(['Date', 'Participant'])
    feature_names = users_context.columns
    try:
        scaling_factors = np.array(input_dict['scaling_factors']).reshape((2, -1))
    except KeyError:
        scaling_factors = np.vstack((np.ones((len(feature_names))), np.zeros((len(feature_names)))))
    try:
        mu = np.array(input_dict['means']).reshape((3, -1))
    except KeyError:
        mu = np.zeros((3, len(feature_names)))
    try:
        std = np.array(input_dict['stds']).reshape((3, -1))
    except KeyError:
        std = np.ones((3, len(feature_names)))
    try:
        weighting_coeffs = np.array(input_dict['weighting_coeffs']).reshape((3, -1))
    except KeyError:
        weighting_coeffs = np.ones((3, len(feature_names)))/3

    all_features_list_normalize= normalize(users_context, scaling_factors, mu, std, weighting_coeffs)

    data = data.set_index(['Date', 'Participant'])
    data_new = pd.concat([all_features_list_normalize, data[lst]], axis=1, sort=False, join='inner')
    


    #Step 3: Calls the sampling policy to select action for each user


    X = dmodels.calculate_hypo_regressors(hypo_params, data)
    [a_post, b_post, cov_post, mean_post] = dthompson.calculate_posteriors_nig(data['prediction'],
                                        X, mean_pre, cov_pre, a_pre,b_pre)

    #print(a_post)
    logging.info("a_post {}".format(a_post))
    logging.info("b_post {}".format(b_post))
    #logging.info("cov_post {}".format(cov_post))
    #logging.info("mean_post {}".format(mean_post))
    #logging.info("mean_post {}".format(len(mean_post)))
    #logging.info("mean_post {}".format(mean_post.shape))
    cov_df = pd.DataFrame(cov_post)
    mean_df = pd.DataFrame(mean_post)
    cov_df.to_csv('saved_output/cov.csv')
    mean_df.to_csv('saved_output/mean.csv')
    #print(b_post)
    #print(cov_post)
    #print(mean_post)



    #regret
    #optimal_action_ratio
    #beta_thompson_coeffs
    #mse
    #coeff_sign_error_per
    #Bias Correction

    # Add Random Policy




def normalize(dataset, scaling_factors, mu, std, weighting_coeffs):
    """

    Normalizes and scales variables in the dataset appropriately, according
    to n1*(uniform random data) + n2*(10 people in December) + n3*(team-decided
    prior). n1, n2, and n3 are referred to as weighting coefficients. Scaling factors
    will then be applied to normalized variables following cX + d, for each feature X.

    :param dataset: np array of shape user_days * features
    :param scaling_factors: np array of shape 2*(features), represents cx+d scaling.
    First row is c value, second row is d value.
    :param mu: np matrix of shape 3*features for means of each feature
    First row is uniform random data, second row is from 10 people in December, third row is estimates from team.
    :param std: np matrix of shape 3*features for standard deviations of each feature
    First row is uniform random data, second row is from 10 people in December, third row is estimates from team.
    :param weighting_coeffs: np matrix of shape 3*features for mixing means and standard deviations
    First row is uniform random data, second row is from 10 people in December, third row is estimates from team.
    """

    try:
        weighted_mu = np.sum(np.multiply(weighting_coeffs, mu), axis=0)
        weighted_std = np.sum(np.multiply(weighting_coeffs, std), axis=0)

    except:
        weighted_mu = np.zeros((1, dataset.shape[1]))
        weighted_std = np.ones((1, dataset.shape[1]))


    try:
        new_dataset = np.divide(np.subtract(dataset, weighted_mu), weighted_std) # pylint: disable=assignment-from-no-return
        new_dataset = np.multiply(scaling_factors[0, :], new_dataset) + scaling_factors[1, :]

        return new_dataset
    except:

        return dataset




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('input_file', metavar='input_file', type=str, nargs=1,
                        help='Name of the json config file')
    args = parser.parse_args()

    if (len(args.input_file) != 1) or (not args.input_file[0].endswith(".json")):
        print( "Error: Function should have only one input, name of the JSON config file." )
        sys.exit(1)

    save_output_folder = 'saved_output/raw_data_'+str(TODAY)+'_'+str(NOW.hour)+'-'+str(NOW.minute)+'-'+str(NOW.second)+"/"
    if not os.path.exists(save_output_folder):
        os.mkdir(save_output_folder)
    copy2('{}'.format(args.input_file[0]), '{}/'.format(save_output_folder))

    input_data = args.input_file[0]
    input_data = json.load(open(input_data))
    main(input_data, sim_name=str(args.input_file[0]))

