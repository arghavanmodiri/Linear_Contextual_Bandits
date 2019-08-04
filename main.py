#!/usr/bin/env python
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import true_hypo_models as models
import policies.thompson_sampling_nig as thompson
import policies.random_sampling as random
import plots.plot_basics as bplots
import matplotlib.pyplot as plt
#import seaborn as sns
import statsmodels.formula.api as sm
from datetime import date
from datetime import datetime
from results import *

TODAY = date.today()
NOW = datetime.now()

def main(input_dict, mode=None):
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
 
	true_model_params = input_dict['true_model_params']
	hypo_params = [input_dict['hypo_model_params']]
	hypo_params_no_interaction = [input_dict['hypo_params_no_interaction']]
	noise_stats = true_model_params['noise']
	true_coeff = true_model_params['true_coeff']
	context_vars = true_model_params['context_vars']
	experiment_vars = true_model_params['experiment_vars']
	true_coeff_list = list(true_coeff.values())
	bandit_arms = models.find_possible_actions(experiment_vars)
	print(experiment_vars)
	print(bandit_arms)
	for arm in bandit_arms:
		action = {}
		for i in range(len(experiment_vars)):
			if arm[i] == 1:
				action[experiment_vars[i]] = 1
		if not models.is_valid_action(action):
			raise AssertionError('The arm is not valid' + str(arm))
	hypo_params_independent = models.read_independent_model(experiment_vars)

	user_count = input_dict['user_count']
	batch_size = input_dict['batch_size'] # 10
	simulation_count = input_dict['simulation_count']  # 2500
	extensive = input_dict['extensive']
	rand_sampling_applied = input_dict['rand_sampling_applied']
	independent_applied = input_dict['independent_applied']
	no_interaction_applied = input_dict['no_interaction_applied']
	show_fig = input_dict['show_fig']
	regret_top = input_dict['regret_top']

	regrets_rand = np.zeros(user_count)
	optimal_action_ratio_rand = np.zeros(user_count)

	policies = []
	policies.append(['Thompson with Interaction'])

	save_output_folder = 'saved_output/raw_data/'+str(TODAY)+'_'+str(NOW.hour)+str(NOW.minute)+str(NOW.second)+"/"
	if not os.path.exists(save_output_folder):
		os.makedirs(save_output_folder)

	thompson_interaction_results = results(save_output_folder, "thompson_with_interaction", user_count, true_coeff, hypo_params)
	thompson_no_interaction_results = results(
		save_output_folder, "thompson_without_interaction", user_count, true_coeff, hypo_params_no_interaction)
	independent_results = results(save_output_folder, "independent_bandit", user_count, true_coeff, hypo_params_independent)
	random_results = results(save_output_folder, "random", user_count, true_coeff, hypo_params)

	for sim in range(0, simulation_count):
		print("sim: ",sim)
		a_pre = input_dict['NIG_priors']['a']
		b_pre = input_dict['NIG_priors']['b']
		#Hammad: Bias Correction
		mean_pre = np.zeros(sum(len(x) for x in hypo_params))
		cov_pre = np.identity(sum(len(x) for x in hypo_params))

		#Step 3: Calls the right policy
		'''
		Calls thompson sampling, random sampling, or any others which is
		specified
		by the user in command line (default: Calls thompson sampling)
			default priors: 0 unless it is specified by the user
		'''
		users_context = models.generate_true_dataset(context_vars, user_count, input_dict['dist_of_context'])

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
		thompson_interaction_results.add_regret(thompson_output[2])

		thompson_interaction_results.add_optimal_action_ratio(thompson_output)

		thompson_interaction_results.add_beta_thompson_coeff(thompson_output[5])

		thompson_interaction_results.add_mse(thompson_output[5], sim)

		thompson_interaction_results.add_coeff_sign_err(thompson_output[5], sim)

		thompson_interaction_results.add_bias_in_coeff(thompson_output, sim, experiment_vars)

		if independent_applied:
			a_pre = input_dict['NIG_priors']['a']
			b_pre = input_dict['NIG_priors']['b']
			#Hammad: Bias Correction
			mean_pre = np.zeros(sum(len(x) for x in hypo_params_independent))
			cov_pre = np.identity(sum(len(x) for x in hypo_params_independent))

			independent_outputs = thompson.apply_thompson_sampling(users_context,
													experiment_vars,
													bandit_arms,
													hypo_params_independent,
													true_coeff,
													batch_size,
													extensive,
													mean_pre,
													cov_pre,
													a_pre,
													b_pre,
													noise_stats)

			independent_results.add_regret(independent_outputs[2])

			independent_results.add_optimal_action_ratio(independent_outputs)

			independent_results.add_beta_thompson_coeff(independent_outputs[5])

			independent_results.add_mse(independent_outputs[5], sim)

			independent_results.add_coeff_sign_err(independent_outputs[5], sim)

			independent_results.add_bias_in_coeff(independent_outputs, sim, experiment_vars)

		if no_interaction_applied:
			a_pre = input_dict['NIG_priors']['a']
			b_pre = input_dict['NIG_priors']['b']
			#Hammad: Bias Correction
			mean_pre = np.zeros(sum(len(x) for x in hypo_params_no_interaction))
			cov_pre = np.identity(sum(len(x) for x in hypo_params_no_interaction))

			no_interaction_outputs = thompson.apply_thompson_sampling(users_context,
													experiment_vars,
													bandit_arms,
													hypo_params_no_interaction,
													true_coeff,
													batch_size,
													extensive,
													mean_pre,
													cov_pre,
													a_pre,
													b_pre,
													noise_stats)

			thompson_no_interaction_results.add_regret(no_interaction_outputs[2])

			thompson_no_interaction_results.add_optimal_action_ratio(no_interaction_outputs)

			thompson_no_interaction_results.add_beta_thompson_coeff(no_interaction_outputs[5])

			thompson_no_interaction_results.add_mse(no_interaction_outputs[5], sim)

			thompson_no_interaction_results.add_coeff_sign_err(no_interaction_outputs[5], sim)

			thompson_no_interaction_results.add_bias_in_coeff(no_interaction_outputs, sim, experiment_vars)

		if(rand_sampling_applied):
			rand_outputs= random.apply_random_sampling(users_context,
														experiment_vars,
														bandit_arms,
														hypo_params,
														true_coeff,
														batch_size,
														extensive,
														noise_stats)
			random_results.add_regret(rand_outputs[2])

			random_results.add_optimal_action_ratio(rand_outputs)

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

		#Only saving the output of OLS Random Policy, and not plotting it

		if(rand_sampling_applied):
			d1_r = [rand_outputs[1][i][0] for i in range(0,
						len(rand_outputs[1]))]
			d1_x1_r = [a*b for a,b in zip(d1_r,x1)]
			df_r = pd.DataFrame({'d1':d1_r, 'd1x1':d1_x1_r,
									'y':rand_outputs[0]})
			for iteration in range(1, user_count+1):
				regression_r = sm.ols(formula="y ~ d1 + d1x1",
									data=df_r.iloc[:iteration]).fit()
				regression_intercept_random.append(
											regression_r.params['Intercept'])
				regression_d1_random.append(regression_r.params['d1'])
				regression_d1x1_random.append(regression_r.params['d1x1'])

			regression_intercept_all_sim_random.append(
												regression_intercept_random)
			regression_d1_all_sim_random.append(regression_d1_random)
			regression_d1x1_all_sim_random.append(regression_d1x1_random)

	regression_intercept_all_sim_df=pd.DataFrame(regression_intercept_all_sim)
	regression_d1_all_sim_df=pd.DataFrame(regression_d1_all_sim)
	regression_d1x1_all_sim_df=pd.DataFrame(regression_d1x1_all_sim)

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
									save_output_folder),
									index_label='iteration')

	regression_intercept_all_sim_mean = np.mean(
			regression_intercept_all_sim_df, axis=0)

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
				simulation_count, batch_size, save_fig=True)'''
	################# OLS REGRESSION ENDS ########################

	thompson_interaction_results.save_to_csv()

	thompson_interaction_results.average_per_sim(simulation_count)

	if no_interaction_applied:
		policies.append(['Thompson without Interaction'])
		thompson_no_interaction_results.save_to_csv()
		thompson_no_interaction_results.average_per_sim(simulation_count)

	if independent_applied:
		policies.append(['Independent Bandits'])
		independent_results.save_to_csv()
		independent_results.average_per_sim(simulation_count)

	if(rand_sampling_applied):
		policies.append(['Random Sampling'])
		random_results.save_to_csv()
		random_results.average_per_sim(simulation_count)

	#Step 4: Plots
	'''
	Plots some basic figures. In "Extensive Mode", details will be saved so
	user can plots more figures if desired.
	'''

	regrets_all_policies = np.array([thompson_interaction_results.regrets])
	optimal_action_ratio_all_policies = np.array([thompson_interaction_results.optimal_action_ratio])
	if no_interaction_applied:
		regrets_all_policies = np.append(regrets_all_policies, [thompson_no_interaction_results.regrets], axis=0)
		optimal_action_ratio_all_policies = np.append(optimal_action_ratio_all_policies, 
			[thompson_no_interaction_results.optimal_action_ratio], axis=0)
	if independent_applied:
		regrets_all_policies = np.append(regrets_all_policies, [independent_results.regrets], axis=0)
		optimal_action_ratio_all_policies = np.append(optimal_action_ratio_all_policies, 
			[independent_results.optimal_action_ratio], axis=0)
	if rand_sampling_applied:
		regrets_all_policies = np.append(regrets_all_policies, [random_results.regrets], axis=0)
		optimal_action_ratio_all_policies = np.append(optimal_action_ratio_all_policies, 
			[random_results.optimal_action_ratio], axis=0)
	
	bplots.plot_regret(user_count, policies, regrets_all_policies,
						simulation_count, batch_size, top=regret_top)
	
	bplots.plot_optimal_action_ratio(user_count, policies,
			optimal_action_ratio_all_policies, simulation_count, batch_size,
			mode='per_user')
	
	bplots.plot_coeff_ranking(user_count, 'Thompson with Interaction',
				thompson_interaction_results.beta_thompson_coeffs, hypo_params[0], simulation_count,
				batch_size, save_fig=True)

	bplots.plot_coeff_sign_error(user_count, 'Thompson with Interaction', hypo_params[0],
				thompson_interaction_results.coeff_sign_error, simulation_count, batch_size, save_fig=True)

	
	bplots.plot_bias_in_coeff(user_count, 'Thompson with Interaction', hypo_params[0],
				thompson_interaction_results.bias_in_coeff, simulation_count, batch_size, save_fig=True)

	# TODO create methods
	if no_interaction_applied:
		bplots.plot_coeff_ranking(user_count, 'Thompson without Interaction',
				thompson_no_interaction_results.beta_thompson_coeffs, hypo_params_no_interaction[0], simulation_count,
				batch_size, save_fig=True)

		bplots.plot_coeff_sign_error(user_count, 'Thompson without Interaction', hypo_params_no_interaction[0],
				thompson_no_interaction_results.coeff_sign_error, simulation_count, batch_size, save_fig=True)
	
		bplots.plot_bias_in_coeff(user_count, 'Thompson without Interaction', hypo_params_no_interaction[0],
				thompson_no_interaction_results.bias_in_coeff, simulation_count, batch_size, save_fig=True)

	if independent_applied:
		bplots.plot_coeff_ranking(user_count, 'Independent bandits',
				independent_results.beta_thompson_coeffs, independent_results.flat_hypo_params, simulation_count,
				batch_size, save_fig=True)

		bplots.plot_coeff_sign_error(user_count, 'Independent bandits', independent_results.flat_hypo_params,
				independent_results.coeff_sign_error, simulation_count, batch_size, save_fig=True)
	
		bplots.plot_bias_in_coeff(user_count, 'Independent bandits', independent_results.flat_hypo_params,
				independent_results.bias_in_coeff, simulation_count, batch_size, save_fig=True)
	
	if(show_fig):
		#plt.show(block=False)
		plt.show()

	return

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
