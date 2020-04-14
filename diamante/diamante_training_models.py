import numpy as np
import pandas as pd
import diamante.diamante_thompson_sampling_nig as dthompson
import policies.random_sampling as random
import making_decision
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

class training_bandit_model(object):
    """docstring for ClassName"""

    def __init__(self, user_count, ts_update_day_count, day_count,
                experiment_vars, bandit_arms, true_coeff, extensive, hypo_params = None):
        super(training_bandit_model, self).__init__()

        
        self.user_count = user_count
        self.batch_day = ts_update_day_count
        self.day_count = day_count
        self.experiment_vars = experiment_vars
        self.true_coeff = true_coeff
        self.extensive = extensive
        self.bandit_arms = bandit_arms
        self.hypo_params = hypo_params
        self.cumulative_regret = np.zeros(user_count)
        self.save_regret_df = pd.DataFrame()
        self.save_context_selected_action_df = pd.DataFrame()
        self.save_optimal_action_ratio_df = pd.DataFrame()
        
        if hypo_params != None:
            self.thompson_output = np.zeros(6)
            self.beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
            self.beta_thompson_coeffs_df = pd.DataFrame()


    def apply_thompson(self, users_context, a_pre, b_pre, mean_pre, cov_pre,
                        noise_stats):
        #mean_pre = np.zeros(len(self.hypo_params))
        #cov_pre = np.identity(len(self.hypo_params))
        thompson_output = dthompson.apply_thompson_sampling(
                                                    users_context,
                                                    self.experiment_vars,
                                                    self.bandit_arms,
                                                    self.hypo_params,
                                                    self.true_coeff,
                                                    self.batch_day,
                                                    self.day_count,
                                                    mean_pre,
                                                    cov_pre,
                                                    a_pre,
                                                    b_pre,
                                                    noise_stats)

        #Note: thompson_output[1] is the selected action by TS from possible_actions list
        self.save_regret(thompson_output[2], thompson_output[6])
        self.save_context_selected_action(thompson_output[6], thompson_output[1], thompson_output[3]) # thompson_output[6] is users_context_all
        self.save_optimal_action_ratio(thompson_output[1], thompson_output[0], thompson_output[6])
        self.save_beta_thompson_coeffs_sum(thompson_output[5])

    def apply_random(self, users_context, noise_stats):

        rand_output = random.apply_random_sampling(users_context,
                                            self.experiment_vars,
                                            self.bandit_arms,
                                            self.hypo_params,
                                            self.true_coeff,
                                            self.batch_size,
                                            self.extensive,
                                            noise_stats)

        self.save_regret(rand_output[2])
        true_optimal_action_all = []
        for user in range(self.user_count):
            true_optimal_action = making_decision.pick_true_optimal_arm(
                                                    self.true_coeff,
                                                    users_context.iloc[user],
                                                    self.experiment_vars,
                                                    self.bandit_arms)
            true_optimal_action_all.append(true_optimal_action[0])
        self.save_optimal_action_ratio(rand_output[1], true_optimal_action_all)

    def regret_cumulative():
        return True

    def get_regret_average(self):
        #simulation_count = self.save_regret_df.shape[1]
        return (self.save_regret_df.mean(axis=1))#/simulation_count)

    def get_regret_std(self):
        return (self.save_regret_df.std(axis=1))

    def get_optimal_action_ratio_average(self):
        #simulation_count = self.save_optimal_action_ratio_df.shape[1]
        return (self.save_optimal_action_ratio_df.mean(axis=1))#/simulation_count)

    def get_optimal_action_ratio_std(self):
        return (self.save_optimal_action_ratio_df.std(axis=1))

    def save_regret(self, new_regret, users_context):
        new_regret_df = pd.DataFrame(new_regret)
        new_regret_df.index = users_context.index
        self.save_regret_df = pd.concat([self.save_regret_df,
                                            new_regret_df],
                                            ignore_index=True, axis=1)

    def save_context_selected_action(self, user_context, selected_action_per_sim, reward):
        simulation_count = self.save_regret_df.shape[1]
        if type(user_context) == pd.core.frame.DataFrame:
            users_context_df = user_context
        else:
            users_context_df = pd.DataFrame(list(user_context))
        users_context_df.insert(0, 'simulation_number', simulation_count)
        selected_action_per_sim_df = pd.DataFrame(selected_action_per_sim)
        selected_action_per_sim_df.index = users_context_df.index
        selected_action_per_sim_df.columns = self.experiment_vars
        reward_ser = pd.Series(reward, name='Reward')
        reward_ser.index = users_context_df.index
        temp_df = pd.concat([users_context_df, selected_action_per_sim_df, reward_ser], axis=1)
        self.save_context_selected_action_df = pd.concat([self.save_context_selected_action_df,
                                            temp_df])


    def save_optimal_action_ratio(self, hypo_optimal_action, true_optimal_action, users_context):
        optimal_action_ratio_per_sim = np.array(list((hypo_optimal_action[i] in
            true_optimal_action[i]) for i in range(0,len(users_context)))).astype(int)
        optimal_action_ratio_per_sim_df = pd.DataFrame(optimal_action_ratio_per_sim)
        optimal_action_ratio_per_sim_df.index = users_context.index
        self.save_optimal_action_ratio_df = pd.concat([
                                self.save_optimal_action_ratio_df,
                                optimal_action_ratio_per_sim_df],
                                ignore_index=True, axis=1)

    def save_beta_thompson_coeffs_sum(self, new_beta_coeff):
        self.beta_thompson_coeffs += 0#np.array(new_beta_coeff)
        sim_id = int(self.beta_thompson_coeffs_df.shape[1] / len(new_beta_coeff[0].keys()))
        new_beta_coeff_df = pd.DataFrame(new_beta_coeff)
        new_beta_coeff_df.columns = pd.MultiIndex.from_product([
                [sim_id], new_beta_coeff_df.columns])
        self.beta_thompson_coeffs_df = pd.concat([
                                self.beta_thompson_coeffs_df,
                                new_beta_coeff_df], axis=1)

    def calculate_mse(self):
        true_params_in_hypo = []
        for  idx, hypo_param_name in enumerate(self.hypo_params):
            if(hypo_param_name in self.true_coeff):
                true_params_in_hypo.append(true_coeff[hypo_param_name])

        mse = np.power(
                    self.beta_thompson_coeffs - np.array(true_params_in_hypo),
                    2)

        mse_per_sim_df = pd.DataFrame(mse_per_sim)
        mse_per_sim_df.columns = pd.MultiIndex.from_product([
                [sim], hypo_params])
        save_mse_thompson_df = pd.concat([save_mse_thompson_df,
                                mse_per_sim_df], axis=1)


    def get_regret(self):
        return self.save_regret_df

    def get_selected_action(self):
        return self.save_context_selected_action_df

    def get_optimal_action_ratio(self):
        return self.save_optimal_action_ratio_df

    def get_beta_thompson_coeffs_average(self):
        return self.beta_thompson_coeffs_df.groupby(level=1, axis=1).mean()

    def save_coeff_sign_err_thompson_df(self):
        return True

    def save_mse_thompson_df(self):
        return True

    def save_bias_in_coeff_thompson_df(self):
        return True