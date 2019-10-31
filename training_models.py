import numpy as np
import pandas as pd
import policies.thompson_sampling_nig as thompson
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

class training_bandit_model(object):
    """docstring for ClassName"""

    def __init__(self, hypo_params, user_count, batch_size,
                experiment_vars, bandit_arms, true_coeff, extensive):
        super(training_bandit_model, self).__init__()

        self.hypo_params = hypo_params
        self.user_count = user_count
        self.batch_size = batch_size
        #self.users_context = users_context
        self.experiment_vars = experiment_vars
        self.bandit_arms = bandit_arms
        self.true_coeff = true_coeff
        self.extensive = extensive
        self.cumulative_regret = np.zeros(user_count)
        self.thompson_output = np.zeros(6)

        self.save_regret_thompson_df = pd.DataFrame()
        self.save_optimal_action_ratio_thompson_df = pd.DataFrame()
        self.beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
        self.beta_thompson_coeffs_df = pd.DataFrame()


    def apply_thompson(self, users_context, a_pre, b_pre, noise_stats):
        mean_pre = np.zeros(len(self.hypo_params))
        cov_pre = np.identity(len(self.hypo_params))
        thompson_output = thompson.apply_thompson_sampling(
                                                    users_context,
                                                    self.experiment_vars,
                                                    self.bandit_arms,
                                                    self.hypo_params,
                                                    self.true_coeff,
                                                    self.batch_size,
                                                    self.extensive,
                                                    mean_pre,
                                                    cov_pre,
                                                    a_pre,
                                                    b_pre,
                                                    noise_stats)

        self.save_regret(thompson_output[2])
        self.save_optimal_action_ratio(thompson_output[1], thompson_output[0])
        self.save_beta_thompson_coeffs_sum(thompson_output[5])

    def regret_cumulative():
        return True

    def get_regret_average(self):
        simulation_count = self.save_regret_thompson_df.shape[1]
        return (self.save_regret_thompson_df.sum(axis=1)/simulation_count)

    def get_optimal_action_ratio_average(self):
        simulation_count = self.save_optimal_action_ratio_thompson_df.shape[1]
        return (self.save_optimal_action_ratio_thompson_df.sum(axis=1)/simulation_count)

    def save_regret(self, new_regret):
        self.save_regret_thompson_df = pd.concat([self.save_regret_thompson_df,
                                            pd.DataFrame(new_regret)],
                                            ignore_index=True, axis=1)


    def save_optimal_action_ratio(self, hypo_optimal_action, true_optimal_action):
        optimal_action_ratio_per_sim = np.array(list((hypo_optimal_action[i] in
            true_optimal_action[i]) for i in range(0,self.user_count))).astype(int)
        self.save_optimal_action_ratio_thompson_df = pd.concat([
                                self.save_optimal_action_ratio_thompson_df,
                                pd.DataFrame(optimal_action_ratio_per_sim)],
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
        return self.save_regret_thompson_df

    def get_optimal_action_ratio(self):
        return self.save_optimal_action_ratio_thompson_df

    def get_beta_thompson_coeffs_average(self):
        return self.beta_thompson_coeffs_df.groupby(level=1, axis=1).mean()

    def save_coeff_sign_err_thompson_df(self):
        return True

    def save_mse_thompson_df(self):
        return True

    def save_bias_in_coeff_thompson_df(self):
        return True