import numpy as np
import pandas as pd
import policies.thompson_sampling_nig as thompson

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


    def apply_policy(self, users_context, mean_pre, cov_pre, a_pre, b_pre, noise_stats):
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
        self.save_beta_thompson_coeffs()

    def regret_cumulative():
        return True

    def regret_average(self):
        return (self.save_regret_thompson_df.sum(axis=1)/self.user_count)

    def optimal_action_ratio_average():
        return (self.save_optimal_action_ratio_thompson_df.sum(axis=1)/self.user_count)

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


    def save_beta_thompson_coeffs(self):
        return True

    def save_coeff_sign_err_thompson_df(self):
        return True

    def save_mse_thompson_df(self):
        return True

    def save_bias_in_coeff_thompson_df(self):
        return True