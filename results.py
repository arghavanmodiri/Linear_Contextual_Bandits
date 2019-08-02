import pandas as pd
import numpy as np

class results:
    def __init__(self, save_output_folder, algo, user_count: int, true_coeff: dict, hypo_params: list):
        self.save_output_folder = save_output_folder
        self.algo = algo
        self.user_count = user_count
        self.hypo_params = hypo_params
        self.regrets = np.zeros(user_count)
        self.optimal_action_ratio = np.zeros(user_count)
        self.mse = np.zeros((user_count, len(hypo_params)))
        self.beta_thompson_coeffs = np.zeros((user_count, len(hypo_params)))
        self.coeff_sign_error = np.zeros((user_count, len(hypo_params)))
        self.bias_in_coeff = np.zeros((user_count, len(hypo_params)))
        self.optimal_action_ratio_df = pd.DataFrame()
        self.mse_df = pd.DataFrame()
        self.coeff_sign_err_df = pd.DataFrame()
        self.bias_in_coeff_df = pd.DataFrame()
        self.regret_df = pd.DataFrame()
        self.suboptimal_action_ratio_group_df = pd.DataFrame()
        self.context_action_df = pd.DataFrame()
        self.true_coeff_list = list(true_coeff.values())

        self.true_params_in_hypo = []
        for  idx, hypo_param_name in enumerate(hypo_params):
            if(hypo_param_name in true_coeff):
                self.true_params_in_hypo.append(true_coeff[hypo_param_name])

    def add_regret(self, regret):
        self.regrets += regret
        self.regret_df = pd.concat([self.regret_df, pd.DataFrame(regret)], ignore_index=True, axis=1)

    def add_optimal_action_ratio(self, outputs):
        optimal_action_ratio_per_sim = np.array(list((outputs[1][i] in outputs[0][i]) for i in range(0, self.user_count))).astype(int)
        self.optimal_action_ratio += optimal_action_ratio_per_sim
        self.optimal_action_ratio_df = pd.concat([self.optimal_action_ratio_df, 
            pd.DataFrame(optimal_action_ratio_per_sim)], ignore_index=True, axis=1)

    def add_beta_thompson_coeff(self, beta_thompson_coeff):
        self.beta_thompson_coeffs += beta_thompson_coeff

    def add_mse(self, beta_thompson_coeff, sim):
        mse_per_sim = np.power(np.array(np.array(beta_thompson_coeff) -
                                np.array(self.true_params_in_hypo)),2)
        self.mse += mse_per_sim
        mse_per_sim_df = pd.DataFrame(mse_per_sim)
        mse_per_sim_df.columns = pd.MultiIndex.from_product([[sim], self.hypo_params])
        self.mse_df = pd.concat([self.mse_df, mse_per_sim_df], axis=1)

    def add_coeff_sign_err(self, beta_thompson_coeff, sim):
        coeff_sign_error_per_sim = np.sign(
            np.array(self.true_params_in_hypo)) - np.sign(np.array(beta_thompson_coeff)) == np.zeros(len(self.hypo_params))
        self.coeff_sign_error +=coeff_sign_error_per_sim
        coeff_sign_error_per_sim_df = pd.DataFrame(coeff_sign_error_per_sim.astype(int))
        coeff_sign_error_per_sim_df.columns = pd.MultiIndex.from_product([[sim], self.hypo_params])
        self.coeff_sign_err_df = pd.concat([self.coeff_sign_err_df, 
            coeff_sign_error_per_sim_df], axis=1)

    def add_bias_in_coeff(self, thompson_output, sim, experiment_vars):
        if len(self.true_params_in_hypo) == len(self.true_coeff_list):
            bias_in_coeff_per_sim = np.array(np.array(thompson_output[5]) - np.array(self.true_params_in_hypo))

        # Under specified model bias (Y = A0 + A1D)
        else:
            bias_in_coeff_per_sim = np.array(thompson_output[5])[:, 0] - self.true_coeff_list[0]
            for act in range(len(experiment_vars)):
                act_taken = np.array(thompson_output[1])[:, act]
                true_reward_act = np.array(thompson_output[3])[act_taken == 1]
                true_reward_other = np.array(thompson_output[3])[act_taken == 0]
                bias_in_coeff_per_sim = np.column_stack((
                    bias_in_coeff_per_sim, np.array(
                        thompson_output[5])[:, act+1] - (np.mean(true_reward_act) - np.mean(true_reward_other))))

        self.bias_in_coeff += bias_in_coeff_per_sim
        bias_in_coeff_per_sim_df = pd.DataFrame(bias_in_coeff_per_sim)
        bias_in_coeff_per_sim_df.columns = pd.MultiIndex.from_product([[sim], self.hypo_params])
        self.bias_in_coeff_df = pd.concat([self.bias_in_coeff_df, 
            bias_in_coeff_per_sim_df], axis=1)

    def save_to_csv(self):
        self.regret_df.to_csv('{}{}_regrets.csv'.format(
                                self.save_output_folder, self.algo), index_label='iteration')
        self.optimal_action_ratio_df.to_csv(
                                '{}{}_optimal_action_ratio.csv'.format(
                                self.save_output_folder, self.algo), index_label='iteration')
        self.mse_df.to_csv('{}{}_mse.csv'.format(
                                self.save_output_folder, self.algo), index_label='iteration')
        self.bias_in_coeff_df.to_csv(
                                '{}{}_bias_in_coeff.csv'.format(
                                self.save_output_folder, self.algo))
        self.coeff_sign_err_df.to_csv(
                                '{}{}_coeff_sign_err.csv'.format(
                                self.save_output_folder, self.algo))
        self.context_action_df.to_csv(
                                '{}{}_context_action.csv'.format(
                                self.save_output_folder, self.algo))

    def average_per_sim(self, simulation_count):
        self.regrets = self.regrets / simulation_count
        self.optimal_action_ratio = self.optimal_action_ratio /simulation_count
        self.mse = self.mse / simulation_count
        self.beta_thompson_coeffs = self.beta_thompson_coeffs / simulation_count
        self.bias_in_coeff = self.bias_in_coeff / simulation_count
        self.coeff_sign_error = self.coeff_sign_error / simulation_count