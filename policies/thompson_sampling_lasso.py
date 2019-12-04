import numpy as np
import true_hypo_models as models
import making_decision
from scipy.stats import invgamma
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

def calculate_posteriors_lasso(dependant_var, regressors, mean_pre, cov_pre,
                                a_pre,b_pre, hypo_params):
    """
    Calculate the posterior probablibity after observing the true donation by
    appliying applied_arm to user with user_context value and assuming prior
    distribution is Normal-Inverse-Gamma(NIG).

    Args:
        dependant_var (ndarray): 1D array containing the true value of variable
        user_context (ndarray): 2D array containing users' contextual values
        applied_arms (ndarray): 2D array containing the arm that took effect
        mean_pre (ndarray): 1D array containing mean of NIG priors.
        cov_pre (ndarray): 2D array containing covariance of NIG priors.
        a_pre (ndarray): shape of inverse-gamma distribution
        b_pre (ndarray): scale of inverse-gamma distribution

    Returns:
        list: containing [mean,cov,a,b] of posteriors distribution
    """

    data_size = len(dependant_var)

    # X transpose

    regressors_trans = np.matrix.transpose(regressors)

    #resid = np.subtract(dependant_var, np.dot(regressors,mean_pre))
    #resid_trans = np.matrix.transpose(resid)
    
    beta_pre = list(draw_posterior_sample_lasso(hypo_params, mean_pre, cov_pre, a_pre, b_pre))
    beta_pre_trans = np.matrix.transpose(beta_pre)

    a = (len(regressors) + len(regressors[0]) - 1)/2

    b = np.dot(np.dot(beta_pre_trans, cov), beta_pre)/2
    resid = np.subtract(dependant_var, np.dot(regressors,beta_pre))
    resid_trans = np.matrix.transpose(resid)
    b = b + np.dot(resid_trans, resid)/2

    A = np.dot(regressors_trans, regressors) + np.linalg.inv(cov_pre)
    B = np.dot(regressors_trans, dependant_var) + np.linalg.inv(cov_pre)*mean_pre
    cov_post = np.linalg.inv(A)
    mean_post = np.dot(cov_post, B)

    # b + (1/2)(y - Xmu)'(I + XVX')^{-1}(y - Xmu) (scale parameter)
    b_post = b_pre + (np.dot(np.dot(resid_trans, mid_term), resid))/2



    return [a_post, b_post, cov_post, mean_post]


def draw_posterior_sample_lasso(hypo_model_params, mean, cov, a, b):
    """
    Draw samples from Normal-Inverse-Gamma(NIG) distribution.

    Args:
        mean (ndarray): 1D array containing mean of the distribution.
        cov (ndarray): 2D array containing covariance of the distribution.
        a (ndarray): shape of inverse-gamma distribution
        b (ndarray): scale of inverse-gamma distribution

    Returns:
        list: containing samples from multivariant NIG distribution
    """
    #sample from inverse gamma (shape, loc, scale, draws)
    #np.random.seed(123)
    var_draw = invgamma.rvs(a, 0, b, size = 1)

    # Coeffecients from multivariate normal 
    cholesky_decomposition = np.linalg.cholesky(var_draw*cov)
    standard_rand = np.random.standard_normal(len(cov))
    beta_draw = mean + np.dot(cholesky_decomposition, standard_rand)
    #beta_draw = np.random.multivariate_normal(mean, var_draw*cov)
    beta_draw = {hypo_model_params[i]:beta_draw[i] for i in range(0,len(hypo_model_params))}

    return beta_draw


def apply_thompson_sampling_lasso(user_context, experiment_vars, bandit_arms, hypo_model_params, true_coeff,
                            batch_size, extensive, mean_pre, cov_pre,
                            a_pre,b_pre, noise_stats):
    """
    Apply thompson sampling on the input dataset at each batch and calculate
    regret/reward of the selected option.

    Args:
        donations (ndarray): 1D array containing the true donations
        user_context (ndarray): 2D array containing user contextual values
        bandit_arms (ndarray): 2D array containing all arms
        batch_size (int): size of each batch for thompson sampling iteration
        extensive (bool): if True, all the details will be saved in a file
        mean_pre (ndarray): 1D array containing mean of NIG priors.
        cov_pre (ndarray): 2D array containing covariance of NIG priors.
        a_pre (ndarray): shape of inverse-gamma distribution
        b_pre (ndarray): scale of inverse-gamma distribution

    Returns:
        list: calculated regret for each user
    """
    user_count = user_context.shape[0]
    #print("in sampling,user_count: ", user_count)
    batch_count = int(user_count/batch_size)
    start_batch = 0
    end_batch = batch_size
    dependant_var = np.zeros(batch_size)
    X_pre = np.zeros([batch_size,len(hypo_model_params)])

    regret_all = []
    true_optimal_action_all = []
    hypo_optimal_action_all = []
    true_reward_all = []
    hypo_reward_all = []
    beta_thompson_all = []

    for batch in range(0, batch_count):
        X_batch = []
        y_batch = []

        thompson_dist = calculate_posteriors_nig(dependant_var,
                            X_pre, mean_pre, cov_pre, a_pre,b_pre)

        mean = thompson_dist[3]
        cov = thompson_dist[2]
        a = thompson_dist[0]
        b = thompson_dist[1]

        for user in range(start_batch, end_batch):
            beta_thompson = draw_posterior_sample(hypo_model_params,mean, cov, a, b)
            #beta_thompson_all.append([beta_thompson[i] for i in beta_thompson.keys()])
            beta_thompson_all.append(beta_thompson)

            hypo_optimal_action = making_decision.pick_hypo_optimal_arm(
                                                        beta_thompson,
                                                        user_context[user],
                                                        experiment_vars,
                                                        bandit_arms)
            received_reward = models.true_model_output(true_coeff,
                                                        experiment_vars,
                                                        user_context[user],
                                                        hypo_optimal_action[0],
                                                        noise_stats)
            received_reward_no_noise = models.true_model_output(true_coeff,
                                                        experiment_vars,
                                                        user_context[user],
                                                        hypo_optimal_action[0],
                                                        {"noise_mean": 0,
                                                        "noise_std": 0.0})
            true_optimal_action = making_decision.pick_true_optimal_arm(
                                                        true_coeff,
                                                        user_context[user],
                                                        experiment_vars,
                                                        bandit_arms)
            regret = making_decision.calculate_regret(true_optimal_action[1], 
                                                    received_reward_no_noise)
            X = models.calculate_hypo_regressors(hypo_model_params,
                                                experiment_vars,
                                                user_context[user],
                                                hypo_optimal_action[0])

            X_batch.append(X)
            y_batch.append(received_reward)
            regret_all.append(regret)
            true_optimal_action_all.append(true_optimal_action[0])
            hypo_optimal_action_all.append(hypo_optimal_action[0])
            true_reward_all.append(received_reward)
            hypo_reward_all.append(hypo_optimal_action[1])

        X_pre = np.array(X_batch)
        dependant_var = np.array(y_batch)
        mean_pre = mean
        cov_pre = cov
        a_pre = a
        b_pre = b
        start_batch = end_batch
        end_batch = end_batch+batch_size

    return [true_optimal_action_all, hypo_optimal_action_all, regret_all, true_reward_all, hypo_reward_all, beta_thompson_all]



