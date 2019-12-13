import numpy as np
import true_hypo_models as models
import making_decision
from scipy.stats import invgamma
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

def calculate_posteriors_nig(dependant_var, regressors, mean_pre, cov_pre,
                                a_pre,b_pre):
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

    resid = np.subtract(dependant_var, np.dot(regressors,mean_pre))
    resid_trans = np.matrix.transpose(resid)

    # N x N middle term for gamma update: (I + XVX')^{-1}
    mid_term = np.linalg.inv(np.add(np.identity(data_size), np.dot(np.dot(regressors, cov_pre),regressors_trans)))

    ## Update coeffecients priors

    # Update mean vector: [(V^{-1} + X'X)^{-1}][V^{-1}mu + X'y]
    mean_post = np.dot(np.linalg.inv(np.add(np.linalg.inv(cov_pre), np.dot(
        regressors_trans,regressors))), np.add(np.dot(np.linalg.inv(cov_pre),
        mean_pre), np.dot(regressors_trans,dependant_var)))
    #logging.info("mean_post: \n{}".format(mean_post))
    # Update covariance matrix: (V^{-1} + X'X)^{-1}
    cov_post = np.linalg.inv(np.add(np.linalg.inv(cov_pre), np.dot(
            regressors_trans,regressors)))
    #logging.info("cov_post: \n{}".format(cov_post))

    ## Update precesion prior

    # Update gamma parameters: a + n/2 (shape parameter)
    a_post = a_pre + data_size/2

    # b + (1/2)(y - Xmu)'(I + XVX')^{-1}(y - Xmu) (scale parameter)
    b_post = b_pre + (np.dot(np.dot(resid_trans, mid_term), resid))/2


    return [a_post, b_post, cov_post, mean_post]


def draw_posterior_sample(hypo_model_params, mean, cov, a, b):
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


def apply_thompson_sampling(user_context,
                            experiment_vars,
                            bandit_arms,
                            hypo_model_params,
                            true_coeff,
                            batch_size,
                            mean_pre,
                            cov_pre,
                            a_pre,
                            b_pre,
                            noise_stats):
    """
    Apply thompson sampling on the input dataset at each batch and calculate
    regret/reward of the selected option.

    Args:
        user_context (dataframe): containing user contextual values
        experiment_vars(ndarray): 1D array containing the name of variables
        bandit_arms (list): list containing values of arms
        hypo_model_params(list): containing all parameters of hypo model
        true_coeff(dict): containing all parameters and values of true model
        batch_size (int): size of each batch for thompson sampling iteration
        mean_pre (ndarray): 1D array containing mean of NIG priors.
        cov_pre (ndarray): 2D array containing covariance of NIG priors.
        a_pre (int): shape of inverse-gamma distribution
        b_pre (int): scale of inverse-gamma distribution
        noise_stats(dict): in the format of {'noise_mean': 0, 'noise_std': 1}

    Returns:
        list: calculated regret for each user
    """
    user_count = user_context.shape[0]
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
                                                    user_context.iloc[user],
                                                    experiment_vars,
                                                    bandit_arms)
            received_reward = models.true_model_output(true_coeff,
                                                    experiment_vars,
                                                    user_context.iloc[user],
                                                    hypo_optimal_action[0],
                                                    noise_stats)
            received_reward_no_noise = models.true_model_output(true_coeff,
                                                    experiment_vars,
                                                    user_context.iloc[user],
                                                    hypo_optimal_action[0],
                                                    {"noise_mean": 0,
                                                    "noise_std": 0.0})
            true_optimal_action = making_decision.pick_true_optimal_arm(
                                                    true_coeff,
                                                    user_context.iloc[user],
                                                    experiment_vars,
                                                    bandit_arms)
            regret = making_decision.calculate_regret(true_optimal_action[1], 
                                                    received_reward_no_noise)
            X = models.calculate_hypo_regressors(hypo_model_params,
                                                experiment_vars,
                                                user_context.iloc[user],
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



