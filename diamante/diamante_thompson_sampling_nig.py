import numpy as np
import pandas as pd
import diamante.diamante_true_hypo_models as dmodels
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
    dependant_var = np.array(dependant_var, dtype='float')
    regressors = np.array(regressors, dtype='float')
    regressors_trans = regressors.T
    resid = (dependant_var - regressors.dot(mean_pre))
    resid_trans = resid.T

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
                            batch_day,
                            day_count,
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

    #user_count = user_context.shape[0]
    #batch_count = int(user_count/batch_size)
    #start_batch = 0
    #end_batch = batch_size
    #dependant_var = np.zeros(batch_size)
    #X_pre = np.zeros([batch_size,len(hypo_model_params)])

    # For saving all information of 1 round of simulations
    regret_all = []
    true_optimal_action_all = []
    hypo_optimal_action_all = []
    true_reward_all = []
    hypo_reward_all = []
    beta_thompson_all = []
    contexts_all = []
    user_context_all = pd.DataFrame()



    # For saving all information for all users in 1 batch (day_count multiplied by number of users)
    y_batch = []
    X_batch = pd.DataFrame()

    for day in range(day_count):
        # For saving all information for all users in 1 day
        selected_arms_per_day = pd.DataFrame(columns=experiment_vars)
        steps_count = []

        for user in range(user_context.shape[0]):
            beta_thompson = draw_posterior_sample(hypo_model_params,mean_pre,
                        cov_pre, a_pre, b_pre)

            #beta_thompson_all.append([beta_thompson[i] for i in beta_thompson.keys()])
            beta_thompson_all.append(beta_thompson)
            hypo_optimal_action = making_decision.pick_hypo_optimal_arm(
                                                    beta_thompson,
                                                    user_context.iloc[user],
                                                    experiment_vars,
                                                    bandit_arms)
            received_reward = dmodels.true_model_output(true_coeff,
                                                    experiment_vars,
                                                    user_context.iloc[user],
                                                    hypo_optimal_action[0],
                                                    noise_stats)
            received_reward_no_noise = dmodels.true_model_output(true_coeff,
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
            selected_arm = pd.DataFrame([hypo_optimal_action[0]],
                    columns=experiment_vars)
            selected_arms_per_day = pd.concat([selected_arms_per_day, selected_arm],
                    ignore_index = True)

            steps_count.append(received_reward)

            y_batch.append(received_reward)

            regret_all.append(regret)
            true_optimal_action_all.append(true_optimal_action[0])
            hypo_optimal_action_all.append(hypo_optimal_action[0])
            true_reward_all.append(received_reward)
            hypo_reward_all.append(hypo_optimal_action[1])

        selected_arms_per_day.index = user_context.index
        #update contextual variables
        X = dmodels.calculate_hypo_regressors(hypo_model_params,
                    pd.concat([user_context,selected_arms_per_day],axis=1))
        X_batch = pd.concat([X_batch, X])
        user_context_all = pd.concat([user_context_all, user_context])

        if day%batch_day == 0:

            dependant_var = pd.Series(y_batch)
            dependant_var.index = user_context.index

            thompson_dist = calculate_posteriors_nig(dependant_var,
                            X_batch, mean_pre, cov_pre, a_pre,b_pre)
            mean_pre = thompson_dist[3]
            cov_pre = thompson_dist[2]
            a_pre = thompson_dist[0]
            b_pre = thompson_dist[1]
            y_batch = []
            X_batch = pd.DataFrame()

        #update user_context
        last_seven_steps = user_context_all.loc[user_context_all['Date']==(pd.to_datetime(user_context['Date'])-pd.DateOffset(5))[0]]['yesterday_steps']
        if len(last_seven_steps) == 0:
            last_seven_steps = pd.Series(np.zeros(shape=(user_context.shape[0])), index=user_context.index)
        #pd.set_option('display.max_columns', 500)
        user_context = next_day_user_context(user_context, steps_count, last_seven_steps, selected_arms_per_day)

    return [true_optimal_action_all,
            hypo_optimal_action_all,
            regret_all, 
            true_reward_all,
            hypo_reward_all,
            beta_thompson_all,
            user_context_all]



def next_day_user_context(user_context, today_steps_count, seven_days_ago_step_count, selected_arm):
    next_day_user_context = user_context.copy()
    next_day_user_context['Date'] = pd.to_datetime(user_context['Date'])+pd.DateOffset(1)
    next_day_user_context['day_mon'] = user_context['day_sun']
    next_day_user_context['day_tue'] = user_context['day_mon']
    next_day_user_context['day_wed'] = user_context['day_tue']
    next_day_user_context['day_thu'] = user_context['day_wed']
    next_day_user_context['day_fri'] = user_context['day_thu']
    next_day_user_context['day_sat'] = user_context['day_fri']
    next_day_user_context['day_sun'] = user_context['day_sat']
    next_day_user_context['day_weekend'] = next_day_user_context['day_sat'] + next_day_user_context['day_sun']
    next_day_user_context['yesterday_steps'] = today_steps_count
    next_day_user_context['week_steps'] = user_context['week_steps'] + today_steps_count - seven_days_ago_step_count
    next_day_user_context['yesterday_progress'] = today_steps_count / next_day_user_context['daily_goal']
    next_day_user_context['week_progress'] = next_day_user_context['week_steps'] / next_day_user_context['weekly_goal']
    if 'T1' in selected_arm.columns:
        next_day_user_context['days-since-T1'] = selected_arm['T1'] + user_context['days-since-T1'].apply(lambda x: x+1 if x>=0 else x)* (1-selected_arm['T1'])
        next_day_user_context['yesterday-sent-T1'] = selected_arm['T1']
    else:
        T1_sent = 1 - (selected_arm['T2']+selected_arm['T3']+selected_arm['T4'])
        next_day_user_context['days-since-T1'] = T1_sent + user_context['days-since-T1'].apply(lambda x: x+1 if x>=0 else x) * (1-T1_sent)
        next_day_user_context['yesterday-sent-T1'] = T1_sent 

    if 'T2' in selected_arm.columns:
        next_day_user_context['yesterday-sent-T2'] = selected_arm['T2']
        next_day_user_context['days-since-T2'] = selected_arm['T2'] + user_context['days-since-T2'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['T2'])
    else:
        T2_sent = 1 - (selected_arm['T1']+selected_arm['T3']+selected_arm['T4'])
        next_day_user_context['days-since-T2'] = T2_sent + user_context['days-since-T2'].apply(lambda x: x+1 if x>=0 else x) * (1-T2_sent)
        next_day_user_context['yesterday-sent-T2'] = T2_sent 

    if 'T3' in selected_arm.columns:
        next_day_user_context['yesterday-sent-T3'] = selected_arm['T3']
        next_day_user_context['days-since-T3'] = selected_arm['T3'] + user_context['days-since-T3'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['T3'])
    else:
        T3_sent = 1 - (selected_arm['T1']+selected_arm['T2']+selected_arm['T4'])
        next_day_user_context['days-since-T3'] = T3_sent + user_context['days-since-T3'].apply(lambda x: x+1 if x>=0 else x) * (1-T3_sent)
        next_day_user_context['yesterday-sent-T3'] = T3_sent 

    if 'T4' in selected_arm.columns:
        next_day_user_context['yesterday-sent-T4'] = selected_arm['T4']
        next_day_user_context['days-since-T4'] = selected_arm['T4'] + user_context['days-since-T4'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['T4'])
    else:
        T4_sent = 1 - (selected_arm['T1']+selected_arm['T2']+selected_arm['T3'])
        next_day_user_context['days-since-T4'] = T4_sent + user_context['days-since-T4'].apply(lambda x: x+1 if x>=0 else x) * (1-T4_sent)
        next_day_user_context['yesterday-sent-T4'] = T4_sent 

    if 'M0' in selected_arm.columns:
        next_day_user_context['yesterday-sent-M0'] = selected_arm['M0']
        next_day_user_context['days-since-M0'] = selected_arm['M0'] + user_context['days-since-M0'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['M0'])
    else:
        M0_sent = 1 - (selected_arm['M1']+selected_arm['M2']+selected_arm['M3'])
        next_day_user_context['days-since-M0'] = M0_sent + user_context['days-since-M0'].apply(lambda x: x+1 if x>=0 else x) * (1-M0_sent)
        next_day_user_context['yesterday-sent-M0'] = M0_sent 

    next_day_user_context['yesterday-sent-M1'] = selected_arm['M1']
    next_day_user_context['days-since-M1'] = selected_arm['M1'] + user_context['days-since-M1'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['M1'])

    next_day_user_context['yesterday-sent-M2'] = selected_arm['M2']
    next_day_user_context['days-since-M2'] = selected_arm['M2'] + user_context['days-since-M2'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['M2'])

    next_day_user_context['yesterday-sent-M3'] = selected_arm['M3']
    next_day_user_context['days-since-M3'] = selected_arm['M3'] + user_context['days-since-M3'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['M3'])

    if 'F0' in selected_arm.columns:
        next_day_user_context['yesterday-sent-F0'] = selected_arm['F0']
        next_day_user_context['days-since-F0'] = selected_arm['F0'] + user_context['days-since-F0'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['F0'])
    else:
        F0_sent = 1 - (selected_arm['F1']+selected_arm['F2']+selected_arm['F3']+selected_arm['F4'])
        next_day_user_context['days-since-F0'] = F0_sent + user_context['days-since-F0'].apply(lambda x: x+1 if x>=0 else x) * (1-F0_sent)
        next_day_user_context['yesterday-sent-F0'] = F0_sent 

    next_day_user_context['yesterday-sent-F1'] = selected_arm['F1']
    next_day_user_context['days-since-F1'] = selected_arm['F1'] + user_context['days-since-F1'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['F1'])

    next_day_user_context['yesterday-sent-F2'] = selected_arm['F2']
    next_day_user_context['days-since-F2'] = selected_arm['F2'] + user_context['days-since-F2'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['F2'])

    next_day_user_context['yesterday-sent-F3'] = selected_arm['F3']
    next_day_user_context['days-since-F3'] = selected_arm['F3'] + user_context['days-since-F3'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['F3'])

    next_day_user_context['yesterday-sent-F4'] = selected_arm['F4']
    next_day_user_context['days-since-F4'] = selected_arm['F4'] + user_context['days-since-F4'].apply(lambda x: x+1 if x>=0 else x)*(1-selected_arm['F4'])

    return next_day_user_context