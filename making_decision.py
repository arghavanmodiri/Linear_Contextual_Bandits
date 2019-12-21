import numpy as np
import numpy.random as nprnd
import true_hypo_models as models


def pick_hypo_optimal_arm(estimated_hypo_coeff, user_context, experiment_vars,
                            bandit_arms):
    """
    Find the optimal arms to be applied to the user based on its user_context.

    Args:
        user_context (ndarray): 2D array containing users contextual values
        bandit_arms (ndarray): 2D array containing all arms
        noise_mean (float): mean of the white noise in true model
        noise_std (float): std of the white noise in true model

    Returns:
        list: containing the optimal arm and its regret
    """
    optimal_arm = []
    optimal_arm_est_output = -100000
    for arm in bandit_arms:
        temp_result = models.hypo_model_output(estimated_hypo_coeff, experiment_vars, user_context, arm)
        if( abs(temp_result-optimal_arm_est_output)<0.00000001):
            optimal_arm.append(arm)
        elif(temp_result > optimal_arm_est_output):
            optimal_arm = [arm]
            optimal_arm_est_output = temp_result

    optimal_arm = optimal_arm[nprnd.randint(len(optimal_arm))]

    return [optimal_arm, optimal_arm_est_output]


def pick_true_optimal_arm(true_coeff, user_context, experiment_vars,
                            bandit_arms):
    """
    Find the optimal arms to be applied to the user based on its user_context.

    Args:
        user_context (ndarray): 2D array containing users contextual values
        bandit_arms (ndarray): 2D array containing all arms
        noise_mean (float): mean of the white noise in true model
        noise_std (float): std of the white noise in true model

    Returns:
        list: containing the optimal arm and its regret
    """
    optimal_arm = []
    optimal_arm_est_output = -100000

    for arm in bandit_arms:
        temp_result = models.true_model_output(true_coeff, experiment_vars,
                                                user_context, arm,
                                                {"noise_mean": 0,
                                                "noise_std": 0.0})
        if(abs(temp_result-optimal_arm_est_output)<0.00000001):
            optimal_arm.append(arm)
        elif(temp_result > optimal_arm_est_output):
            optimal_arm = [arm]
            optimal_arm_est_output = temp_result

    return [optimal_arm, optimal_arm_est_output]


def calculate_regret(optimal_reward, received_reward):
    return (optimal_reward - received_reward)
