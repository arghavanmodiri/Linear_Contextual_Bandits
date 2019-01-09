def pick_optimal_arm(user_context, bandit_arms, noise_mean, noise_std):
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
    pass


def calculate_regret(true_donation, hypo_donation):
    pass
