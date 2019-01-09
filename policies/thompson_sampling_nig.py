def calculate_posteriors_nig(donations, user_context, applied_arms,
                            mean_pre, cov_pre, a_pre,b_pre):
    """
    Calculate the posterior probablibity after observing the true donation by
    appliying applied_arm to user with user_context value and assuming prior
    distribution is Normal-Inverse-Gamma(NIG).

    Args:
        donations (ndarray): 1D array containing the true donations
        user_context (ndarray): 2D array containing users' contextual values
        applied_arms (ndarray): 2D array containing the arm that took effect
        mean_pre (ndarray): 1D array containing mean of NIG priors.
        cov_pre (ndarray): 2D array containing covariance of NIG priors.
        a_pre (ndarray): shape of inverse-gamma distribution
        b_pre (ndarray): scale of inverse-gamma distribution

    Returns:
        list: containing [mean,cov,a,b] of posteriors distribution
    """
    pass


def draw_posterior_sample(mean, cov, a, b):
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
    pass


def apply_thompson_sampling(donations, user_context, bandit_arms, batch_size,
                            extensive, mean_pre, cov_pre, a_pre,b_pre):
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
    pass