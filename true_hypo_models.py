def read_true_model(true_model_params_file='True_Model_Coefficients.csv'):
    """
    Read the csv file containing the true model coefficients for all variables

    Args:
        true_model_params_file (str): name of the file

    Returns:
        ndarray: containing the coefficients of the true model
    """
    pass

def read_hypo_model(hypo_model_params_file='Hypo_Model_Design.csv'):
    """
    Read the csv file containing the hypo model design. 1 indicats that the
    parameters will be available in hypothesized model and 0 otherwise.

    Args:
        true_model_params_file (str): name of the file

    Returns:
        ndarray: containing the available parameters in hypothesized model
    """
    pass


def true_model(true_coeff, user_context, applied_arm, noise_mean,
                noise_std):
    """
    Calculates the true donation for the specified user and arm

    Args:
        true_coeff (ndarray): coefficients of the true model
        user_context (ndarray): 1D array containing user contextual values
        bandit_arm (ndarray): 1D array containing the applied arm
        noise_mean (float): mean of the white noise in true model
        noise_std (float): std of the white noise in true model

    Returns:
        float: the true donation
    """
    pass


def hypo_model(estimated_coeff, user_context, applied_arm):
    """
    Calculates the estimated donation for the specified user and arm

    Args:
        estimated_coeff (ndarray): 1D array containing the estimated coeff
        bandit_arm (ndarray): 1D array containing the applied arm
        noise_mean (float): mean of the white noise in true model
        noise_std (float): std of the white noise in true model

    Returns:
        float: the hypothesized donation
    """
    pass


def generate_true_dataset(user_context, user_count, user_dist=[],
                            write_to_file=True):
    """
    Generate the users dataset randomly.

    Args:
        user_context (ndarray): 1D array containing user contextual values
        user_count (int): number of users to be generated
        user_dist (ndarray): probability of generating 1 for each contextual variable. If not especified, the probability is 0.5
        write_to_file (bool): If True, the dataset will be stored in a file

    Returns:
        float: the true donation
    """
    pass

