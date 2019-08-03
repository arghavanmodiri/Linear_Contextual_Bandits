import numpy as np
import numpy.random as nprnd

def read_true_model(true_model_params_file='True_Model_Coefficients.csv'):
    """
    Read the csv file containing the true model coefficients for all variables

    Args:
        true_model_params_file (str): name of the file

    Returns:
        ndarray: containing the coefficients of the true model
    """
    #noise = np.array([["noise_mean", 0],["noise_std", 5]], dtype=object)
    noise = {"noise_mean": 0, "noise_std": 100}
    '''true_coeff = {"bias": 40,
                    "ch2": 10,
                    "ch3": 20, 
                    "matching": 15,#10,
                    "republic": -10,
                    "matching*ch2": -5,
                    "matching*ch3": 5,
                    "republic*matching": -5}
    context_vars = np.array(['republic'])
    experiment_vars = np.array(['ch2','ch3', 'matching'])'''
    true_coeff = {"intercept": 0,
                    "d1": 0.3,
                    "d1*x1": -0.6}

    context_vars = np.array(['x1'])
    experiment_vars = np.array(['d1'])

    true_model_params = {'noise':noise,
                        'true_coeff': true_coeff,
                        'context_vars': context_vars,
                        'experiment_vars': experiment_vars}
    return true_model_params


def find_possible_actions(experiment_vars):
    """
    Read the csv file containing the true model coefficients for all variables

    Args:
        true_model_params_file (str): name of the file

    Returns:
        list: List of all possible actions
    """
    action_space = {var : [0, 1] for var in experiment_vars}
    all_possible_actions = [{}]

    for cur in sorted(action_space):

      # Store set values corresponding to action labels
      cur_options = action_space[cur]

      # Initialize list of feasible actions
      new_possible = []

      # Itterate over action set
      for a in all_possible_actions:

        # Itterate over value sets correspdong to action labels
        for cur_a in cur_options:
          new_a = a.copy()
          new_a[cur] = cur_a

          # Check if action assignment is feasible
          if is_valid_action(new_a):

            # Append feasible action to list
            new_possible.append(new_a)
            all_possible_actions = new_possible

    n_actions = len(all_possible_actions)

    possible_actions = [list(all_possible_actions[i].values()) for i in range(0, n_actions)]
    
    return possible_actions

def is_valid_action(action):
    '''
    checks whether an action is valid, meaning, no more than one vars under same category are assigned 1
    '''

    keys = action.keys()

    for cur_key in keys:
        if '_' not in cur_key:
            continue
        value = 0
        prefix = cur_key.rsplit('_', 1)[0] + '_'
        for key in keys:
            if key.startswith(prefix):
                value += action[key]
        if value > 1:
            return False

    return True


def read_hypo_model(hypo_model_params_file='Hypo_Model_Design.csv'):
    """
    Read the csv file containing the hypo model design. 1 indicats that the
    parameters will be available in hypothesized model and 0 otherwise.

    Args:
        true_model_params_file (str): name of the file

    Returns:
        list: containing the available parameters in hypothesized model
    """
    #hypo_model_params = ['bias', 'ch2','ch3', 'matching', 'republic']
    hypo_model_params = ['intercept','d1']
        #, 'd1*x1']

    '''hypo_model_params =["bias",
                        "ch2",
                        "ch3", 
                        "matching",
                        "republic",
                        "republic*matching"]
    '''
    return hypo_model_params

def read_independent_model(experiment_vars):
    prefix = []
    for var in experiment_vars:
        prefix.append(var.rsplit('_', 1)[0])
    prefix = set(prefix)

    hypo_params_independent = []
    for pre in prefix:
        sub_list = ['intercept']
        if pre == "intercept":
            continue
        i = 1
        multi_levels = pre + '_' + str(i)
        if multi_levels not in experiment_vars:
            sub_list.append(pre)
        while multi_levels in experiment_vars:
            sub_list.append(multi_levels)
            i += 1
            multi_levels = pre + '_' + str(i)
        hypo_params_independent.append(sub_list)

    return hypo_params_independent


def true_model_output(true_coeff, experiment_vars, user_context, applied_arm,
                        noise_stats):
    """
    Calculates the true donation for the specified user and arm

    Args:
        true_coeff (dict): coefficients of the true model
        user_context (dict): 1D array containing user contextual values
        bandit_arm (dict): 1D array containing the applied arm
        noise_stats (dict): mean and std of the white noise in true model

    Returns:
        float: the true value of the dependant variable
    """
    applied_arm_dict = {experiment_vars[i]:applied_arm[i] for i
                        in range(0, len(applied_arm))}
    user_params = {**user_context, **applied_arm_dict} 
    dependant_var = 0

    for coeff_name, coeff_value in true_coeff.items():
        temp = 0
        if(coeff_name == 'intercept'):
            temp += coeff_value
        elif('*' in coeff_name):
            interact_vars = coeff_name.split('*')
            temp = 1
            for var in interact_vars:
                temp = temp * user_params[var]
            temp = coeff_value * temp
        else:
            temp = coeff_value * user_params[coeff_name]
        dependant_var += temp

    added_noise = nprnd.normal(loc=noise_stats['noise_mean'], scale=noise_stats['noise_std'], size=1)[0]
    dependant_var = dependant_var + added_noise
    return dependant_var


def calculate_hypo_regressors(hypo_model_params, experiment_vars, user_context,
                                 applied_arm):
    applied_arm_dict = {experiment_vars[i]:applied_arm[i] for i
                        in range(0, len(applied_arm))}
    user_params = {**user_context, **applied_arm_dict} 
    X =[]
    for param in hypo_model_params:
        temp = 0
        if(param == 'intercept'):
            temp = 1
        elif('*' in param):
            interact_vars = param.split('*')
            temp = 1
            for var in interact_vars:
                temp = temp * user_params[var]
        else:
            temp = user_params[param]
        X.append(temp)

    return X

def hypo_model_output(estimated_coeff, experiment_vars, user_context,
                        applied_arm):
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
    applied_arm_dict = {experiment_vars[i]:applied_arm[i] for i
                        in range(0, len(applied_arm))}
    user_params = {**user_context, **applied_arm_dict}
    dependant_var_estimate = 0

    for coeff_name, coeff_value in estimated_coeff.items():
        temp = 0
        if(coeff_name == 'intercept'):
            temp += coeff_value
        elif('*' in coeff_name):
            interact_vars = coeff_name.split('*')
            temp = 1
            for var in interact_vars:
                temp = temp * user_params[var]
            temp = coeff_value * temp
        else:
            temp = coeff_value * user_params[coeff_name]

        dependant_var_estimate = np.add(dependant_var_estimate, temp)

    return dependant_var_estimate


def generate_true_dataset(context_vars, user_count, dist_of_context, write_to_file=True):
    """
    Generate the users dataset randomly.

    Args:
        context_vars (ndarray): 1D array containing context variables
        user_count (int): number of users to be generated
        dist_of_context (dict): each key is a context variable and each value specifies its distribution
        write_to_file (bool): If True, the dataset will be stored in a file

    Returns:
        #TODO: WRITE THIS DESCRIPTION
        users_list (ndarray): 1D array of dictionaries
    """
    users_list = [{} for i in range(user_count)]
    for context in context_vars:
        dist = dist_of_context[context]  # Gets the distribution associated with a given context variable
        if dist[0] == 'bin':
            context_array = np.random.choice(2, user_count, p=[dist[1], dist[2]] )  # Sample binary outcomes with specified probabilities
        elif dist[0] == 'norm':
            context_array = np.random.normal(loc=dist[1], scale=dist[2], size=user_count)
        elif dist[0] == "beta":
            context_array = np.random.beta(a=dist[1], b=dist[2], size=user_count)
        else:   # Default to binary with 50/50 dist
            context_array = np.random.choice(2, user_count)
        for i in range(0, user_count):
            users_list[i].update({context: context_array[i]})
    return np.asarray(users_list)
