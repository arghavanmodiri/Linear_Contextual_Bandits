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
                    "m1": 100,
                    "m2": 150, 
                    "f1": 100, 
                    "f2": 200,
                    "m1*f1": 250,
                    "m1*f2": 0,
                    "m2*f1": 0,
                    "m2*f2": 50}

    context_vars = np.array([])
    experiment_vars = np.array([["m1", "m2"], ["f1", "f2"]])

    true_model_params = {'noise':noise,
                        'true_coeff': true_coeff,
                        'context_vars': context_vars,
                        'experiment_vars': experiment_vars}
    return true_model_params


def find_possible_actions(true_model_params_file='True_Model_Coefficients.csv'):
    """
    Read the csv file containing the true model coefficients for all variables

    Args:
        true_model_params_file (str): name of the file

    Returns:
        list: List of all possible actions
    """
    #possible_actions =  [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
    possible_actions =  [[None, None], [None, "f1"], [None, "f2"], ["m1", None], ["m1", "f1"], ["m1", "f2"], ["m2", None], ["m2", "f1"], ["m2", "f2"]]
    
    return possible_actions


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
    hypo_model_params = ['intercept','m1', 'm2', 'f1', 'f2', 'm1*f1', 'm1*f2', 'm2*f1', 'm2*f2']
    # TODO read from csv
    if hypo_model_params_file == "Hypo_Model_No_Interection.csv":
        hypo_model_params = ['intercept','m1', 'm2', 'f1', 'f2']

    '''hypo_model_params =["bias",
                        "ch2",
                        "ch3", 
                        "matching",
                        "republic",
                        "republic*matching"]
    '''
    return hypo_model_params

# TODO get list of coefficients (same as hypo_model_params) to use calculate_hypo_regressors
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
    #applied_arm_dict = {experiment_vars[i]: applied_arm[i] for i in range(0, len(applied_arm))}
    #user_params = {**user_context, **applied_arm_dict} 
    dependant_var = true_coeff['intercept']

    for arm in applied_arm:
        if arm != None:
            dependant_var += true_coeff[arm]
            for i in range(applied_arm.index(arm) + 1, len(applied_arm)): # 2-way interactions
                if applied_arm[i] != None:
                    interaction = arm + "*" + applied_arm[i]
                    dependant_var += true_coeff[interaction]
            for cont in user_context: # 2-way interactions between action and context
                # TODO continuous contexts
                if cont != None:
                    interaction = arm + "*" + cont
                    dependant_var += true_coeff[interaction]

    '''for coeff_name, coeff_value in true_coeff.items():
        temp = 0
        if(coeff_name == 'intercept'):
            #temp += coeff_value
            pass
        elif('*' in coeff_name):
            interact_vars = coeff_name.split('*')
            temp = 1
            for var in interact_vars:
                temp = temp * user_params[var]
            temp = coeff_value * temp
        else:
            temp = coeff_value * user_params[coeff_name]
        dependant_var += temp'''

    added_noise = nprnd.normal(loc=noise_stats['noise_mean'], scale=noise_stats['noise_std'], size=1)[0]
    dependant_var = dependant_var + added_noise
    return dependant_var


def calculate_hypo_regressors(hypo_model_params, experiment_vars, user_context,
                                 applied_arm, interaction=True):
    # applied_arm_dict = {experiment_vars[i]:applied_arm[i] for i in range(0, len(applied_arm))}
    # user_params = {**user_context, **applied_arm_dict} 
    X = np.zeros(len(hypo_model_params))
    X[0] = 1

    for arm in applied_arm:
        if arm != None:
            index = hypo_model_params.index(arm)
            X[index] = 1
            if interaction:
                for i in range(applied_arm.index(arm) + 1, len(applied_arm)): # 2-way interactions between actions
                    if applied_arm[i] != None:
                        interaction_term = arm + "*" + applied_arm[i]
                        index = hypo_model_params.index(interaction_term)
                        X[index] = 1
                for cont in user_context: # 2-way interactions between action and context
                    # TODO continuous contexts
                    if cont != None:
                        interaction_term = arm + "*" + cont
                        index = hypo_model_params.index(interaction_term)
                        X[index] = 1

    '''X =[]
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
        X.append(temp)'''

    return X


# TODO get hypo_model_params to use calculate_hypo_regressors
def hypo_model_output(estimated_coeff, experiment_vars, user_context,
                        applied_arm, interaction=True):
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
    #applied_arm_dict = {experiment_vars[i]: applied_arm[i] for i in range(0, len(applied_arm))}
    #user_params = {**user_context, **applied_arm_dict}
    dependant_var_estimate = estimated_coeff['intercept']

    for arm in applied_arm:
        if arm != None:
            dependant_var_estimate += estimated_coeff[arm]
            if interaction:
                for i in range(applied_arm.index(arm) + 1, len(applied_arm)): # 2-way interactions
                    if applied_arm[i] != None:
                        interaction_term = arm + "*" + applied_arm[i]
                        dependant_var_estimate += estimated_coeff[interaction_term]
                for cont in user_context: # 2-way interactions between action and context
                    # TODO continuous contexts
                    if cont != None:
                        interaction_term = arm + "*" + cont
                        dependant_var_estimate += estimated_coeff[interaction_term]

    '''for coeff_name, coeff_value in estimated_coeff.items():
        temp = 0
        if(coeff_name == 'intercept'):
            #temp += coeff_value
            pass
        elif('*' in coeff_name):
            interact_vars = coeff_name.split('*')
            temp = 1
            for var in interact_vars:
                temp = temp * user_params[var]
            temp = coeff_value * temp
        else:
            temp = coeff_value * user_params[coeff_name]

        dependant_var_estimate = np.add(dependant_var_estimate, temp)'''

    return dependant_var_estimate


def generate_true_dataset(context_vars, user_count, user_dist=[],
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
    #users_list = np.array([{context_vars[j]:nprnd.randint(2) for j in range(0,len(context_vars))} for i in range(0, user_count)])

    #return users_list
    return np.array([[None]] * user_count)
