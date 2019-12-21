import numpy as np
import pandas as pd
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


def find_possible_actions(true_model_params_file='True_Model_Coefficients.csv'):
    """
    Read the csv file containing the true model coefficients for all variables

    Args:
        true_model_params_file (str): name of the file

    Returns:
        list: List of all possible actions
    """
    #possible_actions =  [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1]]
    possible_actions =  [[0],[1]]
    
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


def calculate_hypo_regressors(hypo_model_params, users_context_arm):
    '''
    user_context_arm (DataFrame): contains the context and applied arm for each
    user
    '''
    X = pd.DataFrame(index=users_context_arm.index)
    u_count = users_context_arm.shape[0]
    for param in hypo_model_params:
        if(param == 'intercept'):
            temp = pd.DataFrame(1.0,columns=['intercept'],index=range(u_count))
            temp.index = X.index
        elif('*' in param):
            interact_vars = param.split('*')
            #temp = users_context_arm.loc[:,interact_vars].prod(axis=1)
            temp = pd.DataFrame(users_context_arm.loc[:,interact_vars].prod(axis=1),columns=[param])
        else:
            temp = users_context_arm[param]
        X = pd.concat([X, temp], axis=1)

    return X


def hypo_model_output(estimated_coeff, experiment_vars, user_context,
                        applied_arm):
    """
    Calculates the estimated donation for the specified user and arm

    Args:
        estimated_coeff (dict): estimated coeff for each hypo param
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
    users_context_df = pd.DataFrame()
    for context in context_vars:
        if context == 'day_weekend': #hacky way for this variable!!!
            continue
        elif type(context) == list:
            group_context = "&".join(context)
            dist = dist_of_context[group_context]
            context_array = np.zeros([user_count, len(context)], dtype=int)
            day_weekend = np.zeros(user_count, dtype=int) #hacky way for this variable!!!
            if dist[0] == 'uniform':
                pick_from_group = np.random.choice(len(context), user_count)
            if dist[0] == 'uniform_or_no':
                pick_from_group = np.random.choice(len(context)+1, user_count)
            for i in range(user_count):
                if pick_from_group[i] < len(context):
                    context_array[i,pick_from_group[i]] = 1
                    #hacky way for this variable!!!
                    if context[pick_from_group[i]] == 'day_sat' or context[pick_from_group[i]] == 'day_sun':
                        day_weekend[i] = 1
            '''
            for i in range(0, user_count):
                for j in range(0, len(context)):
                    users_list[i].update({context[j]: context_array[i][j]})
                users_list[i].update({'day_weekend': day_weekend[i]}) #hacky way for this variable!!
            '''
            context_df = pd.DataFrame(context_array)
            context_df.columns = context
            users_context_df =pd.concat([users_context_df, context_df], axis=1)
            users_context_df['day_weekend'] = day_weekend
        else:
            dist = dist_of_context[context]  # Gets the distribution associated with a given context variable
            if dist[0] == 'bin':
                context_array = np.random.choice(2, user_count, p=[dist[1], dist[2]] )  # Sample binary outcomes with specified probabilities
            elif dist[0] == 'norm':
                context_array = np.random.normal(loc=dist[1], scale=dist[2], size=user_count)
            elif dist[0] == "beta":
                context_array = np.random.beta(a=dist[1], b=dist[2], size=user_count)
            else:   # Default to binary with 50/50 dist
                context_array = np.random.choice(2, user_count)
            '''
            for i in range(0, user_count):
                users_list[i].update({context: context_array[i]})
            '''
            users_context_df[context] = context_array

    #return np.asarray(users_list)
    return users_context_df


# def generate_true_dataset(context_vars, user_count, is_binary_context=[], user_dist=[],
#                             write_to_file=True):
#     """
#     Generate the users dataset randomly.
#
#     Args:
#         user_context (ndarray): 1D array containing user contextual values
#         user_count (int): number of users to be generated
#         user_dist (ndarray): probability of generating 1 for each contextual variable. If not especified, the probability is 0.5
#         write_to_file (bool): If True, the dataset will be stored in a file
#
#     Returns:
#         float: the true donation
#     """
#     # users_list = np.array([{context_vars[j]:nprnd.randint(2) for j in range(0,len(context_vars))} for i in range(0, user_count)])
#     users_list = []
#
#     if not is_binary_context:
#         users_list = np.array(
#             [{context_vars[j]: nprnd.randint(2) for j in range(0, len(context_vars))} for i in range(0, user_count)])
#     else:
#         if len(is_binary_context) != len(context_vars):
#             print("Some contectual varaiables are not set as binary or not.")
#             print("len(is_binary_context) != len(context_vars)")
#         for idx in range(0, len(context_vars)):
#             if idx == 0:
#                 if is_binary_context[idx]:
#                     users_list = np.array([
#                         {context_vars[idx]: nprnd.randint(2)} for i in range(0,
#                                                                              user_count)])
#                 else:
#                     users_list = np.array([
#                         {context_vars[idx]: np.random.normal(0.7, 0.3)} for i in range(0, user_count)])
#
#             else:
#                 if is_binary_context[idx]:
#                     for user in range(0, user_count):
#                         users_list[user].update({
#                             context_vars[idx]: nprnd.randint(2)})
#                 else:
#                     for user in range(0, user_count):
#                         users_list[user].update(
#                             {context_vars[idx]: np.random.normal(0.7, 0.3)})
#     # test
#     return users_list
