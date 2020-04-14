import numpy as np
import pandas as pd
import numpy.random as nprnd
import logging

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(message)s')

def generate_true_dataset(context_vars, user_count, dist_of_context,
    batch_size = None, write_to_file=True):
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
        if type(context) == list or type(context) == np.ndarray:
            if context[0].startswith("day_"):
                context_array = np.zeros([user_count,len(context)-1],dtype=int)
                weekday = np.ones(user_count, dtype=int)
                pick_from_group = np.random.choice(len(context),
                    int(user_count/batch_size))
                pick_from_group = pick_from_group.repeat(batch_size)
                for i in range(user_count):
                    if pick_from_group[i] > 0:
                        context_array[i,pick_from_group[i]-1] = 1
                    if context[pick_from_group[i]] == 'day_sat' or context[pick_from_group[i]] == 'day_sun':
                        weekday[i] = 0
                '''
                for i in range(0, user_count):
                    for j in range(0, len(context)):
                        users_list[i].update({context[j]: context_array[i][j]})
                    users_list[i].update({'day_weekend': day_weekend[i]}) #hacky way for this variable!!
                '''
                context_df = pd.DataFrame(context_array)
                context_df.columns = context[1:]
                users_context_df =pd.concat([users_context_df, context_df], axis=1)
                users_context_df['weekday'] = weekday
        elif context == 'weekday': #hacky way for this variable!!!
            continue
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
