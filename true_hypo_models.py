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
    '''true_coeff = {"intercept": 0,
                    "d1": 0.3,
                    "d1*x1": -0.6}'''
    true_coeff = {"intercept": 1000,
                    "feedbackMsg1": 500,
                    "feedbackMsg2": 800,
                    "feedbackMsg3": 100,
                    "feedbackMsg4": 0,
                    "motivationMsg1": 1000,
                    "motivationMsg2": -200,
                    "motivationMsg3": 500,
                    "Sent Message Code": 0, #discrete 0 for now
                    "time1": 50,
                    "time2": 1000,
                    "time3": -100,
                    "daysSinceFeedbackMsg1": 1,
                    "daysSinceFeedbackMsg2": 1,
                    "daysSinceFeedbackMsg3": 1,
                    "daysSinceFeedbackMsg4": 1,
                    "daysSinceMotivationMsg1": 1,
                    "daysSinceMotivationMsg2": 1,
                    "daysSinceMotivationMsg3": 1,
                    "Age": 0, #discrete 0 for now
                    "Gender": 0, #0=female, 1=male
                    "Days-in-study": 0,
                    "Study-week": 0, #discrete 0 for now
                    "DaysText": 0, #discrete 0 for now
                    "WeeklyMood": 0, #discrete 0 for now
                    "Steps-yesterday": 0, #Continuoues 0 for now
                    "Steps-day-before": 0, #Continuoues 0 for now
                    "Step-difference": 0, #Continuoues 0 for now
                    "Steps7days": 0, #Continuoues 0 for now
                    "PercentageGoalWeek": 1000,
                    "PercentageGoalYesterday": 0, #Continuoues not 0 for now
                    "PHQBaseline": 0, # 0 for now
                    "BABaseline": 0, # 0 for now
                    "feedbackMsg1*motivationMsg1": 1,
                    "feedbackMsg1*motivationMsg2": 1,
                    "feedbackMsg1*motivationMsg3": 1,
                    "feedbackMsg2*motivationMsg1": 1,
                    "feedbackMsg2*motivationMsg2": 1,
                    "feedbackMsg2*motivationMsg3": 1,
                    "feedbackMsg3*motivationMsg1": 1,
                    "feedbackMsg3*motivationMsg2": 1,
                    "feedbackMsg3*motivationMsg3": 1,
                    "feedbackMsg4*motivationMsg1": 1,
                    "feedbackMsg4*motivationMsg2": 1,
                    "feedbackMsg4*motivationMsg3": 1,
                    "feedbackMsg1*time1": 1,
                    "feedbackMsg1*time2": 1,
                    "feedbackMsg1*time3": 1,
                    "feedbackMsg2*time1": 1,
                    "feedbackMsg2*time2": 1,
                    "feedbackMsg2*time3": 1,
                    "feedbackMsg3*time1": 1,
                    "feedbackMsg3*time2": 1,
                    "feedbackMsg3*time3": 1,
                    "feedbackMsg4*time1": 1,
                    "feedbackMsg4*time2": 1,
                    "feedbackMsg4*time3": 1,
                    "motivationMsg1*time1": 1,
                    "motivationMsg1*time2": 1,
                    "motivationMsg1*time3": 1,
                    "motivationMsg2*time1": 1,
                    "motivationMsg2*time2": 1,
                    "motivationMsg2*time3": 1,
                    "motivationMsg3*time1": 1,
                    "motivationMsg3*time2": 1,
                    "motivationMsg3*time3": 1
                    }
    context_vars = np.array(["Sent Message Code",
                            "daysSinceFeedbackMsg1",
                            "daysSinceFeedbackMsg2", 
                            "daysSinceFeedbackMsg3", 
                            "daysSinceFeedbackMsg4", 
                            "daysSinceMotivationMsg1", 
                            "daysSinceMotivationMsg2", 
                            "daysSinceMotivationMsg3", 
                            "Age",
                            "Gender",
                            "Days-in-study",
                            "Study-week",
                            "DaysText",
                            "WeeklyMood",
                            "Sent Message Code",
                            "Steps-yesterday",
                            "Steps-day-before",
                            "Step-difference",
                            "Steps7days",
                            "PercentageGoalYesterday",
                            "PercentageGoalWeek",
                            "PHQBaseline",
                            "BABaseline"])
    experiment_vars = np.array([
                            "feedbackMsg1",
                            "feedbackMsg2",
                            "feedbackMsg3",
                            "feedbackMsg4",
                            "motivationMsg1",
                            "motivationMsg2",
                            "motivationMsg3",
                            "time1",
                            "time2",
                            "time3"
                            ])

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
    # possible_actions =  [[0],[1]]
    possible_actions =  [
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 1, 1, 0, 0, 1, 0, 0],

                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 1, 0, 1, 0, 0, 1, 0, 0],

                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],

                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 1, 0, 0, 1, 0, 0],

                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]]


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
    #hypo_model_params = ['intercept','d1', 'd1*x1']
    hypo_model_params = [
                    "intercept",
                    "feedbackMsg1",
                    "feedbackMsg2",
                    "feedbackMsg3",
                    "feedbackMsg4",
                    "motivationMsg1",
                    "motivationMsg2",
                    "motivationMsg3",
                    "Sent Message Code",
                    "time1",
                    "time2",
                    "time3",
                    "daysSinceFeedbackMsg1",
                    "daysSinceFeedbackMsg2", 
                    "daysSinceFeedbackMsg3", 
                    "daysSinceFeedbackMsg4", 
                    "daysSinceMotivationMsg1", 
                    "daysSinceMotivationMsg2", 
                    "daysSinceMotivationMsg3", 
                    "Age",
                    "Gender",
                    "Days-in-study",
                    "Study-week", #discrete 0 for now
                    "DaysText", #discrete 0 for now
                    "WeeklyMood", #discrete 0 for now
                    "Steps-yesterday",
                    "Steps-day-before",
                    "Step-difference",
                    "Steps7days",
                    "PercentageGoalWeek",
                    "PercentageGoalYesterday",
                    "PHQBaseline",
                    "BABaseline",
                    "feedbackMsg1*motivationMsg1", 
                    "feedbackMsg1*motivationMsg2", 
                    "feedbackMsg1*motivationMsg3", 
                    "feedbackMsg2*motivationMsg1", 
                    "feedbackMsg2*motivationMsg2", 
                    "feedbackMsg2*motivationMsg3", 
                    "feedbackMsg3*motivationMsg1", 
                    "feedbackMsg3*motivationMsg2", 
                    "feedbackMsg3*motivationMsg3", 
                    "feedbackMsg4*motivationMsg1", 
                    "feedbackMsg4*motivationMsg2", 
                    "feedbackMsg4*motivationMsg3", 
                    "feedbackMsg1*time1", 
                    "feedbackMsg1*time2", 
                    "feedbackMsg1*time3", 
                    "feedbackMsg2*time1", 
                    "feedbackMsg2*time2", 
                    "feedbackMsg2*time3", 
                    "feedbackMsg3*time1", 
                    "feedbackMsg3*time2", 
                    "feedbackMsg3*time3", 
                    "feedbackMsg4*time1", 
                    "feedbackMsg4*time2", 
                    "feedbackMsg4*time3", 
                    "motivationMsg1*time1", 
                    "motivationMsg1*time2", 
                    "motivationMsg1*time3", 
                    "motivationMsg2*time1", 
                    "motivationMsg2*time2", 
                    "motivationMsg2*time3", 
                    "motivationMsg3*time1", 
                    "motivationMsg3*time2", 
                    "motivationMsg3*time3"]
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
    users_list = np.array([{context_vars[j]:nprnd.randint(2) for j in range(0,len(context_vars))} for i in range(0, user_count)])

    return users_list
