import numpy as np
import true_hypo_models as models
import making_decision



def apply_random_sampling(user_context, experiment_vars, bandit_arms,
                            hypo_model_params, true_coeff, batch_size,
                            extensive, noise_stats):

    user_count = user_context.shape[0]
    # Size of action set
    action_count = len(bandit_arms)
    # Save outcomes per batch
    received_reward_all = []
    rand_regret_all = []

    # Save actions over time
    applied_action_all = []

    # Loop over all users
    for user in range(0, user_count):

        # Randomly assign action
        action_index = np.random.randint(0,action_count)
        rand_action = bandit_arms[action_index]

        # Compute outcome (from true model) using action and context
        received_reward = models.true_model_output(true_coeff,
                                                    experiment_vars,
                                                    user_context.iloc[user],
                                                    rand_action,
                                                    noise_stats)
        received_reward_no_noise = models.true_model_output(true_coeff,
                                                    experiment_vars,
                                                    user_context.iloc[user],
                                                    rand_action,
                                                    {"noise_mean": 0,
                                                    "noise_std": 0.0})
        true_optimal_action = making_decision.pick_true_optimal_arm(
                                                    true_coeff,
                                                    user_context.iloc[user],
                                                    experiment_vars,
                                                    bandit_arms)
        # Update outcomes list
        received_reward_all.append(received_reward)


        # Compute regret
        rand_regret = making_decision.calculate_regret(true_optimal_action[1], 
                                                    received_reward_no_noise)
        rand_regret_all.append(rand_regret)

        # Update actions list
        applied_action_all.append(bandit_arms[action_index])
        #print("X = ",user_context[user]['x1'], "D = ",rand_action, rand_regret)

    return [received_reward_all,applied_action_all,rand_regret_all]