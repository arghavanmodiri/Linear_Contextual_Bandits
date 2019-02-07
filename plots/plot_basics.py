from datetime import date
import matplotlib.pyplot as plt

TODAY = date.today()

def plot_regret(user_count, policy_names, regrets_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, regrets_all_policies[idx], label = policy)


    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Regret', fontsize = 12)
    plt.title('Calculated Regret for Each User\n(Simulations = {sims}, Batch Size={batches})'.format(sims=simulation_count, batches=batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_regrets.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))


def plot_optimal_action_ratio(user_count, policy_names, optimal_action_ratio_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, optimal_action_ratio_all_policies[idx], label =
            policy)

    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Proportion of Optimal Action Assignment', fontsize = 12)
    plt.title('Proportion of Optimal Action at Each Itteration\n(Simulations =\
        {sims}, Batch Size={batches})'.format(sims=simulation_count, batches=
            batch_size),fontsize = 12)
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}\
            sims_optimal_action_ratio.png'.format(date=TODAY, i=user_count,
                sim=simulation_count))


def plot_mse(user_count, policy_names, mse_all_policies,
                simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for idx, policy in enumerate(policy_names):
        plt.plot(UserItter, mse_all_policies[idx], label = policy)

    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('MSE', fontsize = 12)
    plt.title('MSE at Each Itteration\n(Simulations = {sims},\
                Batch Size={batches})'.format(sims=simulation_count,
                batches=batch_size), fontsize = 12)

    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}sims_mse.png'.
            format(date=TODAY, i=user_count, sim=simulation_count))



def plot_coeff_ranking(user_count, policy_name, beta_thompson_coeffs,
                    hypo_params, simulation_count, batch_size, save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)

    for param in range(0,len(hypo_params)):
        plt.plot(UserItter, beta_thompson_coeffs[:,param],
                label = hypo_params[param])

    plt.legend(loc='upper left', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Hypo Coeff', fontsize = 12)
    plt.title('Coeff at Each Itteration for {policy}\n(Simulations = {sims},\
            Batch Size={batches})'.format(policy=policy_name ,
            sims=simulation_count, batches=batch_size),fontsize = 12)

    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}\
            sims_coeff_ranking.png'.format(date=TODAY, i=user_count, sim=
                simulation_count))


def plot_coeff_sign_error(user_count, policy_name, hypo_params,
                        coeff_sign_error, simulation_count, batch_size,
                        save_fig=True):
    plt.figure()
    UserItter = range(1,user_count+1)
    #plt.plot(UserItter, prop_best_sim_itter, label = "Contextual Policy")

    for param in range(0,len(hypo_params)):
        plt.plot(UserItter, coeff_sign_error[:,param], label = hypo_params[
                param])

    plt.legend(loc='upper right', fontsize = 12)
    plt.xlabel('User Itterations', fontsize = 18)
    plt.ylabel('Coeff Sign Error', fontsize = 12)
    plt.title('Coeff Sign Error at Each Itteration for {policy}\n(Simulations\
            = {sims}, Batch Size={batches})'.format(policy=policy_name,sims=
            simulation_count, batches=batch_size),fontsize = 12)
    #plt.show()
    if(save_fig):
        plt.savefig('saved_output//{date}_{i}iterations_{sim}\
                    sims_coeff_sign_err.png'.format(date=TODAY, i=user_count,
                    sim=simulation_count))