### TS algorithm

import numpy as np

# from scipy.stats import beta 
# import scipy.stats as sst
# import scipy.special

def thom_samp_alg(arms,seed_val,hori):

    np.random.seed(seed_val)
    curr_time = 0
    num_arms = len(arms)
    arm_chosen = 0
    arms_index = np.arange(0,len(arms))
    # tot_rew =  np.zeros((hori,len(arms)))
    tot_rew =  np.zeros((hori,num_arms))
    rewards = np.zeros((hori,len(arms)))
    times_sampled = np.zeros((num_arms))
    # times_sampled_hori = np.zeros((hori,len(arms)))
    alpha_par = np.zeros((num_arms))
    beta_par = np.zeros((num_arms))
    gamma_dist_samp = np.zeros((num_arms))

    #emp_means =  np.zeros((hori,len(arms)))

    all_arms_list = []
    all_arms_list = [a for a in range(num_arms)]
   
    # main body of algorithm
    while curr_time < hori:

        alpha_par[:] = times_sampled + 1  # scale parameter
        beta_par[:] = tot_rew[curr_time-1] + 1 # shape parameter
    
        gamma_dist_samp = np.random.gamma(alpha_par,1/beta_par)
        # print(alpha_par,1/beta_par,gamma_dist_samp)

        # first_argmax = np.argmax(beta_dist_samp[:])
        # all_argmax = np.argwhere(beta_dist_samp[:] == beta_dist_samp[curr_time,first_argmax])
        # all_argmax = np.transpose(all_argmax)
        arm_chosen = np.argmin(gamma_dist_samp[:])
        # print(arm_chosen)

        rewards[curr_time,arm_chosen] = np.random.gamma(arms[arm_chosen],1)
        tot_rew[curr_time] = tot_rew[curr_time-1] + rewards[curr_time]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled

        # emp_means[curr_time] = np.divide(tot_rew[curr_time],times_sampled)
        curr_time = curr_time + 1

    # print(tot_rew[curr_time-1])
    return(np.sum(tot_rew,1)) #,times_sampled_hori)
