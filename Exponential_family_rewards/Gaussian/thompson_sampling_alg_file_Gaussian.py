### TS algorithm

import numpy as np

def thom_samp_alg(arms,seed_val,hori):
    
    np.random.seed(seed_val)
    curr_time = 0
    num_arms = len(arms)
    arm_chosen = 0
    arms_index = np.arange(0,len(arms))
    # tot_rew =  np.zeros((hori,len(arms)))
    tot_rew =  np.zeros((hori,num_arms))
    rewards = np.zeros((hori,len(arms)))
    times_sampled = np.ones((num_arms))
    # times_sampled_hori = np.zeros((hori,len(arms)))
    mean_par = np.zeros((num_arms))
    variance_par = np.zeros((num_arms))
    gaussian_dist_samp = np.zeros((num_arms))

    all_arms_list = []
    all_arms_list = [a for a in range(num_arms)]
   
    # main body of algorithm
    while curr_time < hori:

        mean_par[:] = tot_rew[curr_time-1]/(times_sampled +1)
        variance_par[:] = np.sqrt(1/(times_sampled + 1))
        
        # mu_post = tot_rew[curr_time-1]/(times_sampled + 1) 
        # var_post = 1/(times_sampled + 1)
        gaussian_dist_samp[:] = np.random.normal(mean_par,variance_par)
#            print(beta_dist_samp[curr_time])

        # first_argmax = np.argmax(beta_dist_samp[:])
        # all_argmax = np.argwhere(beta_dist_samp[:] == beta_dist_samp[curr_time,first_argmax])
        # all_argmax = np.transpose(all_argmax)
        arm_chosen = np.argmax(gaussian_dist_samp[:])

        # print(arm_chosen)
        rewards[curr_time,arm_chosen] = np.random.normal(arms[arm_chosen],1)
        tot_rew[curr_time] = tot_rew[curr_time-1] + rewards[curr_time]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled

        # emp_means[curr_time] = np.divide(tot_rew[curr_time],times_sampled)
        curr_time = curr_time + 1

    return(np.sum(tot_rew,1)) #,times_sampled_hori)
