### e-TS algorithm,  e constant, e >= 1/K (ICML 23 paper)

import numpy as np
# from scipy.stats import uniform 
# from numpy.random import Generator, PCG64
# from scipy.stats import binom


def e_third_TS_alg(arms,seed_val,hori):

    
    np.random.seed(seed_val)
    curr_time = 0
    num_arms = len(arms)
    arm_chosen = 0
    arm_index = np.arange(0,num_arms)
    tot_rew =  np.zeros((hori,num_arms))
    rewards = np.zeros((hori,num_arms))
    times_sampled = np.ones(num_arms)
    # times_sampled_hori = np.zeros((hori,num_arms))
    mean_par = np.zeros((num_arms))
    variance_par= np.zeros((num_arms))
    gaussian_dist_samp = np.zeros((num_arms))
    # reward_sample = np.zeros((hori,num_arms))
    #kl_ucb = np.zeros((hori,num_arms))
    # ucb = np.zeros((hori,num_arms))
    # emp_means =  np.zeros((hori,num_arms)) + 0.001


    all_arms_list = []
    all_arms_list = [a for a in range(num_arms)]
   
    while curr_time < hori:

        mean_par[:] = tot_rew[curr_time-1]/(times_sampled +1)
        variance_par[:] = np.sqrt(1/(times_sampled + 1))
        
        bet = 1-1/3 
        
        bet_samp = np.random.binomial(1,bet) ## prob of success is p, so prob of 1 is bet
        

        if bet_samp == 0:
            # print(alpha_par[curr_time,:]+1,beta_par[curr_time,:]+1)
            # mu_post = (times_sampled*1/(times_sampled*1 + 1))*mean_par[:] + 1/(times_sampled*1 + 1)
            # var_post = 1/(times_sampled + 1)
            gaussian_dist_samp[:] = np.random.normal(mean_par,variance_par)
            arm_chosen_gaussian = np.argmax(gaussian_dist_samp[:])
            arm_chosen = arm_chosen_gaussian
        
        elif bet_samp == 1:
            # means_of_arms_dist = (alpha_par[:]+1)/(alpha_par[:]+1+beta_par[:]+1)
            arm_chosen_mu = np.argmax(mean_par[:])
            arm_chosen = arm_chosen_mu

        # print(arm_chosen)
        rewards[curr_time,arm_chosen] = np.random.normal(arms[arm_chosen],1)
        tot_rew[curr_time] = tot_rew[curr_time-1] + rewards[curr_time]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled
        # times_sampled_hori[curr_time,:] = times_sampled

        curr_time = curr_time + 1

    return(np.sum(tot_rew,1)) #,times_sampled_hori)

