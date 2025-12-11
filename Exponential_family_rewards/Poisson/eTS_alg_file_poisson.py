###  TS decay, greedy limit, pow(curr_time/hori,1/2)

import numpy as np

#####################################################
    

def eTS_alg(arms,seed_val,hori):

    
    np.random.seed(seed_val)
    curr_time = 0
    num_arms = len(arms)
    arm_chosen = 0
    arm_index = np.arange(0,num_arms)
    tot_rew =  np.zeros((hori,num_arms))
    rewards = np.zeros((hori,num_arms))
    times_sampled = np.zeros(num_arms)
    # times_sampled_hori = np.zeros((hori,num_arms))
    alpha_par = np.zeros((num_arms))
    beta_par = np.zeros((num_arms))
    gamma_dist_samp = np.zeros((num_arms))
    # reward_sample = np.zeros((hori,num_arms))
    #kl_ucb = np.zeros((hori,num_arms))
    # ucb = np.zeros((hori,num_arms))
    # emp_means =  np.zeros((hori,num_arms)) + 0.001

    threshold = np.zeros((hori,num_arms))

    all_arms_list = []
    all_arms_list = [a for a in range(num_arms)]
   
    while curr_time < hori:

        alpha_par[:] = tot_rew[curr_time-1] + 1  # 
        beta_par[:] = times_sampled + 1  # 
        
        bet = 1-1/np.log(np.log(curr_time+16))
        
        bet_samp = np.random.binomial(1,bet) ## prob of success is p, so prob of 1 is bet

        if bet_samp == 0:
            # print(alpha_par[curr_time,:]+1,beta_par[curr_time,:]+1)
            gamma_dist_samp = np.random.gamma(alpha_par,1/beta_par)
            arm_chosen = np.argmax(gamma_dist_samp[:])
             
        elif bet_samp == 1:
            means_of_arms_dist = alpha_par[:]/beta_par[:]
            arm_chosen = np.argmax(means_of_arms_dist)
            
        # print(arm_chosen)
        rewards[curr_time,arm_chosen] = np.random.poisson(arms[arm_chosen],1)
        tot_rew[curr_time] = tot_rew[curr_time-1] + rewards[curr_time]
        times_sampled[arm_chosen] = times_sampled[arm_chosen] + 1
        # times_sampled_hori[curr_time,:] = times_sampled
        # times_sampled_hori[curr_time,:] = times_sampled

        curr_time = curr_time + 1

    return(np.sum(tot_rew,1)) #,times_sampled_hori)

