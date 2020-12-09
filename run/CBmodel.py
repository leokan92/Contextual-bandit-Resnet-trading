# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:10:05 2020

@author: leona

Testing Online Adaptative greedy

"""

import pandas as pd
import os
import numpy as np
import time
import sys
import matplotlib
import matplotlib.pyplot as plt
# adding the path and importing the function
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\RRL non vectorized'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\support'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning'
sys.path.insert(0,path) # adding the code path
#path =  os.getcwd()
#sys.path.insert(0,path)
from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
from contextualbandits.online import AdaptiveGreedy, BootstrappedTS, ActiveExplorer
from sklearn.linear_model import LogisticRegression
from copy import deepcopy

######################################################################
# Model parameters
######################################################################
def cb_model(train_env,valid_env,test_env,epochs,fold,asset_name):
    #Action set definition
    action_space = [-1,0,1]
    nchoices = len(action_space)
    
    beta_prior = ((3./nchoices, 4), 2) # until there are at least 2 observations of each class, will use this prior
    beta_prior_ucb = ((5./nchoices, 4), 2) # UCB gives higher numbers, thus the higher positive prior
    beta_prior_ts = ((2./np.log2(nchoices), 4), 2)
    
    base_algorithm = LogisticRegression(solver='lbfgs', warm_start=True,max_iter = 200000, random_state = 3)
    # Use neuralnetworks from sklearn
    model = BootstrappedTS(base_algorithm = base_algorithm, nchoices = nchoices)
    #model = AdaptiveGreedy(deepcopy(base_algorithm), nchoices=nchoices,decay_type='threshold',beta_prior = beta_prior)
    
    # Create X input
    
    train_hist = []
    test_hist = []
    max_valid = -1000000000
    tic = time.clock()

    #Generate state for vectorized update on the training and test

    state_mod_train = []
    action  = 1
    #Generate train data
    obs = train_env.reset()
    state = obs
    done = False
    while not done:
        state_mod_train.append(state)
        next_state, reward, done, info = train_env.step(action)
        state = next_state
        
    state_mod_train = np.array(state_mod_train)
    
    #Generate valid data
    state_mod_valid = []
    action  = 1
    obs = valid_env.reset()
    state = obs
    done = False
    while not done:
        state_mod_valid.append(state)
        next_state, reward, done, info = valid_env.step(action)
        state = next_state
        
    state_mod_valid = np.array(state_mod_valid)
    
    #Generate test data
    state_mod_test = []
    obs = test_env.reset()
    state = obs
    done = False
    while not done:
        state_mod_test.append(state)
        next_state, reward, done, info = test_env.step(action)
        state = next_state
        
    state_mod_test = np.array(state_mod_test)
    
    def generate_actions_train(state):
        action_batch = []
        action_batch.append(model.predict(state))
        return np.concatenate(action_batch)
    
    
    def binary_reward(reward):
        if reward>0:
            return 1
        elif reward<0:
            return 0
        elif reward==0:
            return 0 
    
    #Train and validate data.
    train_hist = []
    valid_hist = []
    max_valid = -100000000000
    
    state_batch = []
    action_batch = []
    reward_batch = []
    window_batch = 500000 # use sometimes
    #Train and validate data.
    for epoch in range(epochs):
        done = False
        state = train_env.reset()
        step = 0
        policy_train = generate_actions_train(state_mod_train)
        while not done:
            action = policy_train[step]
            action_batch.append(action.item())
            action  = action_space[int(action.item())]
            next_state, reward, done, info = train_env.step(action)    
            state_batch.append(state)
            reward_batch.append(binary_reward(reward))
            state = next_state
            step += 1
        #if epoch%10==0: #update after some steps over the time series
        model.fit(np.array(state_batch[-window_batch:]), np.array(action_batch[-window_batch:]), np.array(reward_batch[-window_batch:])) #generates another batch of actions to be used
        toc = time.clock()
        
        
        done = False
        state = valid_env.reset()
        step = 0
        policy_valid = generate_actions_train(state_mod_valid)
        while not done:
            action = policy_valid[step]
            action  = action_space[int(action.item())]
            next_state, reward, done, info = valid_env.step(action)   
            state = next_state#generates another batch of actions to be used
            step += 1
            
        if epoch>50:
            if max_valid<(np.sum(valid_env.agent_returns)):
                best_oracle = model._oracles
                max_valid = np.sum(valid_env.agent_returns)
        
        train_hist.append(sum(train_env.agent_returns))
        valid_hist.append(np.sum(valid_env.agent_returns))
        
        print('Episode: ' +str(epoch) + ' | Train cR: '+str(round(sum(train_env.agent_returns)))+' | Valid cR: '+str(round(sum(valid_env.agent_returns)))+' | Time: '+str(round(toc-tic,0))+'s|')
    #############################################################
    # Test
    #############################################################   
        
    done = False
    state = test_env.reset()
    step = 0
    model._oracles = best_oracle
    policy_test = generate_actions_train(state_mod_test)
    while not done:
        action = policy_test[step]
        action  = action_space[int(action.item())]
        next_state, reward, done, info = test_env.step(action)   
        state_batch.append(state)
        reward_batch.append(binary_reward(reward))
        state = next_state
        step += 1
    
    #############################################################  
    
    toc = time.clock()
    print('Total train reward: ' + str(round(np.sum(train_env.agent_returns),0))+'|Test reward: '+str(round(np.sum(test_env.agent_returns),0))+'|Time: '+str(round(toc-tic,0))+'s|')

    np.save('cbadapt_valid_hist'+str(fold)+'_'+asset_name+'.npy',valid_hist)
    np.save('cbadapt_r_train_hist'+str(fold)+'_'+asset_name+'.npy',train_hist)
    np.save('cbadapt_agent_returns'+str(fold)+'_'+asset_name+'.npy',test_env.agent_returns)
    np.save('cbadapt_signals'+str(fold)+'_'+asset_name+'.npy',test_env.position_history)
       
# plt.plot(train_hist)
# plt.title('Train accumulated reward x epochs')
# plt.xlabel('epochs')
# plt.ylabel('Reward sum')
# plt.show()
# plt.close()

# plt.plot(valid_hist)
# plt.title('Valid accumulated reward x epochs')
# plt.xlabel('epochs')
# plt.ylabel('Reward sum')
# plt.show()
# plt.close()

# plt.plot(np.cumsum(test_env.agent_returns),label='CB')
# plt.plot(np.cumsum(test_env.r[M:]),label = 'BH')
# plt.title('Test returns x steps')
# plt.xlabel('Step')
# plt.ylabel('Reward')
# plt.show()
# plt.close()

#'/home/jovyan/work/readonly/spx_holdings_and_spx_closeprice.csv



    