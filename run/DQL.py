# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 11:07:57 2020

@author: leona
"""

"""
This is the RRL implementation following the Markov Decision process frame work used in many works.
Leonardo Kanashiro Felizardo
"""


import pandas as pd
import numpy as np
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import matplotlib.pyplot as plt
import time 
import sys
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

def dql_run(train_env,valid_env,M,comission,decay,fold,gamma,n_epoch,asset_name): 
    ###################################################################################
    # IMPLEMENTATION OF DQL
    ###################################################################################
    
    print('Running DQL implementation')
    
    # if gpu is to be used
    
    env = train_env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))
    
    class ReplayMemory(object):
    
        def __init__(self, capacity):
            self.capacity = capacity
            self.memory = []
            self.position = 0
    
        def push(self, *args):
            """Saves a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = Transition(*args)
            self.position = (self.position + 1) % self.capacity
    
        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)
    
        def __len__(self):
            return len(self.memory)
     
    class DQN(nn.Module):
        def __init__(self, M):
            super(DQN, self).__init__()
            self.dense1 = nn.Linear(M,M)
            self.dense2 = nn.Linear(M,M)
            self.dense3 = nn.Linear(M,3)
            
        def forward(self, x):
            #print(x.shape)
            x = F.sigmoid(self.dense1(x))
            x = F.sigmoid(self.dense2(x))
            out = F.softmax(self.dense3(x))
            return out.to(device).view(-1,3)
        
    ############################################################################
    # Parameters and environment setup
    ############################################################################
            
    BATCH_SIZE = 64
    GAMMA = gamma
    EPS_START = 0.9
    EPS_END = 0.001
    EPS_DECAY = decay
    num_episodes = n_epoch
    
    
    F_policy = []
    F_policy.append(0)
    action_space = [-1,0,1]
    valid_hist = []
    train_hist = []
    update_rate = 30 # steps
    TARGET_UPDATE = 200 # steps
    
    tic = time.clock()
    
    policy_net = DQN(M).to(device)
    target_net = DQN(M).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(100000)
    
    decay_factor = 0
    eps_threshold = 0
    max_valid = -100000000
    
    def select_action(state,decay_factor):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * decay_factor / EPS_DECAY)    
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return torch.max(policy_net(state)[0],0)[1].item()
        else:
            return random.randrange(3) #hardcode on the number of actions possible {-1,0,1}
    
    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
    
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
       # state_batch = np.asarray(batch.state).reshape(BATCH_SIZE,-1)
        action_batch = torch.stack(batch.action)
        state_batch = torch.cat(batch.state)
        reward_batch = torch.cat(batch.reward)
    
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #state_action_values = policy_net(state_batch).gather(1, action_batch.long())
        state_action_values = policy_net(state_batch).gather(1, action_batch.view(-1,1))
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
        # Optimize the model
        optimizer.zero_grad()
        
        loss.backward()
        #print(loss)
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        
    for i_episode in range(1,num_episodes):
        # Initialize the environment and state
        obs = env.reset()
        state = obs.tolist()# re initialize the state 'x'
        #state = [1] + state + [F_policy[0]]
        state = torch.tensor(state, device=device, dtype=torch.float)
        sum_reward_train = 0
        decay_factor+=1
        for t in range(1, len(env.r)-M-1):
            #if t%1000 == 0:
                #print('Actual step: ', t,'/',len(env.r)-M-1)
            # concatenate the last action and the one in the beginning
            # Select and perform an action
            a = select_action(state,decay_factor)
            action = torch.tensor(action_space[a], device=device, dtype=torch.float)
            obs, reward, done, _ = env.step(action.item())
            sum_reward_train += reward
            
            reward = torch.tensor([reward], device=device)
            action = a
            reward = reward.float()
            F_policy.append(action)
            # to calculate the next state
            next_state = obs.tolist()
            #next_state = torch.cat((torch.tensor([1], device=device, dtype=torch.float),torch.tensor(next_state, device=device, dtype=torch.float),torch.tensor([F_policy[t-1]], device=device, dtype=torch.float)),0)
            #state = torch.tensor(state, device = device, dtype=torch.float)
            next_state = torch.tensor(next_state, device = device, dtype=torch.float)
            # Store the transition in memory
            memory.push(state.view(1,-1), torch.tensor(action, device = device), next_state.view(1,-1), reward)
            
            # Move to the next state
            state = next_state
    
            # Perform one step of the optimization (on the target network)
            if t % update_rate == 0:
                #print('Updating...')
                optimize_model()
            if t % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
        #cum_sum_train = np.cumsum(np.cumsum(env.agent_returns))[-1]
        train_hist.append(np.sum(env.agent_returns))
    ##################################################################
    # VALIDATION
    ##################################################################
        with torch.no_grad():    
            obs = valid_env.reset()
            state = obs.tolist()# re initialize the state 'x'
            #state = [1] + state + [F_policy[0]]
            state_tensor = torch.tensor(state, device=device, dtype=torch.float)
            state = state_tensor
            sum_reward_valid = 0
            for step in range(1, len(valid_env.r)-M-1):  
                a = torch.max(policy_net(state)[0],0)[1].item()
                action = torch.tensor(action_space[a], device=device)
                obs, reward, done, _ = valid_env.step(action.item())
                sum_reward_valid += reward
                reward = torch.tensor([reward], device=device)
                action = a
                reward = reward.float()
                F_policy.append(action)
                next_state = obs.tolist()
                #next_state = torch.cat((torch.tensor([1], dtype=torch.float),torch.tensor(next_state, dtype=torch.float),torch.tensor([F_policy[t-1]], dtype=torch.float)),0)
                next_state = torch.tensor(next_state, device = device, dtype=torch.float)
                state = next_state
            #cum_sum_valid = np.cumsum(np.cumsum(valid_env.agent_returns))[-1]
            valid_hist.append(np.sum(valid_env.agent_returns))
        if i_episode > 3:
            if valid_hist[-1]>max_valid:
                max_valid = valid_hist[-1]
                torch.save(policy_net.state_dict(),'policy_DQL_parmeters.pt')
    ##################################################################
        toc = time.clock()
        print('Episode: ', str(i_episode), '|Time: ',round(toc-tic,2),'s|','Train.: ',round(train_hist[-1],2), '|Valid.: ', round(valid_hist[-1],2))
        
        # Update the target network, copying all weights and biases in DQN
        
        #torch.cuda.empty_cache()
    print('Complete')
    #env.render()
    env.close()
    
    np.save('DQN_valid_hist'+str(fold)+'_'+asset_name+'.npy',valid_hist)
    np.save('DQN_r_train_hist'+str(fold)+'_'+asset_name+'.npy',train_hist)
    ####################################################################################
    # TESTING
    ####################################################################################
def test_dql_run(test_env,M,comission,decay,fold,gamma,n_epoch,asset_name):     
    
    class DQN(nn.Module):
        def __init__(self, M):
            super(DQN, self).__init__()
            self.dense1 = nn.Linear(M,M)
            self.dense2 = nn.Linear(M,M)
            self.dense3 = nn.Linear(M,3)
            
        def forward(self, x):
            #print(x.shape)
            x = F.sigmoid(self.dense1(x))
            x = F.sigmoid(self.dense2(x))
            out = F.softmax(self.dense3(x))
            return out.to(device).view(-1,3)
    
    
    env = test_env
    obs = env.reset()
    state = obs.tolist()# re initialize the state 'x'
    #state = [1] + state + [F_policy[0]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_tensor = torch.tensor(state, device=device, dtype=torch.float)
    state = state_tensor
    cum_reward_hist = []
    F_policy = []
    F_policy.append(0)
    action_space = [-1,0,1]
    total_reward_test = 0
    policy_net = DQN(M).to(device)
    policy_net.load_state_dict(torch.load('policy_DQL_parmeters.pt'))
    step = 0
    done = False
    while not done:      
        a = torch.max(policy_net(state)[0],0)[1].item()
        action = torch.tensor(action_space[a], device=device)
        obs, reward, done, _ = env.step(action.item())
        total_reward_test += reward
        reward = torch.tensor([reward], device=device)
        action = a
        reward = reward.float()
        F_policy.append(action)
        next_state = obs.tolist()
        #next_state = torch.cat((torch.tensor([1], dtype=torch.float),torch.tensor(next_state, dtype=torch.float),torch.tensor([F_policy[t-1]], dtype=torch.float)),0)
        next_state = torch.tensor(next_state, device = device, dtype=torch.float)
        cum_reward_hist.append(total_reward_test)
        state = next_state
        step += 1
    
    np.save('DQN_agent_returns_'+str(fold)+'_'+asset_name+'.npy',env.agent_returns)
    np.save('DQN_signals'+str(fold)+'_'+asset_name+'.npy',env.position_history)
    
    # plt.plot(np.cumsum(env.agent_returns))
    # plt.plot(np.cumsum(np.load('bh_returns.npy')))
    # plt.show()


