# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 19:37:54 2020

@author: leona
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 16:44:55 2020

@author: leona
A2C implementation for the cartpole

"""
import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import time
import os

import settings

def a2c_run(train_env,valid_env,M,comission,fold,gamma,n_epoch,asset_name): 
    
    ########################################################################################
    # IMPLEMENTATION OF  THE A2C
    ########################################################################################
    
    print('Running A2C implementation')
        
    # helper function to convert numpy arrays to tensors
    def t(x): return torch.from_numpy(x).float()
    
    # Actor module, categorical actions only
    class Actor(nn.Module):
        def __init__(self, state_dim, n_actions):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, n_actions),
                nn.Softmax()
            )
        
        def forward(self, X):
            return self.model(X)
        
    
    # Critic module
    class Critic(nn.Module):
        def __init__(self, state_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, X):
            return self.model(X)
        
    # Memory
    # Stores results from the networks, instead of calculating the operations again from states, etc.
    class Memory():
        def __init__(self):
            self.log_probs = []
            self.values = []
            self.rewards = []
            self.dones = []
    
        def add(self, log_prob, value, reward, done):
            self.log_probs.append(log_prob)
            self.values.append(value)
            self.rewards.append(reward)
            self.dones.append(done)
        
        def clear(self):
            self.log_probs.clear()
            self.values.clear()
            self.rewards.clear()
            self.dones.clear()  
        
        def _zip(self):
            return zip(self.log_probs,
                    self.values,
                    self.rewards,
                    self.dones)
        
        def __iter__(self):
            for data in self._zip():
                return data
        
        def reversed(self):
            for data in list(self._zip())[::-1]:
                yield data
        
        def __len__(self):
            return len(self.rewards)
              
    
    # train function
    def train(memory, q_val):
        values = torch.stack(memory.values)
        q_vals = np.zeros((len(memory), 1))
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            q_val = reward + gamma*q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning
            
        advantage = torch.Tensor(q_vals) - values
        
        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()
        
        actor_loss = (-torch.stack(memory.log_probs)*advantage.detach()).mean()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()
    # config,
    
    env = train_env
    state_dim = M
    action_set = [-1,0,1]
    n_actions = 3
    actor = Actor(state_dim, n_actions)
    critic = Critic(state_dim)
    adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
    adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
    memory = Memory()
    train_hist = []
    valid_hist = []
    tic = time.time()
    n_epochs = n_epoch
    max_valid = -1000000000
    
    for i in range(n_epochs):
        done = False
        total_reward = 0
        state = env.reset()
        steps = 0
        while not done:
            probs = actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            action_category = dist.sample()
            action = action_set[action_category.item()]
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            memory.add(dist.log_prob(action_category), critic(t(state)), reward, done)
            state = next_state
            
            # train if done or num steps > max_steps
            if done:
                last_q_val = critic(t(next_state)).detach().data.numpy()
                train(memory, last_q_val)
                memory.clear()
        #############################################################
        # Validation
        #############################################################   
        
        done = False
        total_reward_valid = 0
        state = valid_env.reset()
        steps = 0
        while not done:
            probs = actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            #action = dist.sample()-torch.tensor(1) # stochastic choice
            action = torch.tensor(np.where(probs.detach().numpy() == np.amax(probs.detach().numpy()))[0][0]) # deterministic policy
            action = action_set[action.detach().item()]
            next_state, reward, done, info = valid_env.step(action)
            total_reward_valid += reward
            steps += 1      
            state = next_state        
            # train if done or num steps > max_steps
        train_hist.append(np.sum(env.agent_returns))
        valid_hist.append(np.sum(valid_env.agent_returns))
        if i > 3:
            if valid_hist[-1]>max_valid:
                max_valid = valid_hist[-1]
                torch.save(actor.state_dict(), os.path.join(settings.SAVE_PARAMS, 'policy_A2C_parmeters.pt'))
        #############################################################  
        if i%1==0:
            toc = time.time()
            print('|Episode: '+str(i)+'|Total reward: ' + str(round(np.sum(env.agent_returns),0))+'|Valid: '+str(round(np.sum(valid_env.agent_returns),0))+'|Time: '+str(round(toc-tic,0))+'s|')
    
    np.save(os.path.join(settings.RESULTS_DIR, 'A2C_valid_hist'+str(fold)+'_'+asset_name+'.npy'),valid_hist)
    np.save(os.path.join(settings.RESULTS_DIR, 'A2C_r_train_hist'+str(fold)+'_'+asset_name+'.npy'),train_hist) 

def test_a2c_run(test_env,M,comission,fold,gamma,n_epoch,asset_name):   
    
    def t(x): return torch.from_numpy(x).float()
    
    
    class Actor(nn.Module):
        def __init__(self, state_dim, n_actions):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, n_actions),
                nn.Softmax()
            )
        
        def forward(self, X):
            return self.model(X)
        
    state_dim = M
    n_actions = 3
    actor = Actor(state_dim, n_actions)
    
    action_set = [-1,0,1]
    done = False
    total_reward_test = 0
    env = test_env
    state = env.reset()
    steps = 0
    cum_reward_hist = []
    actor.load_state_dict(torch.load(os.path.join(settings.SAVE_PARAMS, 'policy_A2C_parmeters.pt')))
    while not done:
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = torch.tensor(np.where(probs.detach().numpy() == np.amax(probs.detach().numpy()))[0][0])
        action = action_set[action.detach().item()]
        next_state, reward, done, info = env.step(action)
        total_reward_test += reward
        cum_reward_hist.append(total_reward_test)
        steps += 1      
        state = next_state  
    
    np.save(os.path.join(settings.RESULTS_DIR, 'A2C_agent_returns'+str(fold)+'_'+asset_name+'.npy'),env.agent_returns)
    np.save(os.path.join(settings.RESULTS_DIR, 'A2C_signals'+str(fold)+'_'+asset_name+'.npy'),env.position_history)
       