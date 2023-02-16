# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 15:10:31 2023

@author: leona
"""

import os
import time
import gym
import torch
import numpy as np
from envs import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO,A2C,DQN,SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

def sb_run(train_env,valid_env,asset_name,epochs,model_name = 'PPO',parallelization = True):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(BASE_DIR,'params')
    time_step_ep = len(train_env.r)
    if parallelization:
        # For cpu parallelization in StableBaseline learning
        def make_env(seed):
            def _init():
                env = Monitor(
                    train_env,
                    os.path.join(f'{LOG_DIR}','monitor',f'{model_name}_sb_{asset_name}_{seed}'),
                    allow_early_resets=True
                )
                return env
            return _init
        num_cpu = 5
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    else:
        env = Monitor(
            train_env,
            os.path.join(f'{LOG_DIR}','monitor',f'{model_name}_sb_{asset_name}')
        )
    eval_callback = EvalCallback(
        env,
        best_model_save_path=os.path.join(f'{LOG_DIR}',f'best_{model_name}_sb_{asset_name}'),
        log_path=f'{LOG_DIR}/',
        eval_freq=100,
        deterministic=True,
        verbose=0,
        render=False
    ) 
    
    model_name = model_name
    
    policy = 'MlpPolicy'
    if model_name == 'PPO':
            model = PPO(
            policy,
            env,verbose = 1, batch_size = 256, n_steps = 256, gamma = 0.96, gae_lambda = 0.9, n_epochs = 20, ent_coef = 0.0, max_grad_norm = 0.5, vf_coef = 0.5, learning_rate = 5e-3, use_sde = False, clip_range = 0.4, policy_kwargs = dict(log_std_init=-2,ortho_init=False,activation_fn=torch.nn.ReLU,net_arch=[dict(pi=[300, 300], vf=[300, 300])])
        )
    elif model_name == 'A2C':
            model = A2C(
            policy,
            env,verbose = 1, learning_rate=0.002, n_steps=100, gamma = 0.95, vf_coef = 0.7,policy_kwargs= dict(net_arch=[300, 300]), seed = None
        )
    elif model_name == 'DQN':
            model = DQN(
            policy,
            env, verbose = 1, learning_rate= 2.3e-3, buffer_size=100000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99,target_update_interval=10,train_freq= 256,gradient_steps= 128, exploration_fraction=0.16, exploration_initial_eps=0.04, policy_kwargs= dict(net_arch=[300, 300]), seed = None
        )
    
    print(f"SB: {model_name} learning...")
    start_time = time.time()

    # We define the EvalCallback wrapper to save the best model            
    # Here the model learns using the provided environment in the Stable baseline Agent definition
    # We mutiply the number of epochs by the number of time periods to give the number of training steps
    model.learn(
        epochs*time_step_ep,
        callback=eval_callback,
        # tb_log_name='PPO'
    )

    env.close()

    time_duration = time.time() - start_time
    print(f"Finished Learning {time_duration:.2f} s")
    
def test_sb_run(test_env,asset_name,model_name):    
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    LOG_DIR = os.path.join(BASE_DIR,'code','params')
    RESULTS_DIR = os.path.join(BASE_DIR,'code','results')
    path = os.path.join(f'{LOG_DIR}',f'best_{model_name}_sb_{asset_name}','best_model')
    
    
    # Load agent:
    if model_name == 'PPO':
        model = PPO.load(path)
    elif model_name == 'A2C':    
        model = A2C.load(path)
    elif model_name == 'DQN':    
        model = DQN.load(path)
    elif model_name == 'SAC':    
        model = SAC.load(path)     
    
    done = False
    env = test_env
    state = env.reset()
    total_reward_test = 0
    steps = 0
    cum_reward_hist = []
    while not done:
        action = model.predict(state)
        action = action[0] -1
        next_state, reward, done, info = env.step(action)
        total_reward_test += reward
        cum_reward_hist.append(total_reward_test)
        steps += 1      
        state = next_state  
    
    np.save(os.path.join(RESULTS_DIR, f'{model_name}_agent_returns'+'_'+asset_name+'.npy'),env.agent_returns)
    np.save(os.path.join(RESULTS_DIR, f'{model_name}_signals'+'_'+asset_name+'.npy'),env.position_history)