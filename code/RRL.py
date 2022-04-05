"""
This is the RRL implementation following the Markov Decision process frame work used in many works.
Leonardo Kanashiro Felizardo
"""


import pandas as pd
import numpy as np
from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio
import matplotlib.pyplot as plt
import time
import os

from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
import settings


###################################################################################
# IMPLEMENTATION OF RRL
###################################################################################

def rrl_run(train_env,valid_env,M,comission,mu,T,fold,n_epoch,asset_name):    
    
    print('Running RRL implementation')
    
    env = train_env
    T = T
    
    
    sigma = comission
    rho = 1
    x = []
    x_hist = []
    F = []
    F.append(0)
    R = np.zeros(T)
    np.random.seed(0)
    w = np.random.uniform(size=M+2)
    epoch_S = np.empty(0)
    S = 0
    n_epoch = n_epoch
    progress_period = 1
    
    
    dFdw   = np.zeros(M+2)
    dFpdw  = np.zeros(M+2)
    dSdw   = np.zeros(M+2)
    
    
    def set_x_F(w,obs):
            return np.tanh(np.dot(w, obs))
    
    def save_weight():
        pd.DataFrame(w).to_csv(os.path.join(settings.SAVE_PARAMS,"w.csv"), header=False, index=False)
        pd.DataFrame(epoch_S).to_csv("epoch_S.csv", header=False, index=False)
        
    def load_weight():
        tmp = pd.read_csv(os.path.join(settings.SAVE_PARAMS,"w.csv"), header=None)
        return tmp.T.values[0]
    
    obs = env.reset()
    mu = 1
    valid_hist = []
    tic = time.time()
    valid_S_hist = []
    #cum_sum_train = 0
    train_hist = []
    max_valid = -1000000000
    
    print("Epoch loop start.")
    for e_index in range(0,n_epoch):
        S_opt = S    
        x = []
        x_hist = []
        F = []
        F.append(0)
        R = np.zeros(T)
        epoch_S = np.empty(0)
        S = 0
        sum_reward_train = 0
        #Episode starts:
        for step in range(1, len(env.r)-M-1):        
            # State compostion and action in the environment
            x = obs.tolist()# re initialize the state 'x'
            x = [1] + x + [F[step-1]] # concatenate the last action and the one in the beginning
            x_hist.append(x)
            action = set_x_F(w,x) # calculate the scation for the actual step
            obs,reward,done,_ = env.step(action) #Provide de action to be followed
            sum_reward_train += reward
            F.append(action) # Append the action to the policy used  
        #print('Final step: ',step)    
        ##########episode ends####################
            
        x = [1] + obs.tolist() + [F[step-1]] # concatenate the last action and the one in the beginning
        x_hist.append(x)
        # From here we calculate the gradient to update the weights 'w'
        R = np.array(env.agent_returns[-T:])
        sumR = np.cumsum(R)
        sumR2 = np.cumsum(R**2)
        A      =  sumR[-1] / T
        B      =  sumR2[-1] / T
        S      =  A / np.sqrt(B - A**2)
        dSdA   =  S * (1 + S**2) / A
        dSdB   = -S**3 / 2 / A**2
        dAdR   =  1.0 / T
        dBdR   =  2.0 / T * R
        dRdF   = -mu * sigma * np.sign(np.diff(F))
        dRdFp  = mu * np.array(env.r[-T:]) + mu * sigma * (np.diff(F))
        dFdw   = np.zeros(M+2)
        dFpdw  = np.zeros(M+2)
        dSdw   = np.zeros(M+2)
        
        for i in range(0,len(env.r)-M-2): #subrtract 2 because here we starts at 0
            if i != (len(env.r)-M-1):
                dFpdw = dFdw.copy()
            dFdw  = (1 - F[i]**2) * (x_hist[i] + w[M+2-1] * dFpdw)
            dSdw += (dSdA * dAdR + dSdB * dBdR[i]) * (dRdF[i] * dFdw + dRdFp[i] * dFdw)
        w += rho * dSdw   
    
    
    ###############################################################
    # VALIDATION DURING EPISODES
    ###############################################################   
        obs = valid_env.reset()
        sigma = comission
        rho = 1
        x = []
        x_hist = []
        F = []
        F.append(0)
        R = np.zeros(T)
        obs = valid_env.reset()
        mu = 1 
        sum_reward_valid = 0
        for step in range(1, len(valid_env.r)-M-1):        
            # State compostion and action in the environment
            x = obs.tolist()# re initialize the state 'x'
            x = [1] + x + [F[step-1]] # concatenate the last action and the one in the beginning
            x_hist.append(x)
            action = set_x_F(w,x) # Calcalate the scation for the actual step
            obs,reward,done,_ = valid_env.step(action) #Provide de action to be followed
            sum_reward_valid += reward
            F.append(action) # Append the action to the policy used   
            
            R = np.array(env.agent_returns[-T:])
            sumR = np.cumsum(R)
            sumR2 = np.cumsum(R**2)
            A_valid       =  sumR[-1] / T
            B_valid       =  sumR2[-1] / T
            S_valid      =  A_valid  / np.sqrt(B_valid  - A_valid **2)
            
            valid_S_hist.append(S_valid)
    ###############################################################   
        #cum_sum_valid = np.cumsum(np.cumsum(valid_env.agent_returns))[-1]
        valid_hist.append(np.sum(valid_env.agent_returns))
        #cum_sum_train = np.cumsum(np.cumsum(env.agent_returns))[-1]
        train_hist.append(np.sum(env.agent_returns))
        if e_index>3:
            if valid_hist[-1]>max_valid:
                max_valid = valid_hist[-1]
                save_weight()
        epoch_S = np.append(epoch_S, S)
        
        
        if e_index % progress_period  == 0:
            toc = time.time()
            print("E: " + str(e_index + 1) + "/" + str(n_epoch) +". SR: " + str(round(S,5)) + ". Time: " + str(round(toc-tic,2)) + " s." +" Valid Metric: " +str(sum_reward_valid))
        
        # Reset the env and the policy
        obs = env.reset()
        F = []
        F.append(0)
    
    print("Epoch loop end. Optimized sharp's ratio is " + str(S_opt) + ".")
    
    np.save(os.path.join(settings.RESULTS_DIR, 'RRL_valid_hist_f'+str(fold)+'_'+asset_name+'.npy'), valid_hist)
    np.save(os.path.join(settings.RESULTS_DIR, 'RRL_train_hist_f'+str(fold)+'_'+asset_name+'.npy'), train_hist)
    #np.save('S_epoch.npy',valid_S_hist)
    
    ####################################################################################
    # TESTING
    ####################################################################################
def test_rrl_run(test_env,M,comission,mu,T,fold,n_epoch,asset_name):    
    
    def load_weight():
        tmp = pd.read_csv(os.path.join(settings.SAVE_PARAMS,"w.csv"), header=None)
        return tmp.T.values[0]
    
    def set_x_F(w,obs):
            return np.tanh(np.dot(w, obs))
    
    env = test_env
    sigma = comission
    x = []
    x_hist = []
    F = []
    F.append(0)
    R = np.zeros(T)
    w = np.ones(M+2)
    obs = env.reset()
    mu = 1
    tic = time.time()
    w = load_weight()
    sum_reward_test = 0
    test_reward_hist = []
    done = False
    step = 1   
    
    #for step in range(1, len(env.r)-M-1):    
    while not done:
        # State compostion and action in the environment
        x = obs.tolist()# re initialize the state 'x'
        x = [1] + x + [F[step-1]] # concatenate the last action and the one in the beginning
        x_hist.append(x)
        action = set_x_F(w,x) # Calcalate the scation for the actual step
        obs,reward,done,_ = env.step(action) #Provide de action to be followed
        sum_reward_test += reward
        test_reward_hist.append(sum_reward_test)
        F.append(action) # Append the action to the policy used   
        step += 1   
        
    np.save(os.path.join(settings.RESULTS_DIR, 'RRL_model_returns_f'+str(fold)+'_'+asset_name+'.npy'), env.agent_returns)
    np.save(os.path.join(settings.RESULTS_DIR, 'RRL_signals_f'+str(fold)+'_'+asset_name+'.npy'), F)
