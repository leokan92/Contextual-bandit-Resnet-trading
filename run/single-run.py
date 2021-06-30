# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:18:11 2020

@author: leona
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 01:22:25 2020

@author: leona
k-fold run
"""


import pandas as pd
import os
import numpy as np
import sys
# adding the path and importing the function
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\RRL non vectorized'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\support'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\A2C'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\DQL'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\IRL - RRL'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Algorithms\new_env_testing\Contextual Bandit\contextualbandit-nocython'
sys.path.insert(0,path) # adding the code path
path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning'
sys.path.insert(0,path) # adding the code path
#path =  os.getcwd()
#sys.path.insert(0,path)
from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
#from IRL import irl_run
from RRL import rrl_run,test_rrl_run
from A2C import a2c_run,test_a2c_run
from DQL import dql_run,test_dql_run
from CBmodel import cb_model


#M_rrl_original = 49
M  = 51
mu = 1
decay = 200
n_epoch = 500
asset_name = 'dash'
gamma = 0.99
scaling = True

reward_strategy = 'return'
#reward_strategy = 'sharpe_ratio'
input_data_file = path +r'\Data\Poloniex_NXTBTC_1h.csv'
#input_data_file = path+'/data/SPY.USUSD_Candlestick_1_Hour_BID_11.07.2017-13.03.2020.csv'
comission = 0.00
path = os.getcwd()
#policy = np.load(path+r'\optimal_policy.npy')

df = pd.read_csv(input_data_file,sep = ';')*100000
#df = df.drop(['Symbol'], axis=1)
#df = df.iloc[::-1]
#df = add_indicators(df.reset_index())
# Data set definitions inside de fold
valid_len = int(len(df) * 0.2/2)
test_len = valid_len
train_len = int(len(df)) - valid_len*2   
train_df = df[:train_len]
valid_df = df[train_len+M:train_len+M+valid_len]
test_df = df[train_len+M+valid_len:]
length= len(train_df)
train_env = BitcoinTradingEnv(train_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
length= len(valid_df)
valid_env = BitcoinTradingEnv(valid_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
length= len(test_df)
test_env = BitcoinTradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
T = len(train_df)-M-3
#calling the RL function
np.save('bh_'+'_'+'_'+asset_name,test_env.r[M+2:-(M+1)])

a2c_run(train_env,valid_env,M,comission,'_',gamma,n_epoch,asset_name)
test_a2c_run(test_env,M,comission,'_',gamma,n_epoch,asset_name)

dql_run(train_env,valid_env,M,comission,decay,'_',gamma,n_epoch,asset_name)
test_dql_run(test_env,M,comission,decay,'_',gamma,n_epoch,asset_name)

rrl_run(train_env,valid_env,M,comission,mu,T,'_',n_epoch,asset_name)
test_rrl_run(test_env,M,comission,mu,T,'_',n_epoch,asset_name)

cb_model(train_env,valid_env,test_env,n_epoch,'_',asset_name)


    
    