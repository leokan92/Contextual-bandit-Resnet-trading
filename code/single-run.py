"""
The single-run file provides a better way to execute all the experiments not including the RSLSTM-A.
    To reproduce the results presented in the article (https://github.com/leokan92/Contextual-bandit-Resnet-trading), or try a different set of experiment, just change the parameters in this file.

"""

import pandas as pd
import os
import numpy as np
import sys
# path = r'C:\Users\leona\Google Drive\USP\Doutorado\Artigo RRL-DeepLearning\Git\Contextual-bandit-Resnet-trading\run'
# sys.path.insert(0,path) # adding the code path
from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
from RRL import rrl_run,test_rrl_run
from A2C import a2c_run,test_a2c_run
from DQL import dql_run,test_dql_run
from CBmodel import cb_model,test_cb_model
import settings

# Experiment parameters

M  = 51 # Past window
mu = 1 # mu: number of assetes traded
decay = 200 # epslon decay for exploration purposes
n_epoch = 150 # number of RL episodes (here we call epochs)
asset_name = 'nxt' #This field should be changed to test other assets
gamma = 0.99 # discount factor used in RL algorithms
scaling = True # to scale the input date (space state)
training_mode = True

reward_strategy = 'return' # check the environment for more reward functions
input_data_file = os.path.join(settings.DATA_DIR, 'Poloniex_NXTBTC_1h.csv')
comission = 0.001
path = os.getcwd()
df = pd.read_csv(input_data_file,sep = ',')


# Environment definitions:
    # We have different environments for training, validation and test in order to have a out-of-sample in the test and a out-of-sample in the validation used to control overfitting

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


# Calling the RL function in the same root of this folder
np.save(os.path.join(settings.RESULTS_DIR,'bh_'+'_'+'_'+asset_name),test_env.r[M+2:-(M+1)])




if training_mode:
    a2c_run(train_env,valid_env,M,comission,'_',gamma,n_epoch,asset_name)
    dql_run(train_env,valid_env,M,comission,decay,'_',gamma,n_epoch,asset_name)
    rrl_run(train_env,valid_env,M,comission,mu,T,'_',n_epoch,asset_name)
    cb_model(train_env,valid_env,n_epoch,'_',asset_name)
    
test_a2c_run(test_env,M,comission,'_',gamma,n_epoch,asset_name)
test_dql_run(test_env,M,comission,decay,'_',gamma,n_epoch,asset_name)
test_rrl_run(test_env,M,comission,mu,T,'_',n_epoch,asset_name)
test_cb_model(test_env,'_',asset_name)


    
    