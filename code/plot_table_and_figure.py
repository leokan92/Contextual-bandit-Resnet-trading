# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 20:39:32 2022

@author: leona
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import settings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))  
LOG_DIR = os.path.join(BASE_DIR,'code','params')
RESULTS_DIR = os.path.join(BASE_DIR,'code','results')
path_data = os.path.join(BASE_DIR,'data')
#DQN_agent_returns_xrp.npy

plt.rcParams["font.family"] = "Times New Roman"

##########################################
#Plotting results for one asset:
##########################################

asset_name = 'lsk'

fold = '_'
window = '90'
PATH = RESULTS_DIR
DQN = np.cumsum(np.load(os.path.join(PATH,'DQN_agent_returns_'+asset_name+'.npy')))
RRL = np.cumsum(np.load(os.path.join(PATH,'RRL_model_returns_f'+str(fold)+'_'+asset_name+'.npy')))
A2C = np.cumsum(np.load(os.path.join(PATH,'A2C_agent_returns_'+asset_name+'.npy')))
PPO = np.cumsum(np.load(os.path.join(PATH,'PPO_agent_returns_'+asset_name+'.npy')))
BSTS = np.cumsum(np.load(os.path.join(PATH,'cbadapt_agent_returns'+str(fold)+'_'+asset_name+'.npy')))
RESNET = np.cumsum(np.load(os.path.join(PATH,'resnet_agent_returns'+'_'+window+'_'+asset_name+'.npy')))
BH = np.cumsum(np.load(os.path.join(PATH,'bh_'+str(fold)+'_'+asset_name+'.npy')))
fig = plt.figure(figsize=(17,5))
plt.plot(DQN, 'k', color='grey',linestyle='-',markevery = 140,marker = 'o',label = 'DQL')
plt.plot(RRL, 'k', color='grey',linestyle='-',markevery = 140,marker = '<',label = 'RRL')
plt.plot(A2C, 'k', color='grey',linestyle='-',markevery = 140,marker = 'x',label = 'A2C')
plt.plot(PPO, 'k', color='grey',linestyle='-',markevery = 140,marker = '*',label = 'PPO')
plt.plot(BSTS, 'k', color='grey',linestyle='-',markevery = 140,marker = 'd',label = 'BTS')
plt.plot(RESNET, 'k', color='green',linestyle='-.',label = 'RSLSTM-A')
plt.plot(BH, 'k', color='red',label = 'B&H')
plt.title('Asset: '+asset_name)
plt.legend()
plt.ylabel('Profit and Loss')
plt.xlabel('Time steps')
#plt.imshow(img, alpha=0.25)
#plt.savefig(path +'results - tables;crypto;extended datas/'+'test_all_assets.pdf', bbox_inches='tight')
plt.savefig(os.path.join(RESULTS_DIR,'plot.png'), bbox_inches='tight')

##########################################
#Generating table results for one asset:
##########################################

import settings
path_data = settings.DATA_DIR
path_input = settings.RESULTS_DIR
import matplotlib.pyplot as plt
import numpy as np
import math
from env.TradingEnv import TradingEnv
import os

def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')

def calculate_AR(series,init_value):  
    if math.isnan(((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))**365-1):
      return -((((-np.sum(series)+init_value)/init_value+1)**(1/len(series))))**365-1
    else:
      return ((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))**365-1
    
def calculate_SR(series,init_value):
    risk_free = 0.0001
    if math.isnan((((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))-1-risk_free)/np.std(series)*np.sqrt(365)):
      return -(((((-np.sum(series)+init_value)/init_value+1)**(1/len(series))))-1-risk_free)/np.std(series)*np.sqrt(365)
    else:
      return (((((np.sum(series)+init_value)/init_value+1)**(1/len(series))))-1-risk_free)/np.std(series)*np.sqrt(365)

############################################################
# Load first datas
############################################################

assets = ['btc','dash','eth','ltc','nxt','xmr','etc','pot','xrp','xem','lsk']

######################################################################################################################
# Create table for cumlateive returns
######################################################################################################################
df = pd.DataFrame(columns = [' ','Asset','BH','RRL','DQL','A2C','PPO','BSTS','RSLSTM-A'])
#df['Asset']= assets
#df = df.set_index('Asset')

alpha = 0.05
k_fold = 15
min_df_size = 3000
comission = 0.00
gamma = 0.99
M  = 51
mu = 1
n_epoch = 600
decay = 200
reward_strategy = 'return'
scaling = True
window = '90'


input_datas = []

input_datas.append(os.path.join(path_data,'Poloniex_BTCUSD_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_DASHUSD_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_ETHUSD_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_LTCUSD_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_NXTBTC_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_XMRUSD_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_ETCBTC_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_POTBTC_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_XRPBTC_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_XEMBTC_1h.csv'))
input_datas.append(os.path.join(path_data,'Poloniex_LSKBTC_1h.csv'))

for asset_name, data_name in zip(assets, input_datas):
    data = pd.read_csv(data_name)
    valid_len = int(len(data) * 0.2/2)
    test_len = valid_len
    train_len = int(len(data)) - valid_len*2   
    train_df = data[:train_len]
    valid_df = data[train_len+M:train_len+M+valid_len]
    test_df = data[train_len+M+valid_len:]
    length= len(train_df)
    test_env = TradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
    if asset_name == 'nxt':
      firs_value_test = test_env.df['Close'][0]
    else:
      firs_value_test = test_env.df['Close'][0]
    a2c = np.sum(np.cumsum(np.load(os.path.join(path_input,'A2C_agent_returns_'+asset_name+'.npy'))))
    dql = np.sum(np.cumsum(np.load(os.path.join(path_input,'DQN_agent_returns_'+asset_name+'.npy'))))
    ppo = np.sum(np.cumsum(np.load(os.path.join(path_input,'PPO_agent_returns_'+asset_name+'.npy'))))
    rrl = np.sum(np.cumsum(np.load(os.path.join(path_input,'RRL_model_returns_f__'+asset_name+'.npy'))))
    bts = np.sum(np.cumsum(np.load(os.path.join(path_input,'cbadapt_agent_returns__'+asset_name+'.npy'))))
    resnet = np.sum(np.cumsum(np.load(os.path.join(path_input,'resnet_agent_returns_'+window+'_'+asset_name+'.npy'))))
    bh = np.sum(np.cumsum(np.load(os.path.join(path_input,'bh___'+asset_name+'.npy'))))
    if asset_name == 'eth':
      df_temp = pd.DataFrame({' ':'ACR','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    else:
      df_temp = pd.DataFrame({' ':' ','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    df = df.append(df_temp)

for asset_name, data_name in zip(assets, input_datas):
    data = pd.read_csv(data_name)
    valid_len = int(len(data) * 0.2/2)
    test_len = valid_len
    train_len = int(len(data)) - valid_len*2   
    train_df = data[:train_len]
    valid_df = data[train_len+M:train_len+M+valid_len]
    test_df = data[train_len+M+valid_len:]
    length= len(train_df)
    test_env = TradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
    if asset_name == 'nxt':
      firs_value_test = test_env.df['Close'][0]
      firs_value_test = test_env.df['Close'][0]
    a2c = calculate_SR(np.load(os.path.join(path_input,'A2C_agent_returns_'+asset_name+'.npy')),firs_value_test)
    dql = calculate_SR(np.load(os.path.join(path_input,'DQN_agent_returns_'+asset_name+'.npy')),firs_value_test)
    ppo = calculate_SR(np.load(os.path.join(path_input,'PPO_agent_returns_'+asset_name+'.npy')),firs_value_test)
    rrl = calculate_SR(np.load(os.path.join(path_input,'RRL_model_returns_f__'+asset_name+'.npy')),firs_value_test)
    bts = calculate_SR(np.load(os.path.join(path_input,'cbadapt_agent_returns__'+asset_name+'.npy')),firs_value_test)
    resnet = calculate_SR(np.load(os.path.join(path_input,'resnet_agent_returns_'+window+'_'+asset_name+'.npy')),firs_value_test)
    bh = calculate_SR(np.load(os.path.join(path_input,'bh___'+asset_name+'.npy')),firs_value_test)
    if asset_name == 'eth':
      df_temp = pd.DataFrame({' ':'SR','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    else:
      df_temp = pd.DataFrame({' ':' ','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    df = df.append(df_temp)

for asset_name, data_name in zip(assets, input_datas):
    data = pd.read_csv(data_name)
    valid_len = int(len(data) * 0.2/2)
    test_len = valid_len
    train_len = int(len(data)) - valid_len*2   
    train_df = data[:train_len]
    valid_df = data[train_len+M:train_len+M+valid_len]
    test_df = data[train_len+M+valid_len:]
    length= len(train_df)
    test_env = TradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling=scaling)
    if asset_name == 'nxt':
      firs_value_test = test_env.df['Close'][0]
    else:
      firs_value_test = test_env.df['Close'][0]
    a2c = calculate_AR(np.load(os.path.join(path_input,'A2C_agent_returns_'+asset_name+'.npy')),firs_value_test)
    dql = calculate_AR(np.load(os.path.join(path_input,'DQN_agent_returns_'+asset_name+'.npy')),firs_value_test)
    ppo = calculate_AR(np.load(os.path.join(path_input,'PPO_agent_returns_'+asset_name+'.npy')),firs_value_test)
    rrl = calculate_AR(np.load(os.path.join(path_input,'RRL_model_returns_f__'+asset_name+'.npy')),firs_value_test)
    bts = calculate_AR(np.load(os.path.join(path_input,'cbadapt_agent_returns__'+asset_name+'.npy')),firs_value_test)
    resnet = calculate_AR(np.load(os.path.join(path_input,'resnet_agent_returns_'+window+'_'+asset_name+'.npy')),firs_value_test)    
    bh = calculate_AR(np.load(os.path.join(path_input,'bh___'+asset_name+'.npy')),firs_value_test)
    if asset_name == 'eth':
      df_temp = pd.DataFrame({' ':'AR','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    else:
      df_temp = pd.DataFrame({' ':' ','Asset':asset_name,'BH':bh,'RRL':rrl,'DQL':dql,'A2C':a2c,'PPO':ppo,'BSTS':bts,'RSLSTM-A':resnet}, index=[0])
    df = df.append(df_temp)

df
df = df.reset_index().drop(columns= ['index'])
df[' '][0] = 'ACR'
df[' '][1] = 'SR'
df[' '][2] = 'AR'

df['BH'] = df['BH'].map('{:.2e}'.format)
df['RRL'] = df['RRL'].map('{:.2e}'.format)
df['DQL'] = df['DQL'].map('{:.2e}'.format)
df['A2C'] = df['A2C'].map('{:.2e}'.format)
df['PPO'] = df['PPO'].map('{:.2e}'.format)
df['BSTS'] = df['BSTS'].map('{:.2e}'.format)
df['RSLSTM-A'] = df['RSLSTM-A'].map('{:.2e}'.format)

caption = 'Consolidate results for the the accumulated returns, annualized return and sharpe ration for all the employed models and assets using transaction cost equals to 0.002'
print(df.to_latex(index=False,label = 'tab:final_resultstc0',caption=caption,escape = False))