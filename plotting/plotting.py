# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:48:40 2020

@author: leona
Plotting results

"""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# def moving_average(data_set, periods=3):
#     weights = np.ones(periods) / periods
#     return np.convolve(data_set, weights, mode='valid')
# last_fold = 7
# #assets = ['spy','aap','btc','ltc','eth']
# assets = ['btc','ltc','eth','spy','aap','amz']
# #'spy',
# ###########################################################
# # Plotting
# ###########################################################

# for asset_name in assets:
#     df_DQN = pd.DataFrame()
#     df_A2C = pd.DataFrame()
#     df_RRL = pd.DataFrame()
#     df_BSTS = pd.DataFrame()
#     df_res1 = pd.DataFrame()
#     df_res2 = pd.DataFrame()
#     df_res3 = pd.DataFrame()
#     df_res4 = pd.DataFrame()
#     df = pd.DataFrame()
    
#     bh = pd.DataFrame()
#     for fold in range(1,last_fold):
#         column_name = 'fold_'+str(fold)
#         df_DQN[column_name] = np.cumsum(np.load('DQN_agent_returns_'+str(fold)+'_'+asset_name+'.npy'))
#         df_RRL[column_name] = np.cumsum(np.load('RRL_model_returns_f'+str(fold)+'_'+asset_name+'.npy'))
#         df_A2C[column_name] = np.cumsum(np.load('A2C_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
#         df_BSTS[column_name] = np.cumsum(np.load('cbadapt_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
#         df_res1[column_name] = np.cumsum(np.load('resnet_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
#         df_res2[column_name] = np.cumsum(np.load('resnet_agent_returns'+str(fold)+'_'+asset_name+' (2)'+'.npy'))
#         df_res3[column_name] = np.cumsum(np.load('resnet_agent_returns'+str(fold)+'_'+asset_name+' (3)'+'.npy'))
#         df_res4[column_name] = np.cumsum(np.load('resnet_agent_returns'+str(fold)+'_'+asset_name+' (4)'+'.npy'))
#         bh[column_name] = np.cumsum(np.load('bh_'+str(fold)+'_'+asset_name+'.npy'))
#     x_DQN = df_DQN.mean(axis=1).index
#     y_DQN = df_DQN.mean(axis=1)
#     fig = plt.figure(figsize=(17,6))
#     plt.plot(x_DQN, y_DQN, 'k', color='grey',linestyle='-',markevery = 20,marker = 'o',label = 'DQL')
#     x_RRL = df_RRL.mean(axis=1).index
#     y_RRL = df_RRL.mean(axis=1)
#     plt.plot(x_RRL, y_RRL, 'k',color='grey',linestyle='-',markevery = 20,marker = '<',label = 'RRL')
#     x_A2C = df_A2C.mean(axis=1).index
#     y_A2C = df_A2C.mean(axis=1)
#     plt.plot(x_A2C, y_A2C, 'k', color='grey',linestyle='-',markevery = 20,marker = 'x',label = 'A2C')
#     x_BSTS = df_BSTS.mean(axis=1).index
#     y_BSTS = df_BSTS.mean(axis=1)
#     plt.plot(x_BSTS, y_BSTS, 'k', color='grey',linestyle='-',markevery = 20,marker = 'd',label = 'BSTS')   
#     x_res1 = df_res1.mean(axis=1).index
#     y_res1 = df_res1.mean(axis=1)
#     plt.plot(x_res1, y_res1, 'k', color='grey',linestyle='-.',label = 'res1')   
#     x_res2 = df_res2.mean(axis=1).index
#     y_res2 = df_res2.mean(axis=1)
#     plt.plot(x_res2, y_res2, 'k', color='grey',linestyle=':',label = 'res2')   
#     x_res3 = df_res3.mean(axis=1).index
#     y_res3 = df_res3.mean(axis=1)
#     plt.plot(x_res3, y_res3, 'k', color='grey',linestyle='--',label = 'res3')   
#     x_res4 = df_res4.mean(axis=1).index
#     y_res4 = df_res4.mean(axis=1)
#     plt.plot(x_res4, y_res4, 'k', color='grey',linestyle='-',label = 'res4')   
#     bhx = bh.mean(axis=1).index
#     bhy = bh.mean(axis=1)
fold = '_'
asset_name = 'dash'
DQN = np.cumsum(np.load('DQN_agent_returns_'+str(fold)+'_'+asset_name+'.npy'))
RRL = np.cumsum(np.load('RRL_model_returns_f'+str(fold)+'_'+asset_name+'.npy'))
A2C = np.cumsum(np.load('A2C_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
BSTS = np.cumsum(np.load('cbadapt_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
RESNET = np.cumsum(np.load('resnet_agent_returns'+str(fold)+'_'+asset_name+'.npy'))
BH = np.cumsum(np.load('bh_'+str(fold)+'_'+asset_name+'.npy'))
fig = plt.figure(figsize=(17,6))
plt.plot(DQN, 'k', color='grey',linestyle='-',markevery = 70,marker = 'o',label = 'DQL')
plt.plot(RRL, 'k', color='grey',linestyle='-',markevery = 70,marker = '<',label = 'RRL')
plt.plot(A2C, 'k', color='grey',linestyle='-',markevery = 70,marker = 'x',label = 'A2C')
plt.plot(BSTS, 'k', color='grey',linestyle='-',markevery = 70,marker = 'd',label = 'BSTS')
plt.plot(RESNET, 'k', color='grey',linestyle='-.',label = 'RESNET')
plt.plot(BH, 'k', color='red',label = 'BH')
#plt.title('Asset: '+asset_name)
plt.legend()
plt.ylabel('Profit and Loss')
plt.xlabel('Time steps')
plt.savefig('test_'+asset_name+'.pdf',dpi=400)
plt.show()
plt.close()


#
#for asset_name in assets:
#    df_DQN = pd.DataFrame()
#    df_A2C = pd.DataFrame()
#    df_RRL = pd.DataFrame()
#    df_BSTS = pd.DataFrame()
#    df = pd.DataFrame()
#    
#    bh = pd.DataFrame()
#    for fold in range(1,last_fold):
#        column_name = 'fold_'+str(fold)
#        df_DQN[column_name] = np.cumsum(np.load('DQN_valid_hist'+str(fold)+'_'+asset_name+'.npy'))
#        df_RRL[column_name] = np.cumsum(np.load('RRL_valid_hist_f'+str(fold)+'_'+asset_name+'.npy'))
#        df_A2C[column_name] = np.cumsum(np.load('A2C_valid_hist'+str(fold)+'_'+asset_name+'.npy'))
#        df_BSTS[column_name] = np.cumsum(np.load('cbadapt_valid_hist'+str(fold)+'_'+asset_name+'.npy'))
#        bh[column_name] = np.cumsum(np.load('bh_'+str(fold)+'_'+asset_name+'.npy'))
#    x_DQN = df_DQN.mean(axis=1).index
#    y_DQN = df_DQN.mean(axis=1)
#    plt.plot(x_DQN, y_DQN, 'k', color='blue',label = 'DQL')
#    x_RRL = df_RRL.mean(axis=1).index
#    y_RRL = df_RRL.mean(axis=1)
#    plt.plot(x_RRL, y_RRL, 'k', color='green',label = 'RRL')
#    x_A2C = df_A2C.mean(axis=1).index
#    y_A2C = df_A2C.mean(axis=1)
#    plt.plot(x_A2C, y_A2C, 'k', color='black',label = 'A2C')
#    x_BSTS = df_BSTS.mean(axis=1).index
#    y_BSTS = df_BSTS.mean(axis=1)
#    plt.plot(x_BSTS, y_BSTS, 'k', color='grey',label = 'BSTS')    
#    bhx = bh.mean(axis=1).index
#    bhy = bh.mean(axis=1)
#    error = bh.std(axis=1)
#    plt.plot(bhx, bhy, 'k', color='red')
#    plt.title('Valid Evolution for Asset '+asset_name)
#    plt.legend()
#    plt.ylabel('Total Profit')
#    plt.xlabel('Epoch')
#    plt.savefig('valid_'+asset_name+'.pdf')
#    plt.show()
#    plt.close()
#
#for asset_name in assets:
#    df_DQN = pd.DataFrame()
#    df_A2C = pd.DataFrame()
#    df_RRL = pd.DataFrame()
#    df_BSTS = pd.DataFrame()
#    df = pd.DataFrame()
#    
#    bh = pd.DataFrame()
#    for fold in range(1,last_fold):
#        column_name = 'fold_'+str(fold)
#        df_DQN[column_name] = np.cumsum(np.load('DQN_r_train_hist'+str(fold)+'_'+asset_name+'.npy'))
#        df_RRL[column_name] = np.cumsum(np.load('RRL_train_hist_f'+str(fold)+'_'+asset_name+'.npy'))
#        df_A2C[column_name] = np.cumsum(np.load('A2C_r_train_hist'+str(fold)+'_'+asset_name+'.npy'))
#        df_BSTS[column_name] = np.cumsum(np.load('cbadapt_r_train_hist'+str(fold)+'_'+asset_name+'.npy'))
#        bh[column_name] = np.cumsum(np.load('bh_'+str(fold)+'_'+asset_name+'.npy'))
#    x_DQN = df_DQN.mean(axis=1).index
#    y_DQN = df_DQN.mean(axis=1)
#    plt.plot(x_DQN, y_DQN, 'k', color='blue',label = 'DQL')
#    x_RRL = df_RRL.mean(axis=1).index
#    y_RRL = df_RRL.mean(axis=1)
#    plt.plot(x_RRL, y_RRL, 'k', color='green',label = 'RRL')
#    x_A2C = df_A2C.mean(axis=1).index
#    y_A2C = df_A2C.mean(axis=1)
#    plt.plot(x_A2C, y_A2C, 'k', color='black',label = 'A2C')
#    x_BSTS = df_BSTS.mean(axis=1).index
#    y_BSTS = df_BSTS.mean(axis=1)
#    plt.plot(x_BSTS, y_BSTS, 'k', color='grey',label = 'BSTS')    
#    bhx = bh.mean(axis=1).index
#    bhy = bh.mean(axis=1)
#    error = bh.std(axis=1)
#    plt.plot(bhx, bhy, 'k', color='red')
#    plt.title('Train Evolution for Asset '+asset_name)
#    plt.legend()
#    plt.ylabel('Total Profit')
#    plt.xlabel('Epoch')
#    plt.savefig('train_'+asset_name+'.pdf')
#    plt.show()
#    plt.close()
#
