import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


"""
Plotting the results using matplotlib:
    Inputs: folder
    Output: test_asset_name.pdf in local folder
"""

folder = 'your_folder/' 
asset_name = 'dash'
DQN = np.cumsum(np.load(folder+'DQN_agent_returns___'+asset_name+'.npy'))
RRL = np.cumsum(np.load(folder+'RRL_model_returns_f__'+asset_name+'.npy'))
A2C = np.cumsum(np.load(folder+'A2C_agent_returns__'+asset_name+'.npy'))
BSTS = np.cumsum(np.load(folder+'cbadapt_agent_returns__'+asset_name+'.npy'))
RESNET = np.cumsum(np.load('resnet_agent_returns__'+asset_name+'.npy'))
BH = np.cumsum(np.load(folder+'bh___'+asset_name+'.npy'))
fig = plt.figure(figsize=(17,6)) # Change the size of the figure here to be equal to the 
plt.plot(DQN, 'k', color='grey',linestyle='-',markevery = 70,marker = 'o',label = 'DQL')
plt.plot(RRL, 'k', color='grey',linestyle='-',markevery = 70,marker = '<',label = 'RRL')
plt.plot(A2C, 'k', color='grey',linestyle='-',markevery = 70,marker = 'x',label = 'A2C')
plt.plot(BSTS, 'k', color='grey',linestyle='-',markevery = 70,marker = 'd',label = 'BSTS')
plt.plot(RESNET, 'k', color='grey',linestyle='-.',label = 'RSLSTM-A')
plt.plot(BH, 'k', color='red',label = 'BH')
#plt.title('Asset: '+asset_name)
plt.legend()
plt.ylabel('Profit and Loss')
plt.xlabel('Time steps')
plt.savefig('test_'+asset_name+'.pdf',dpi=400)
plt.show()
plt.close()

