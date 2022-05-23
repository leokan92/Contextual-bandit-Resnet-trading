import gym
import numpy as np
from sklearn.preprocessing import StandardScaler

from empyrical import sortino_ratio, calmar_ratio, omega_ratio, sharpe_ratio


# Delete this if debugging
np.warnings.filterwarnings('ignore')


def calc_S(R,T = 365*24):
    sumR  = np.cumsum(R[::-1])[::-1]
    sumR2  = np.cumsum((R**2)[::-1])[::-1]
    A      =  sumR[0] / T
    B      =  sumR2[0] / T
    S      =  A / np.sqrt(B - A**2)
    return S

class BitcoinTradingEnv(gym.Env):
    '''A Trading environment for OpenAI gym'''
    metadata = {'render.modes': ['human', 'system', 'none']}
    viewer = None

    def __init__(self, df, initial_balance=10000, commission=0.000, reward_func='sortino',M = 50,mu = 1,length = 1000, scaling = False):
        super(BitcoinTradingEnv, self).__init__()
        
        self.mu = mu
        self.initial_balance = initial_balance
        self.commission = commission
        self.reward_func = reward_func
        self.M = M #look back window
        self.length = length

        self.df = df.fillna(method='bfill').reset_index()
        self.r = np.diff(self.df['Close'])
        self.scaling = scaling
        self.scaler = StandardScaler()
        

    def _next_observation(self):
        obs = self.r[self.current_step-self.M-1:self.current_step-1]
        if self.scaling: 
            self.scaler.fit(self.r[:self.current_step-1].reshape(-1,1))
            self.r_scaled = self.scaler.transform(self.r[:self.current_step-1].reshape(-1,1))
            obs = self.r_scaled[self.current_step-self.M-1:self.current_step-1].T[0]
        return obs

    def _current_price(self):
        return self.df['Close'].values[self.current_step]
    
    def _last_price(self):
        return self.df['Close'].values[self.current_step-1]

    def _take_action(self, action):
        #Working descrite with: -1: short, 0: neutral, 1: long
        action_type = int(round(action,0))        
        if action_type == 1: # Assumes Long postion            
            self.position == 'long'
        elif action_type == -1: #Assumes Short position
            self.position == 'short'
        elif action_type == 0: #Assumes Short position
            self.position == 'neutral'
            
        if (self.current_step == self.M+1): #to give the first position the neutral position
            self.agent_returns.append(0)
            self.position_history.append(0)
        else:
            self.agent_returns.append(self.initial_amount*(self.position_history[self.current_step-self.initial_step]*self.r[self.current_step-1] - self.commission*self.df['Close'].values[self.current_step-1]*abs(action - self.position_history[self.current_step-self.initial_step])))
        
        self.price_hist.append(self._current_price())
        self.trades.append({'step': self.current_step,'Position': self.position,'action': action})
        self.balance = self.balance + self.agent_returns[-1:][0]
        self.net_worths.append(self.balance)
        self.position_history.append(action)
        
    def _reward(self):
        returns = np.array(self.agent_returns[-self.length:])
        #returns = np.diff(self.net_worths[-length:])

        if np.count_nonzero(returns) < 1: 
            return 0

        if self.reward_func == 'sortino':
            reward = sortino_ratio(returns, annualization=365*24)#apply anualization correction based on the fact that is hourly
        elif self.reward_func == 'calmar':
            reward = calmar_ratio(returns, annualization=365*24)
        elif self.reward_func == 'omega':
            reward = omega_ratio(returns, annualization=365*24)
        elif self.reward_func == 'sharpe_ratio':
            reward = sharpe_ratio(returns, annualization=365*24)
        elif self.reward_func == 'differential_sharpe_ratio':
            reward = calc_S(returns, T=self.length)
        else:
            reward = returns[-1]

        return reward if np.isfinite(reward) else 0

    def _done(self):
        return self.current_step == (len(self.df) - self.M -1)

    def reset(self):
        self.position = 'neutral'
        self.initial_step = self.M+1
        self.current_step = self.initial_step
        self.initial_amount = self.mu
        self.balance = self.initial_balance
        self.net_worths = []
        self.net_worths.append(self.balance)
        
        self.position_history = []
        self.trades = []
        self.agent_returns = []
        self.price_hist = []

        return self._next_observation()

    def step(self, action):
        self._take_action(action)

        self.current_step += 1

        obs = self._next_observation()
        reward = self._reward()
        done = self._done()
        return obs, reward, done, {}

    def render(self, mode='system'):
        if mode == 'system':
            print('Price: ' + str(self._current_price()))
            print('Net worth: ' + str(self.net_worths[-1]))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
