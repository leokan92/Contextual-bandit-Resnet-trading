# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:29:58 2020

@author: leona
"""




####################################################################
# Import of the environment lib
####################################################################
import pandas as pd
import os
import numpy as np
import sys
from env.BitcoinTradingEnv import BitcoinTradingEnv
from util.indicators import add_indicators
from sklearn.preprocessing import MinMaxScaler
import sklearn

def MinMax(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

####################################################################
# Initial parameters for the environment configuration
####################################################################
M  = 51
mu = 1
gamma = 0.99
comission = 0.00
np.random.seed(1)
action_space = [-1,1]
n_choices = len(action_space)
scaling = True

#reward_strategy = 'differential_sharpe_ratio'
reward_strategy = 'returns'

####################################################################
# Financial data Loading
####################################################################
path  = os.getcwd()
#input_data_file = path+'/data/BTCUSD.Candlestick_1_Hour_BID_06.05.2017-13.03.2020.csv'
input_data_file = path+'/data/Poloniex_DASHUSD_1h.csv'
#input_data_file = path+'/data/TS_SIN.csv'
#input_data_file = path+'/data/ETHUSD.Candlestick_1_Hour_BID_09.12.2017-08.05.2020.csv'
#input_data_file = path+'/data/LTCUSD_Candlestick_1_Hour_BID_01.09.2018-20.05.2020.csv'
asset_name = 'dash'
df = pd.read_csv(input_data_file)
df = df[:]


####################################################################
# Environment creation based on the train, validation and test framework
####################################################################

valid_len = int(len(df) * 0.2/2) #change for the transfer learning
test_len = valid_len
train_len = int(len(df)) - valid_len*2
    
train_df = df[:train_len]
valid_df = df[train_len+M:train_len+M+valid_len]
test_df = df[train_len+M+valid_len:]
length= len(train_df)
train_env = BitcoinTradingEnv(train_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling = scaling)
length= len(valid_df)
valid_env = BitcoinTradingEnv(valid_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling = scaling)
length= len(test_df)
test_env = BitcoinTradingEnv(test_df, commission=comission, reward_func=reward_strategy, M = M , mu = mu,length = length,scaling = scaling)

####################################################################
# Create a supervised dataset to train the TS classifier
####################################################################

#Create Train set

df_train = pd.DataFrame(columns = ['state','action','reward'])

state = train_env.reset()
done = False

while not done:
    next_state, reward, done, info = train_env.step(1)
    df_train = df_train.append({'state': state,'action': np.array([0,1]),'reward': reward}, ignore_index=True)
    #df_train = df_train.append({'state': state,'action': 1,'reward': reward}, ignore_index=True)
    state = next_state
n = len(df_train['action'][df_train['reward'] < 0])


df_train['action'][df_train['reward'] < 0] = df_train['action'][df_train['reward'] < 0].apply(lambda x: np.array([1, 0]))
# df_train['action'][df_train['reward'] < 0] = df_train['action'][df_train['reward'] < 0]*(df_train['reward'][df_train['reward'] < 0]/df_train['reward'].min())
# df_train['action'][df_train['reward'] < 0] = df_train['action'][df_train['reward'] < 0].apply(lambda x: [x[0],1-x[0],0])

# df_train['action'][df_train['reward'] > 0] = df_train['action'][df_train['reward'] > 0]*(df_train['reward'][df_train['reward'] > 0]/df_train['reward'].max())
# df_train['action'][df_train['reward'] > 0] = df_train['action'][df_train['reward'] > 0].apply(lambda x: [0,1-x[2],x[2]])

x_train = np.expand_dims(np.stack(df_train['state'].values,axis = 1).T,-1)
#x_train = df_train['state']
#x_train = MinMax(x_train)
p = np.random.permutation(len(x_train))
#x_train = x_train[p]
#np.random.shuffle(x_train)

y_train = np.stack(df_train['action'].values,axis = 1).T#.reshape(-1,n_choices)
#y_train = df_train['action'].values
#y_train = y_train[p]
#Create Valid set

df_valid = pd.DataFrame(columns = ['state','action','reward'])

state = valid_env.reset()
done = False

while not done:
    next_state, reward, done, info = valid_env.step(1)
    df_valid = df_valid.append({'state': state,'action': np.array([0,1]),'reward': reward}, ignore_index=True)
    #df_valid = df_valid.append({'state': state,'action': 1,'reward': reward}, ignore_index=True)
    state = next_state
n = len(df_valid['action'][df_valid['reward'] < 0])

 
df_valid['action'][df_valid['reward'] < 0] = df_valid['action'][df_valid['reward'] < 0].apply(lambda x: np.array([1, 0]))
# df_valid['action'][df_valid['reward'] < 0] = df_valid['action'][df_valid['reward'] < 0]*(df_valid['reward'][df_valid['reward'] < 0]/df_valid['reward'].min())
# df_valid['action'][df_valid['reward'] < 0] = df_valid['action'][df_valid['reward'] < 0].apply(lambda x: [x[0],1-x[0],0])

# df_valid['action'][df_valid['reward'] > 0] = df_valid['action'][df_valid['reward'] > 0]*(df_valid['reward'][df_valid['reward'] > 0]/df_valid['reward'].max())
# df_valid['action'][df_valid['reward'] > 0] = df_valid['action'][df_valid['reward'] > 0].apply(lambda x: [0,1-x[2],x[2]])

#df_valid['action'][df_valid['reward'] < 0] =-1

x_val =  np.expand_dims(np.stack(df_valid['state'].values,axis = 1).T,-1)
# = df_valid['state']

#x_val = MinMax(x_val)

y_val = np.stack(df_valid['action'].values,axis = 1).T
#y_val = df_valid['action'].values
#y_true = y_val.copy()
y_true = np.argmax(y_val, axis=1)
#y_true = np.argmax(np.stack(np.array(y_val)), axis=1)
#Create Test


#x_test = MinMax(x_test)

# in1 = keras.layers.Input(shape = (M,))
# dense1 = keras.layers.Dense(512,activation='relu')(in1)
# dense1 = keras.layers.Dense(2,activation='sigmoid')(dense1)

# model = keras.models.Model(inputs=in1, outputs=dense1)

# model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01),
#                       metrics=['accuracy'])

# model.fit(x_train[...,0], y_train, batch_size=50, epochs=100,
#                               verbose=1, validation_data=(x_val[...,0], y_val))

# plt.plot(x_train[10,:,0])
# plt.show()
####################################################################
# Resnet initialization
####################################################################

# resnet model 
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers 
#import keras
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

import matplotlib
from utils.utils import save_test_duration

matplotlib.use('agg')
import matplotlib.pyplot as plt

from utils.utils import save_logs
from utils.utils import calculate_metrics


class Classifier_RESNET:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, load_weights=False):
        self.output_directory = output_directory
        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            if load_weights == True:
                self.model.load_weights(self.output_directory
                                        .replace('resnet_augment', 'resnet')
                                        .replace('TSC_itr_augment_x_10', 'TSC_itr_10')
                                        + '/model_init.hdf5')
            else:
                self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = tf.keras.layers.Dropout(0.5)(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = tf.keras.layers.Dropout(0.5)(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        conv_z = tf.keras.layers.Dropout(0.5)(conv_z)
    
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = tf.keras.layers.Dropout(0.5)(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = tf.keras.layers.Dropout(0.5)(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        conv_z = tf.keras.layers.Dropout(0.5)(conv_z)
        
        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)
        conv_x = tf.keras.layers.Dropout(0.5)(conv_x)
        
        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)
        conv_y = tf.keras.layers.Dropout(0.5)(conv_y)
        
        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)
        conv_z = tf.keras.layers.Dropout(0.5)(conv_z)
        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL

        #gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        #gap_layer = keras.layers.GlobalMaxPooling1D()(output_block_3)
        #output_block_3 = keras.layers.Conv1D(1, kernel_size=3, padding='same')(output_block_3)
        #flat = keras.layers.Flatten()(output_block_3)
        
        #dense1 = keras.layers.Dense(100, activation='relu')(flat)
        lstm_layer = keras.layers.LSTM(100)(output_block_3)
        
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(lstm_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 64
        nb_epochs = 150
        
        # class_weight = sklearn.utils.class_weight.compute_class_weight( 'balanced',classes = np.unique(np.argmax(y_train,-1)),y=np.argmax(y_train,-1))
        
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        self.hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        #df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration)


        
        hist_df = pd.DataFrame(self.hist.history)
        hist_df.to_csv(output_directory + 'history.csv', index=False)

        keras.backend.clear_session()

        #return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

output_directory = os.getcwd()+'/results/'

input_shape = x_train.shape[1:]
    
nb_classes = len(action_space)

verbose = True

model = Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)

model.fit(x_train, y_train, x_val, y_val, y_true)

plt.plot(model.hist.history['loss'],label='loss')
plt.plot(model.hist.history['val_loss'],label='val_loss')
plt.title('Resnet Train and Validation Loss')
plt.legend()
plt.show()
plt.savefig('resnet_loss.png')
plt.close()

plt.plot(model.hist.history['acc'],label='acc')
plt.plot(model.hist.history['val_acc'],label='val_acc')
plt.title('Resnet Train and Validation Accuracy')
plt.legend()
plt.show()
plt.savefig('resnet_acc.png')
plt.close()


#####################################################
#Testing
#####################################################
    
state = test_env.reset()
done = False
model_pred = keras.models.load_model(os.getcwd()+'/results/best_model.hdf5')
action_prob_hist = []
while not done:
    action_prob = model_pred.predict(np.expand_dims(np.expand_dims(state.T,-1).T,-1))
    action_prob_hist.append(action_prob[0])
    # if action_prob[0][2]>0.6:
    #     action = 2
    # elif action_prob[0][0]>0.55:
    #     action = 0
    # else:
    #     action = 1
    action = np.argmax(action_prob[0])
    action = action_space[action]
    next_state, reward, done, info = test_env.step(action)
    state = next_state

action_prob_hist = np.stack(action_prob_hist)

plt.plot(action_prob_hist[:,0],label='Prob')
plt.title('Probablity')
plt.legend()
plt.show()
plt.savefig('resnet_test_prob.png')
plt.close()

plt.plot(np.cumsum(test_env.agent_returns),label='Resnet')
plt.plot(np.cumsum(test_env.r[M:]),label='BH')
plt.title('Resnet adn BH Returns')
plt.legend()
plt.show()
plt.savefig('resnet_returns.png')
plt.close()

np.save('resnet_agent_returns'+'_'+asset_name+'.npy',test_env.agent_returns)
np.save('resnet_signals'+'_'+asset_name+'.npy',test_env.position_history)
