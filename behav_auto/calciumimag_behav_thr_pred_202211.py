# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:42:21 2022

@author: MSBak
"""

import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode\\')
sys.path.append('C:\\Users\\skklab\\Documents\mscode\\')
sys.path.append('K:\\mscode_m2\\')

import msFunction
import os  
try: import pickle5 as pickle
except: import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import random
import time
from tqdm import tqdm
from scipy import stats
import scipy
import time

MAXSE = 40

# plt.plot(range(10))

#%% data import

# gsync = 'D:\\2p_pain\\'
# gsync = 'C:\\mass_save\\PSLpain\\'
gsync = 'C:\\SynologyDrive\\2p_data\\'
if os.path.isdir('K:\\mscode_m2'): gsync = 'K:\\SynologyDrive\\2p_data\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
behavss = msdata_load['behavss']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = np.array(msdata_load['signalss']) # 투포톤 이미징데이터 -> 시계열
# signalss_df = np.array(msdata_load['signalss']) # 투포톤 이미징데이터 -> 시계열
signalss_raw = np.array(msdata_load['signalss_raw'])
signalss_df = msdata_load['signalss_df'] # merge시킴 (20221103)


signalss2 = msdata_load['signalss2']
movement_syn = msdata_load['movement_syn']

highGroup = msGroup['highGroup']    # 5% formalin
midleGroup = msGroup['midleGroup']  # 1% formalin
lowGroup = msGroup['lowGroup']      # 0.25% formalin
salineGroup = msGroup['salineGroup']    # saline control
restrictionGroup = msGroup['restrictionGroup']  # 5% formalin + restriciton
ketoGroup = msGroup['ketoGroup'] # 5% formalin + keto 100
lidocaineGroup = msGroup['lidocaineGroup'] # 5% formalin + lidocaine
capsaicinGroup = msGroup['capsaicinGroup'] # capsaicin
yohimbineGroup = msGroup['yohimbineGroup'] # 5% formalin + yohimbine
pslGroup = msGroup['pslGroup'] # partial sciatic nerve injury model
shamGroup = msGroup['shamGroup']
adenosineGroup = msGroup['adenosineGroup']
highGroup2 = msGroup['highGroup2']
CFAgroup = msGroup['CFAgroup']
chloroquineGroup = msGroup['chloroquineGroup']
itSalineGroup = msGroup['itSalineGroup']
itClonidineGroup = msGroup['itClonidineGroup']
ipsaline_pslGroup = msGroup['ipsaline_pslGroup']
ipclonidineGroup = msGroup['ipclonidineGroup']
gabapentinGroup = msGroup['gabapentinGroup']
beevenomGroup =  msGroup['beevenomGroup']
oxaliGroup =  msGroup['oxaliGroup']
glucoseGroup =  msGroup['glucoseGroup']
PSLscsaline =  msGroup['PSLscsaline']

highGroup3 =  msGroup['highGroup3']
highGroup3_late = msGroup['highGroup3_late']
KHU_saline = msGroup['KHU_saline']

PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']
KHUsham = msGroup['KHUsham']
KHU_CFA = msGroup['KHU_CFA']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']
PDmorphine = msGroup['PDmorphine']

PD_ldopa = msGroup['PD_ldopa']
KHU_PSL_magnolin = msGroup['KHU_PSL_magnolin']

oxali_BV = msGroup['oxali_BV']
oxali_sal = msGroup['oxali_sal']

#%%


# SE, se = 273, 1
# SE, se = 1, 1
# SE, se = 50, 1
# plt.plot(movement_syn[SE][se])
# signalss_df
# plt.plot()

#%%

min_len = 490
bins = 50

X0, X1 = [] ,[]
Y, Z = [], []

for SE in tqdm(range(N)):
    for se in range(len(behavss[SE])):
        a_session = np.array(signalss_df[SE][se])
        c2 = len(behavss[SE][se][0]) > 0
        if len(a_session.shape) == 2 and c2:
            x0_tmp = np.array(np.mean(a_session, axis=1))
            x1_tmp = np.array(behavss[SE][se][0])
            y_tmp = float(behavss[SE][se][1])
            
            for f in range(0, len(x0_tmp)-min_len, bins):
                X0.append(x0_tmp[f : f + min_len])
                X1.append(x1_tmp[f : f + min_len])
                Y.append(y_tmp)
                Z.append([SE, se, f])
                
X0, X1, Y, Z = np.array(X0), np.array(X1), np.array(Y), np.array(Z)
print(X0.shape, X1.shape, Y.shape, Z.shape)            

#%% keras setup

lr = 1e-3 # learning rate
n_hidden = int(2**3) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(2**3) # fully conneted laye node 갯수 # 8 # 원래 6 
    
l2_rate = 1e-5
dropout_rate1 = 0.1 # dropout rate
dropout_rate2 = 0.1 # 
    
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import initializers
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten

from numpy.random import seed as nseed #
import tensorflow as tf

# xin = X0.shape[1]
def keras_setup(lr=1e-3, min_len=min_len, layer_1=None, dropout_rate1=0):
    
    init = initializers.he_uniform(seed=0) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
    
    input1_1 = tf.keras.layers.Input(shape=(min_len, 1)) # 각 병렬 layer shape에 따라 input 받음
    input1_2 = Bidirectional(LSTM(4, return_sequences=True))(input1_1) # biRNN -> 시계열에서 단일 value로 나감
    input1_3 = Bidirectional(LSTM(1, return_sequences=True))(input1_2)
    
    input2_1 = tf.keras.layers.Input(shape=(min_len, 1))
    input2_2 = Bidirectional(LSTM(16, return_sequences=True))(input2_1)
    
    concatted = tf.keras.layers.Concatenate()([input1_3, input2_2])
    
    input_12merge = Bidirectional(LSTM(2**7))(concatted)
    

    input10 = Dense(layer_1, kernel_initializer = init, activation='relu')(input_12merge) # fully conneted layers, relu
    input10 = Dropout(dropout_rate1)(input10) # dropout
    if batchnmr: input10 = BatchNormalization()(input10)
    
    input10 = Dense(layer_1, kernel_initializer = init, activation='relu')(input10) # fully conneted layers, relu
    input10 = Dropout(dropout_rate1)(input10) # dropout
    if batchnmr: input10 = BatchNormalization()(input10)
    
    input10 = Dense(layer_1, kernel_initializer = init, activation='sigmoid')(input10) # fully conneted layers, relu
    input10 = Dropout(dropout_rate1)(input10) # dropout
    if batchnmr: input10 = BatchNormalization()(input10)
    
    out1 = Dense(1, kernel_initializer = init)(input10) # fully conneted layers, relu
    model = tf.keras.models.Model(inputs=[input1_1, input2_1], outputs = out1) # input output 선언
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup(lr=lr, layer_1=layer_1)
print(model.summary())

#%%

hist = model.fit([X0, X1], Y, batch_size=2**7, epochs=10, verbose=1)

vix = [0,10,30, 100, 500]
yhat = model.predict([X0[vix], X1[vix]])

print(yhat, Y[vix])







































