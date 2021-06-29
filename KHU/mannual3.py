# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:24:00 2021

@author: MSBak
"""

import sys; 
msdir = 'C:\\Users\\skklab\\Documents\\mscode'; sys.path.append(msdir)
sys.path.append('D:\\mscore\\code_lab\\')
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


#%% mFunction

def msROC(class0, class1):
    import numpy as np
    from sklearn import metrics
    
    pos_label = 1; roc_auc = -np.inf; fig = None

    class0 = np.array(class0); class1 = np.array(class1)
    class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
    
    anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
    predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)       
    fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
    maxix = np.argmax((1-fpr) * tpr)
    specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
    accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
    roc_auc = metrics.auc(fpr,tpr)
    
    return accuracy, roc_auc


def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

def ms_syn(target_signal=None, target_size=None):
    downratio = target_signal.shape[0] / target_size
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
    return allo

def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout

#%% hyper

MAXSE = 20

#%% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열
signalss_raw = msdata_load['signalss_raw']

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
PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

# signals_raw에서 직접 수정할경우
# signalss = msFunction.msarray([N])
# for SE in PSLgroup_khu + morphineGroup:
#     for se in range(len(signalss_raw[SE])):
#         allo = np.zeros(signalss_raw[SE][se].shape) * np.nan
#         for ROI in range(signalss_raw[SE][se].shape[1]):
#             matrix = signalss_raw[SE][se][:,ROI]
#             if len(bahavss[SE][se][0]) > 0:
#                 bratio = (1-np.mean(bahavss[SE][se][0] > 0.15)) * 0.3
#             else: bratio = 0.3
#             base = np.sort(matrix)[0:int(round(matrix.shape[0]*bratio))]
#             base_mean = np.mean(base)
#             matrix2 = matrix/base_mean
#             allo[:,ROI] = matrix2
#             # plt.plot(matrix2)
#         signalss[SE].append(allo)

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = bahavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = downsampling(behav_tmp, signalss[SE][se].shape[0])

#%% data import - PD

# loadpath = 'C:\\mass_save\\PDpain\\mspickle_PD.pickle'  
# with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
#     msdict = pickle.load(f)
   
# signalss_raw_PD = msdict['signalss_raw_PD']
# signalss_PD = msdict['signalss_PD']
# behav_raw_PD = msdict['behav_raw_PD']

# movement_syn_PD = msFunction.msarray([len(signalss_PD),MAXSE])
# for SE in range(len(signalss_PD)):
#     tmp = []
#     for se in range(len(signalss_PD[SE])):
#         behav_tmp = behav_raw_PD[SE][se]
#         if len(behav_tmp) > 0:
#             movement_syn_PD[SE][se] = downsampling(behav_tmp, signalss_PD[SE][se].shape[0])

#%% grouping

group_pain_training = []
group_nonpain_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    for se in range(MAXSE):
        painc, nonpainc, test_only = [], [], []
        
        # khu formalin
        painc.append(SE in list(range(230, 239)) and se in [1])
        painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
        painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])
        
        nonpainc.append(SE in list(range(230, 239)) and se in [0])
        nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
        nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
        nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
        
        # snu psl pain
        painc.append(SE in pslGroup and se in [1,2])
        
        # snu psl+
        
        # snu oxali
        
        
        # khu psl
        nonpainc.append(SE in PSLgroup_khu and se in [0])
        painc.append(SE in PSLgroup_khu and se in [1,2])
        
        nonpainc.append(SE in morphineGroup and se in [0,1])
        nonpainc.append(SE in morphineGroup and se in [10,11,12])
        painc.append(SE in morphineGroup and se in [2,3,4,5,6,7,8,9])
        
        # PD
        nonpainc.append(SE in PDnonpain and se in list(range(2,10)))
        nonpainc.append(SE in PDnonpain and se in list(range(0,2)))
        painc.append(SE in PDpain and se in list(range(2,10)))
        nonpainc.append(SE in PDnonpain and se in list(range(0,2)))
        
       
        # test only
#        test_only.append(SE in PSLgroup_khu and se in [1,2])
        
        if np.sum(np.array(painc)) > 0:
            group_pain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_pain_training.append([SE, se])
            
        if np.sum(np.array(nonpainc)) > 0:
            group_nonpain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_nonpain_training.append([SE, se])

#%% XYZgen
feature_n = 2
X = msFunction.msarray([feature_n])
Y, Z = [], []

# activity distribution
# mssave = []

stanse = 0
THR = 0.22

target_sig = list(signalss)
target_sig2 = list(movement_syn)
forlist = PSLgroup_khu + morphineGroup + PDnonpain + PDpain #  + KHU_saline

matrix = np.zeros((len(target_sig),MAXSE)) * np.nan
for SE in forlist:
    if len(target_sig2[SE][stanse]) > 0:
        vix = np.where(target_sig2[SE][stanse] > 0.1)[0]
        sig = target_sig[SE][stanse][vix,:]
        stand = np.mean(sig, axis=0) / np.mean(sig)
        
        sig = target_sig[SE][stanse]
        stand_nonvix = np.mean(sig, axis=0) / np.mean(sig)
        
        for se in range(len(target_sig[SE])):
            if len(target_sig2[SE][se]) > 0:
                vix = np.where(target_sig2[SE][se] > 0.1)[0]
            
            if len(vix) > 0:
                sig = target_sig[SE][se][vix,:]
                exp = np.mean(sig, axis=0) / np.mean(sig)
                result = np.mean(np.abs(exp - stand) > THR)
                matrix[SE,se] = result
                
                # nonvix
                sig = target_sig[SE][se]
                exp = np.mean(sig, axis=0) / np.mean(sig)
                result2 = np.mean(np.abs(exp - stand_nonvix) > THR)
            
                label = None
                if [SE, se] in group_nonpain_training: label = [1, 0]
                if [SE, se] in group_pain_training: label = [0, 1]
                
                if not(label is None) and se != 0:
                    # xtmp = np.mean(signalss[SE][se], axis=1)
                    # xtmp = np.reshape(xtmp, (xtmp.shape[0], 1))
                    
                    # X[0].append(xtmp)
                    
                    f1 = result
                    f2 = np.mean(movement_syn[SE][se] > 0.15)
                    f3 = np.mean(signalss[SE][se])
                    X[1].append([f1, f2, f3])
                    
                    Y.append(label)
                    Z.append([SE, se])

fn = len(X[1][0])
for f in range(2):
    X[f] = np.array(X[f])
Y = np.array(Y)
Z = np.array(Z)

print(len(X[1]), len(Y))
#%% keras setup
from keras.callbacks import EarlyStopping
Callback = EarlyStopping
class EarlyStopping_ms(Callback):
    def __init__(self, monitor='accuracy', value=0.7, verbose=1, baseline=0.):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # print('current', current, self.value)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            # print('current', current, 'over thr')
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
callbacks = [EarlyStopping_ms(monitor='accuracy', value=0.91, verbose=1)]   

lr = 5e-4 # learning rate
n_hidden = int(2**6) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(2**6) # fully conneted laye node 갯수 # 8 # 원래 6 
    
l2_rate = 0.005
dropout_rate1 = 0.2 # dropout rate
dropout_rate2 = 0.1 # 
    

from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import BatchNormalization

from numpy.random import seed as nseed #
import tensorflow as tf
from keras.layers import Conv1D
from keras.layers import Flatten


def keras_setup(lr=0.01, batchnmr=False, seed=1, add_fn=None):
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras

    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    # input1 = keras.layers.Input(shape=(FS, 1))
    # input1_1 = Bidirectional(LSTM(n_hidden, return_sequences=False))(input1)
    # input1_1 = Dense(int(n_hidden), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input1_1) # fully conneted layers, relu

    
    input2 = keras.layers.Input(shape=(add_fn))
    input2_1 = Dense(int(n_hidden), kernel_initializer = init, kernel_regularizer=regularizers.l2(0), activation='relu')(input2) # fully conneted layers, relu

    # input_cocat = keras.layers.Concatenate(axis=1)([input1_1, input2_1])
    
    input10 = Dense(int(layer_1), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input2_1) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    
    input10 = Dense(int(layer_1), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input2_1) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    
    input10 = Dense(int(layer_1), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='sigmoid')(input10) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout

    merge_4 = Dense(2, kernel_initializer = init, activation='softmax')(input10) # fully conneted layers, relu

    model = keras.models.Model(inputs=input2, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer

    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup(lr=lr, seed=0, add_fn=fn)
print(model.summary())

#%% CV 10 fold


#%%

pix = np.where(Y[:, 1] == 1)[0]
nix = np.where(Y[:, 1] == 0)[0]

plt.figure()
plt.scatter(X[0][pix], X[1][pix])
plt.scatter(X[0][nix], X[1][nix])
# SE 기준

tlist = list(set(Z[:,0]))
shuffle_list = list(tlist); random.shuffle(shuffle_list)

cvn = int(len(tlist)/10)
for cv in range(10):
    cvlist = shuffle_list[cvn*cv:cvn*(cv+1)]
    if cv == 9: cvlist = shuffle_list[cvn*cv:]
    
    tlist2 = list(range(len(Z)))
    telist = []
    for t in cvlist:
        telist += list(np.where(Z[:,0]==t)[0])
    
    trlist = list(set(tlist2)-set(telist))
    
    X_tr = msFunction.msarray([fn])
    X_te = msFunction.msarray([fn])
    for f in [1]: #range(fn):
        X_tr[f] = X[f][trlist]; X_te[f] = X[f][telist]  
    Y_tr = Y[trlist]; Y_te = Y[telist]
    
    print('tr distribution', np.mean(Y_tr, axis=0))
    print('te distribution', np.mean(Y_te, axis=0))
    
    hist = model.fit(X_tr[1], Y_tr, batch_size=2**11, epochs=6000, verbose=1, validation_data= (X_te[1], Y_te), callbacks=callbacks)
    
        
    
    























