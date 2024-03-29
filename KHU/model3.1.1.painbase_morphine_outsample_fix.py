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

MAXSE = 20
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

#%% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['behavss2']   # 움직임 정보
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
KHUsham = msGroup['KHUsham']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = bahavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = downsampling(behav_tmp, signalss[SE][se].shape[0])
            if np.isnan(np.mean(movement_syn[SE][se])): movement_syn[SE][se] = []
            
# signals_raw에서 직접 수정할경우
if False:
    signalss = msFunction.msarray([N])
    for SE in range(N):
        for se in range(len(signalss_raw[SE])):
            allo = np.zeros(signalss_raw[SE][se].shape) * np.nan
            if len(bahavss[SE][se][0]) > 0 and not(np.isnan(np.mean(bahavss[SE][se][0]))):
                behav_thr = bahavss[SE][se][1]
                # if SE >= 230 and behav_thr < 1: behav_thr = 0.15
                movratio = np.mean(movement_syn[SE][se] > behav_thr)
                if movratio == 1: movratio = 0.99; print('mov 100%', SE, se); 
                bratio = (1-movratio) * 0.3
            else: bratio = 0.3
            
            for ROI in range(signalss_raw[SE][se].shape[1]):
                matrix = signalss_raw[SE][se][:,ROI]
                
                minbase = 20
                for jj in range(2000):
                    base = np.sort(matrix)[0:np.max([int(round(matrix.shape[0]*bratio)), minbase])]
                    base_mean = np.median(base)
                    if base_mean != 0: break
                    minbase  += 1
                    
                matrix2 = (matrix-base_mean)/base_mean
                if np.isnan(np.mean(matrix2)): print('nan warning', SE, se, ROI)
                allo[:,ROI] = matrix2
                # plt.plot(matrix2)
            signalss[SE].append(allo)
            if np.isnan(np.mean(signalss[SE][se])): print('nan warning', SE, se)


#%% grouping
group_pain_training = []
group_nonpain_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    if not SE in [179, 181]: # ROI 매칭안되므로 임시 제거
        for se in range(MAXSE):
            painc, nonpainc, test_only = [], [], []
            # snu
            if False:
                nonpainc.append(SE in salineGroup and se in [0,1,2,3,4])
                nonpainc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [0])
                painc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [1])
                
                painc.append(SE in CFAgroup and se in [1,2])
                painc.append(SE in capsaicinGroup and se in [1])
                
                # snu psl pain
                painc.append(SE in pslGroup and se in [1,2])
                nonpainc.append(SE in pslGroup and se in [0])
                nonpainc.append(SE in shamGroup and se in [0,1,2])
                
                # snu psl+
                # painc.append(SE in ipsaline_pslGroup and se in [1,2])
                painc.append(SE in ipsaline_pslGroup and se in [1,3])
                nonpainc.append(SE in ipsaline_pslGroup and se in [0])
                painc.append(SE in ipclonidineGroup and se in [1,3])
                nonpainc.append(SE in ipclonidineGroup and se in [0])
            
                # GBVX 30 mins
                if False:
                    GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
                    nonpainc.append(SE in GBVX and se in [0,1])
                    nonpainc.append(SE in [164, 166] and se in [2,3,4,5])
                    nonpainc.append(SE in [167] and se in [4,5,6,7])
                    nonpainc.append(SE in [172] and se in [4,5,7,8])
                    nonpainc.append(SE in [174] and se in [4,5])
                    nonpainc.append(SE in [177,179,181] and se in [2,3,6,7,10,11])
                    painc.append(SE in [179] and se in [8,9])
                    painc.append(SE in [181] and se in [4,5])
            
                # snu oxali
                if True:
                    painc.append(SE in oxaliGroup and se in [1])
                    painc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [2])
                    nonpainc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [3])
                    nonpainc.append(SE in [188, 189, 200, 201] and se in [2])
                    nonpainc.append(SE in glucoseGroup and se in [0,1,2,3,4])
            
            
            # khu formalin
            painc.append(SE in list(range(230, 239)) and se in [1])
            painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
            painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])
            
            nonpainc.append(SE in list(range(230, 239)) and se in [0])
            nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
            nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
            nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
            
            # khu psl
            nonpainc.append(SE in PSLgroup_khu and se in [0])
            painc.append(SE in PSLgroup_khu and se in [1,2])
            
            nonpainc.append(SE in morphineGroup and se in [0,1])
            # nonpainc.append(SE in morphineGroup and se in [10,11,12])
            painc.append(SE in morphineGroup and se in [2,3,4,5,6,7,8,9])
            
            nonpainc.append(SE in KHUsham and se in range(0,10))
            # nonpainc.append(SE in KHUsham and se in range(10,13))
             
            # PD
            if False:
                nonpainc.append(SE in PDnonpain and se in list(range(2,10)))
                nonpainc.append(SE in PDnonpain and se in list(range(0,2)))
                painc.append(SE in PDpain and se in list(range(2,6)))
                nonpainc.append(SE in PDpain and se in list(range(0,2)))
            
            if [SE, se] in [[285, 4],[290, 5]]: continue # 시간짧음, movement 불일치
            
            if np.sum(np.array(painc)) > 0: group_pain_training.append([SE, se])   
            if np.sum(np.array(nonpainc)) > 0: group_nonpain_training.append([SE, se])
                
#%% keras setup
from tensorflow.keras.callbacks import EarlyStopping
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
        # if current is None:
        #     warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)
        if current > self.value:
            # print('current', current, 'over thr')
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
callbacks = [EarlyStopping_ms(monitor='accuracy', value=0.91, verbose=1)]   

lr = 1e-3 # learning rate
n_hidden = int(2**8) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(2**8) # fully conneted laye node 갯수 # 8 # 원래 6 
    
l2_rate = 1e-3
dropout_rate1 = 0.2 # dropout rate
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

def keras_setup(lr=0.01, batchnmr=False, seed=1, add_fn=None):
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras

    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    # input1 = keras.layers.Input(shape=(FS, 1))
    # input1_1 = Bidirectional(LSTM(n_hidden, return_sequences=False))(input1)
    # input1_1 = Dense(int(n_hidden), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input1_1) # fully conneted layers, relu

    
    input2 = tf.keras.layers.Input(shape=(add_fn))
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
    
    # input10 = Dense(2, kernel_initializer = init, activation='sigmoid')(input10) # fully conneted layers, relu

    merge_4 = Dense(2, kernel_initializer = init, activation='softmax')(input10) # fully conneted layers, relu

    model = tf.keras.models.Model(inputs=input2, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer

    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

def upsampling(X_tmp, Y_tmp, Z_tmp):
    X = np.array(X_tmp)
    Y = np.array(Y_tmp)
    Z = np.array(Z_tmp)
    while True:
        n_ix = np.where(Y[:,0]==1)[0]
        p_ix = np.where(Y[:,1]==1)[0]
        print('sample distributions', 'nonpain', n_ix.shape[0], 'pain', p_ix.shape[0])
        
        nnum = n_ix.shape[0]
        pnum = p_ix.shape[0]
        
        maxnum = np.max([nnum, pnum])
        minnum = np.min([nnum, pnum])
        
        print('ratio', maxnum / minnum)
        addix = np.where(Y[:,np.argmin([nnum, pnum])]==1)[0]
        # if not(maxnum // minnum < 2):
        if maxnum // minnum > 1:
            X = np.append(X, X[addix], axis=0)
            Y = np.append(Y, Y[addix], axis=0)
            Z = np.append(Z, Z[addix], axis=0)
        elif maxnum // minnum == 1:
            rix = random.sample(list(addix), maxnum-minnum)
            X = np.append(X, X[rix], axis=0)
            Y = np.append(Y, Y[rix], axis=0)
            Z = np.append(Z, Z[rix], axis=0)
            break 
        else: break
        print('data set num #', len(Y), np.mean(np.array(Y), axis=0))
    return X, Y, Z

model = keras_setup(lr=lr, seed=0, add_fn=2)
print(model.summary())

#%% XYZgen


settingID = 'model3.1.1_painbase_morphine_outsample_20210824' 
# signal 안함, behav thr 수정, 2 그대로사용, 3에서 set range N으로 minor 수정
# 3그대로, 4에서 cap, sfa, gb/vx base 추가
# 4그대로, 5에서 wanted list만 수정

# major error 수정, se = 0 or 0,1

# wantedlist = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup
# wantedlist = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup + gabapentinGroup  + salineGroup + highGroup3 + morphineGroup
wantedlist = morphineGroup
outsamplelist = [] # PDpain # PDpain # 여기에 outsample 넣어

# GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
nonlabel_group = morphineGroup  # morphine test 용

# twobase = range(247, 306)

RESULT_SAVE_PATH = 'C:\\mass_save\\model3\\' + settingID + '\\'
if not os.path.exists(RESULT_SAVE_PATH): os.mkdir(RESULT_SAVE_PATH)

#%%
repeat = 0
verbose = 0
mssave_final = []
for repeat in range(0, 2):
    X, Y, Z = [], [], []
    X_nonlabel, Z_nonlabel = [], []
    
    THR = 0.2
    target_sig = list(signalss)
    target_sig2 = list(movement_syn)

    matrix = np.zeros((len(target_sig),MAXSE)) * np.nan
    for SE in range(N):
        selist = [2,3]
        if True:
            for stanse in selist:
                if [SE, stanse] in group_pain_training:
                    if len(target_sig2[SE][stanse]) > 0:
                        if np.isnan(np.mean(target_sig2[SE][stanse])): import sys; sys.exit()   
                        bthr = bahavss[SE][stanse][1]
                        vix2 = np.where(target_sig2[SE][stanse] <= bthr)[0]
                        sig = target_sig[SE][stanse][vix2,:]
                        sig2 = np.mean(sig, axis=0)
                        roiNum = target_sig[SE][stanse].shape[1]
                        stand2 = sig2  / np.sum(sig2) * roiNum
                        
                        vix = np.where(target_sig2[SE][stanse] > bthr)[0]
                        if len(vix) == 0: vix = vix2 
                        sig = target_sig[SE][stanse][vix,:]
                        sig2 = np.mean(sig, axis=0)
                        roiNum = target_sig[SE][stanse].shape[1]
                        stand1 = sig2  / np.sum(sig2) * roiNum
                        
                        sig = target_sig[SE][stanse]
                        sig2 = np.mean(sig, axis=0)
                        roiNum = target_sig[SE][stanse].shape[1]
                        stand3 = sig2  / np.sum(sig2) * roiNum
                        
                        for se in range(len(target_sig[SE])):
                            if not se in selist or SE in PSLgroup_khu:
                                behavthr = bahavss[SE][se][0]
                                if len(target_sig2[SE][se]) > 0:
                                    if np.isnan(np.mean(target_sig2[SE][se])): print('exit'); import sys; sys.exit()   
                                    label = None
                                    if [SE, se] in group_nonpain_training: label = [1, 0]
                                    if [SE, se] in group_pain_training: label = [0, 1]
                                    labeled_sample = (not(label is None) and se != stanse) 
                                    nonlabeled_sample = (label is None and se != stanse and SE in nonlabel_group)
                                    
                                    if labeled_sample or nonlabeled_sample:
                                        bthr = bahavss[SE][se][1]
                                        vix = np.where(target_sig2[SE][se] > bthr)[0]
                                        vix2 = np.where(target_sig2[SE][se] <= bthr)[0]
                                    
                                        if len(vix) == 0: vix = vix2  
                                        sig = target_sig[SE][se][vix,:]
                                        sig2 = np.mean(sig, axis=0)
                                        roiNum = target_sig[SE][stanse].shape[1]
                                        exp = sig2  / np.sum(sig2) * roiNum
                                        f1 = np.mean(np.abs(exp - stand1) > THR)
                                        
                                        sig = target_sig[SE][se][vix2,:]
                                        sig2 = np.mean(sig, axis=0)
                                        roiNum = target_sig[SE][stanse].shape[1]
                                        exp = sig2  / np.sum(sig2) * roiNum
                                        f2 = np.mean(np.abs(exp - stand2) > THR)
                                        
                                        # nonvix
                                        sig = target_sig[SE][se]
                                        sig2 = np.mean(sig, axis=0)
                                        roiNum = target_sig[SE][stanse].shape[1]
                                        exp = sig2  / np.sum(sig2) * roiNum
                                        f3 = np.mean(np.abs(exp - stand3) > THR)
                                        
                                        f4 = np.mean(signalss[SE][se])
                                        if f4 > 150: print(SE, se, 'error check'); print('exit'); import sys; sys.exit()
                                        
                                        f5 = np.std(signalss[SE][se]) # / np.mean(signalss[SE][se])
                                        if f5 > 2000: f5 = 0; print(SE, se, 'error check'); import sys; sys.exit()
                                        
                                        label = None
                                        if [SE, se] in group_nonpain_training: label = [1, 0]
                                        if [SE, se] in group_pain_training: label = [0, 1]
                                        
                                        if labeled_sample:
                                            bthr = bahavss[SE][se][1]
                                            f0 = np.mean(movement_syn[SE][se] > bthr)
                                            X.append([f0, f1, f2, f3, f4, f5])
                                            Y.append(label)
                                            Z.append([SE, se])
                                            
                                        elif nonlabeled_sample:
                                            X_nonlabel.append([f0, f1, f2, f3, f4, f5])
                                            Z_nonlabel.append([SE, se])
    
    ###
    fn = len(X[0])
    model = keras_setup(lr=lr, seed=0, add_fn=fn)
    X = np.array(X); X_nonlabel = np.array(X_nonlabel)
    Y = np.array(Y)
    Z = np.array(Z); Z_nonlabel = np.array(Z_nonlabel)
    ### outsample test
    outsample = []
    for t in outsamplelist:
        outsample += list(np.where(Z[:,0]==t)[0])
        
    tlist2 = list(range(len(Z)))
    trlist = list(set(tlist2)-set(outsample))
    X2 = X[trlist]; X_te_outsample = X[outsample]  
    Y2 = Y[trlist]; Y_te_outsample = Y[outsample]
    Z2 = Z[trlist]; Z_te_outsample = Z[outsample]
    
    print('repeat', repeat, 'data num', len(Y2), 'Y2 dis', np.mean(Y2, axis=0))
    mssave = msFunction.msarray([N,MAXSE])
    for cv in range(0, len(wantedlist)):
        if type(wantedlist[cv]) != list: cvlist = [wantedlist[cv]]
        else: cvlist = wantedlist[cv]
        
        tlist2 = list(range(len(Z2)))
        telist = []
        for t in cvlist:
            telist += list(np.where(Z2[:,0]==t)[0])
        trlist = list(set(tlist2)-set(telist))
        
        X_tr = X2[trlist]; X_te = X2[telist]
        Y_tr = Y2[trlist]; Y_te = Y2[telist]
        Z_tr = Z2[trlist]; Z_te = Z2[telist]
        
        X_tr, Y_tr, Z_tr = upsampling(X_tr, Y_tr, Z_tr)        
        if len(Y_te) > 0:
            final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_' + str(cv) + '_final.h5'
            if not(os.path.isfile(final_weightsave)) or True:
                print('learning', cvlist)
                print('tr distribution', np.mean(Y_tr, axis=0))
                print('te distribution', np.mean(Y_te, axis=0))
                
                model = keras_setup(lr=lr, seed=0, add_fn=fn)
                for epoch in range(4000):
                    hist = model.fit(X_tr, Y_tr, batch_size=2**11, epochs=1, verbose=verbose, validation_data= (X_te, Y_te))
                    acc = list(np.array(hist.history['accuracy']))[-1]
                    if acc > 0.75 and epoch > 500: break
                model.save_weights(final_weightsave)
        # test
            model.load_weights(final_weightsave)
            for n in range(len(Z_te)):
                teSE = Z_te[n][0]; tese = Z_te[n][1]
                mssave[teSE][tese].append(model.predict(np.array([X_te[n]]))[0][1])
                if verbose==1: print(teSE, tese, np.mean(mssave[teSE][tese]))
            
    # outsample, learning, test
    if len(Z_te_outsample) > 0: # 있을때만 해
        final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_outsample_final.h5'
        if not(os.path.isfile(final_weightsave)) or False:
            model = keras_setup(lr=lr, seed=0, add_fn=fn)
            for epoch in range(4000):
                hist = model.fit(X2, Y2, batch_size=2**11, epochs=1, verbose=verbose, validation_data= (X_te_outsample, Y_te_outsample))
                acc = list(np.array(hist.history['accuracy']))[-1]
                if acc > 0.75 and epoch > 500: break
            model.save_weights(final_weightsave)
            
        model.load_weights(final_weightsave)
        for n in range(len(Z_te_outsample)):
            teSE = Z_te_outsample[n][0]; tese = Z_te_outsample[n][1]
            mssave[teSE][tese].append(model.predict(np.array([X_te_outsample[n]]))[0][1])
            if verbose==1: print(teSE, tese, np.mean(mssave[teSE][tese]))
            
    # manual test
    if len(Z_nonlabel) > 0: # 있을때만 해
        final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_outsample_final.h5'
        if not(os.path.isfile(final_weightsave)) or False:
            model = keras_setup(lr=lr, seed=0, add_fn=fn)
            for epoch in range(4000):
                hist = model.fit(X2, Y2, batch_size=2**11, epochs=1, verbose=verbose, validation_data= (X_te_outsample, Y_te_outsample))
                acc = list(np.array(hist.history['accuracy']))[-1]
                if acc > 0.75 and epoch > 500: break
            model.save_weights(final_weightsave)
            
        model.load_weights(final_weightsave)
        for n in range(len(Z_nonlabel)):
            teSE = Z_nonlabel[n][0]; tese = Z_nonlabel[n][1]
            mssave[teSE][tese].append(model.predict(np.array([X_nonlabel[n]]))[0][1])
            if verbose==1: print(teSE, tese, np.mean(mssave[teSE][tese]))
    
    mssave2 = np.zeros((N,MAXSE)) * np.nan
    for row in range(N):
        for col in range(MAXSE):
            mssave2[row, col] = np.nanmean(mssave[row][col])

    mssave_final.append(mssave2)

savepath = RESULT_SAVE_PATH + 'mssave_final.pickle'
with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(mssave_final, f, pickle.HIGHEST_PROTOCOL)
    print(savepath, '저장되었습니다.')

#%% 평가
loadpath = RESULT_SAVE_PATH + 'mssave_final.pickle'
with open(loadpath, 'rb') as f:  # Python 3: open(..., 'wb')
    mssave_final = pickle.load(f)

mssave_final2 = np.array(mssave_final)
print('mssave_final2.shape', mssave_final2.shape)

ix1 = list(range(0, len(mssave_final2), 2))
ix2 = list(range(1, len(mssave_final2), 2))

mssave = np.nanmean(mssave_final2, axis=0)

plt.figure()
plt.plot(np.nanmean(mssave[morphineGroup], axis=0), c='b')
plt.plot(np.nanmean(mssave[KHUsham], axis=0), c='r')

import scipy
msplot = mssave[morphineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave[KHUsham,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')


np.nanmean(mssave[PSLgroup_khu, :], axis=0)

import sys; sys.exit()

#%% 평가2 - day간 mean or max

same_days = [[2,3], [4,5], [6,7], [8,9], [10,11,12]]
mssave2 = np.zeros((N, len(same_days))) * np.nan
for i in range(len(same_days)):
    mssave2[:,i] = np.nanmean(mssave[:, same_days[i]], axis=1)
    
plt.figure()
plt.plot(np.nanmean(mssave2[morphineGroup,:], axis=0), c='b')
plt.plot(np.nanmean(mssave2[KHUsham,:], axis=0), c='r')

import scipy
msplot = mssave2[morphineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave2[KHUsham,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

class0 = mssave2[KHUsham,:4].flatten()
class1 = mssave2[morphineGroup,:4].flatten()
roc, acc = msROC(class0, class1); print(roc, acc)


#%% 평가2 - all day mean or max

same_days = [[2,3,4,5,6,7,8,9]]
mssave2 = np.zeros((N, len(same_days))) * np.nan
for i in range(len(same_days)):
    mssave2[:,i] = np.nanmax(mssave[:, same_days[i]], axis=1)
    
plt.figure()
plt.plot(np.nanmean(mssave2[morphineGroup,:], axis=0), c='b')
plt.plot(np.nanmean(mssave2[KHUsham,:], axis=0), c='r')

import scipy
msplot = mssave2[morphineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave2[KHUsham,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

class0 = mssave2[KHUsham,:4].flatten()
class1 = mssave2[morphineGroup,:4].flatten()
roc, acc = msROC(class0, class1); print(roc, acc)

#%% PD pain

forsync = np.isnan(mssave_final2[0] * mssave_final2[1])==0



    # mssave = mssave_final2[i]
for row in morphineGroup:
    plt.plot(np.nanmean(mssave_final2[:,row,:], axis=0))
    
    # plt.figure()
    plt.title('morphine')
    plt.plot(np.nanmean(mssave[morphineGroup,:], axis=0))
    
    # plt.figure()
    # plt.title('PD')
    # plt.plot(np.nanmean(mssave[PDpain,:], axis=0))
    # plt.plot(np.nanmean(mssave[PDnonpain], axis=0))


mssave = np.nanmean(mssave_final2[[1,3,5,7,9]], axis=0)
plt.figure()
plt.plot(np.nanmean(mssave[pslGroup + ipsaline_pslGroup + ipclonidineGroup,:], axis=0))
plt.plot(np.nanmean(mssave[shamGroup,:], axis=0))

highGroup3_1 = [247, 248, 250, 251, 257, 258, 259, 262]
highGroup3_2 = [252, 253, 256, 260, 261, 265, 266, 267, 269, 272]
highGroup3_3= [249, 255, 263, 264, 268, 270, 271]

plt.figure()
plt.plot(np.nanmean(mssave[highGroup3_1,:], axis=0))
plt.plot(np.nanmean(mssave[highGroup3_2,:], axis=0))
plt.plot(np.nanmean(mssave[highGroup3_3,:], axis=0))

#%%

plt.figure()
plt.title('KHU PSL')
plt.plot(np.nanmean(mssave[PSLgroup_khu,:], axis=0))

plt.figure()
plt.title('KHU PSL')
plt.plot(np.nanmean(mssave[highGroup3,:], axis=0))

plt.figure()
plt.title('SNU PSL')
plt.plot(np.nanmean(mssave[pslGroup,:], axis=0))
plt.plot(np.nanmean(mssave[shamGroup,:], axis=0))

plt.plot(np.nanmean(mssave[pslGroup + ipsaline_pslGroup + ipclonidineGroup,:], axis=0))
plt.plot(np.nanmean(mssave[shamGroup,:], axis=0))

p, nop = [], []
for SE in gabapentinGroup:
    for se in range(MAXSE):
        if not np.isnan(mssave[SE,se]):
            if [SE, se] in group_pain_training:
                p.append(mssave[SE,se])
            if [SE, se] in group_nonpain_training:
                nop.append(mssave[SE,se])
print(np.mean(nop))           

plt.plot(np.nanmean(mssave[gabapentinGroup,:], axis=0))

plt.figure()
plt.title('PSL SNU')
plt.plot(np.nanmedian(mssave[oxaliGroup,:], axis=0))
plt.plot(np.nanmedian(mssave[glucoseGroup,:], axis=0))

plt.figure()
plt.title('SNU saline')
plt.plot(np.nanmean(mssave[salineGroup,:], axis=0))
print('formalin', np.nanmean(mssave[highGroup + highGroup2 + midleGroup + ketoGroup,:]))
print('cap', np.nanmean(mssave[capsaicinGroup,:]))
print('cfa', np.nanmean(mssave[CFAgroup,:]))

savepath = 'C:\\mass_save\\predict_result_0707_2.pickle'
with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)
    print(savepath, '저장되었습니다.')

#%%

# PD evaluation
meanmatrix = np.zeros((N,5))
for i in range(0, 5):
    for SE in PDpain + PDnonpain:
        meanmatrix[SE,i] = np.nanmean(mssave[SE,i*2:(i+1)*2])
        
AA_PDpain = meanmatrix[PDpain,:]
AA_PDnonpain = meanmatrix[PDnonpain,:]

plt.plot(np.nanmean(AA_PDpain, axis=0))
plt.plot(np.nanmean(AA_PDnonpain, axis=0))

# SNU PSL evaluation
AA_SNU_psl = mssave[pslGroup + ipsaline_pslGroup + ipclonidineGroup,:4]
AA_SNU_sham = mssave[shamGroup,:4]

plt.figure()
plt.plot(np.nanmean(AA_SNU_psl, axis=0))
plt.plot(np.nanmean(AA_SNU_sham, axis=0))

#%% GB/VX evaluation
GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
base, GBVX3, GBVX10, GBVX15, PSL15 = [], [], [], [], []
for SE in range(N):
    for se in range(MAXSE):
        base_1 = SE in GBVX and se in [0,1] # base
    
        GBVX10_1 = SE in [164, 166] and se in [2,3] # GBVX 10
        GBVX10_2 = SE in [167] and se in [6,7] # GBVX 10
        GBVX10_3 = SE in [172] and se in [7,8] # GBVX 10
        GBVX10_4 = SE in [177,179] and se in [6,7] # GBVX 10

        GBVX3_1 = SE in [167] and se in [4,5] # GBVX 3
        GBVX3_2 = SE in [172] and se in [4,5] # GBVX 3
        GBVX3_3 = SE in [174] and se in [4,5] # GBVX 3
        GBVX3_4 = SE in [177,179,181] and se in [2,3] # GBVX 3
        
        PSL15_1 = SE in [181] and se in [4,5] # PSL 15+
        PSL15_2 = SE in [179] and se in [8,9] # PSL 15+
        
        GBVX15_1 = SE in [181] and se in [6,7] # GBVX 15+
        GBVX15_2 = SE in [179] and se in [10,11] # GBVX 15+
        GBVX15_3 = SE in [164, 166] and se in [4,5] # GBVX 15+
        
        if base_1: base.append(mssave[SE,se])
        if GBVX3_1 or GBVX3_2 or GBVX3_3 or GBVX3_4: GBVX3.append(mssave[SE,se])
        if GBVX10_1 or GBVX10_2 or GBVX10_3 or GBVX10_4: GBVX10.append(mssave[SE,se])
        if PSL15_1 or PSL15_2: PSL15.append(mssave[SE,se])

print('base', np.nanmean(base))      
print('GBVX3', np.nanmean(GBVX3))
print('GBVX10', np.nanmean(GBVX10))
print('GBVX15', np.nanmean(GBVX15))
print('PSL15', np.nanmean(PSL15))

AA_SNU_GBVX = pd.concat([pd.DataFrame(base), pd.DataFrame(GBVX3), pd.DataFrame(GBVX10) \
                                       ,pd.DataFrame(PSL15) ,pd.DataFrame(GBVX15)], ignore_index=True, axis=1)



#%% khu formalin test






































