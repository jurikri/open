 # -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:24:00 2021

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
KHU_saline = msGroup['KHU_saline']
PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']
KHUsham = msGroup['KHUsham']
KHU_CFA = msGroup['KHU_CFA']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

PDmorphine = msGroup['PDmorphine']
KHU_PSL_magnolin = msGroup['KHU_PSL_magnolin']

# pdmorphine = list(range(325, 332))

#%%

signalss_df = msFunction.msarray([N, MAXSE])
for SE in range(N):
    for se in range(len(signalss_raw[SE])):
        if len(signalss_raw[SE][se]) > 0:
            tmp = signalss_raw[SE][se]
            mssave = np.zeros(tmp.shape) * np.nan
            roiNum = tmp.shape[1]
            for ROI in range(roiNum):
                vix = np.argsort(tmp[:,ROI])[:int(round(tmp.shape[0]*0.3))]
                m = np.median(tmp[vix,ROI])
                mssave[:,ROI] = (tmp[:,ROI] - m) / m
            
            signalss_df[SE][se] = mssave

#%% grouping
group_pain_training = []
group_nonpain_training = []
group_drug_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    if not SE in [179, 181]: # ROI 매칭안되므로 임시 제거
        for se in range(MAXSE):
            painc, nonpainc, drugc = [], [], []

            snu_base = True
            snu_acute = False
            snu_chronic = True
            
            khu_base = True
            khu_acute = False
            khu_chronic = True
            
            CFAsw = True
            PSLsw = True
            Oxalisw = True

            # snu
            GBVX = [164, 165, 166, 167, 172, 174, 177, 179, 181]
            if snu_base:
                nonpainc.append(SE in salineGroup and se in [0,1,2,3,4])
                nonpainc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [0])
                nonpainc.append(SE in pslGroup and se in [0])
                nonpainc.append(SE in shamGroup and se in [0,1,2])
                nonpainc.append(SE in ipsaline_pslGroup and se in [0])
                nonpainc.append(SE in ipclonidineGroup and se in [0])
                nonpainc.append(SE in GBVX and se in [0,1])
                nonpainc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [3])
                nonpainc.append(SE in [188, 189, 200, 201] and se in [2])
                nonpainc.append(SE in glucoseGroup and se in [0,1,2,3,4])
            
            if snu_acute:
                painc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [1])
                painc.append(SE in capsaicinGroup and se in [1])
                
            if snu_chronic:
                if CFAsw:
                    painc.append(SE in CFAgroup and se in [1,2])
                
                if PSLsw:
                    painc.append(SE in pslGroup and se in [1,2])
                    painc.append(SE in ipsaline_pslGroup and se in [1,3])
                    painc.append(SE in [179] and se in [8,9]) # GBVX group내의 pain
                    painc.append(SE in [181] and se in [4,5]) # GBVX group내의 pain
                
                if Oxalisw:
                    painc.append(SE in oxaliGroup and se in [1])
                    painc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [2])
                    
            if False: # GBVX analgesic effect
                # nonpainc.append(SE in [164, 166] and se in [2,3,4,5]) # GBVX
                # nonpainc.append(SE in [167] and se in [4,5,6,7]) # GBVX
                # nonpainc.append(SE in [172] and se in [4,5,7,8]) # GBVX
                # nonpainc.append(SE in [174] and se in [4,5]) # GBVX
                # nonpainc.append(SE in [177,179,181] and se in [2,3,6,7,10,11]) # GBVX
                pass
                    

            if khu_base:  
                # khu formalin
                nonpainc.append(SE in list(range(230, 239)) and se in [0])
                nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
                nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
                nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
                nonpainc.append(SE in KHU_CFA and se in [0,1,2,3])
                nonpainc.append(SE in PSLgroup_khu and se in [0])
                nonpainc.append(SE in morphineGroup and se in [0,1]) 
                nonpainc.append(SE in KHU_saline and se in [0,1,2])
                
                mslist = [2,3,6,7] # ove rfit check 용
                mslist_test = [4,5,8,9]
                nonpainc.append(SE in KHUsham and se in [2,3,4,5,6,7,8,9])
                # painc.append(SE in KHUsham and se in mslist_test)
                
                nonpainc.append(SE in PDnonpain and se in list(range(2,10)))
                nonpainc.append(SE in PDnonpain and se in list(range(0,2)))
                nonpainc.append(SE in PDpain and se in list(range(0,2)))
                nonpainc.append(SE in PDmorphine and se in [0,1,2,3])
                nonpainc.append(SE in KHU_PSL_magnolin and se in [0,1,2,3])
                
            if khu_acute:  
                painc.append(SE in list(range(230, 239)) and se in [1])
                painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
                painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])

            
            if khu_chronic:
                if CFAsw:
                    painc.append(SE in KHU_CFA and se in [4,5,8,9])
                    # painc.append(SE in KHU_CFA[7:] and se in [4,5])
                    painc.append(SE in KHU_CFA[:7] and se in [10])
                    painc.append(SE in KHU_CFA[7:] and se in [10,11,12,13])
                    
                if PSLsw:
                    painc.append(SE in morphineGroup and se in mslist)
                    painc.append(SE in morphineGroup and se in mslist_test)
                    painc.append(SE in PSLgroup_khu and se in [1,2])
                    painc.append(SE in KHU_PSL_magnolin and se in [4,5,6,7])
                
                if True: # PD pain
                    painc.append(SE in PDpain and se in list(range(2,10))) 

                    painc.append(SE in PDmorphine and se in [4,5])
                    painc.append(SE in [325, 326] and se in [10,11])
                    painc.append(SE in [327, 328] and se in [10,11,16,17])
                    painc.append(SE in [329, 330] and se in [10,11,16,17])
                    painc.append(SE in [331] and se in [10,11,16,17])
                    
                    # i.p. saline
                    painc.append(SE in [325, 326] and se in [12,13,14,15])
                    painc.append(SE in [327, 328] and se in [6,7,8,9])
                    painc.append(SE in [329, 330] and se in [6,7,8,9,18,19,20,21])
                    painc.append(SE in [331] and se in [12,13,14,15])
                    
                    
            if False: # keto analgesic effects
                drugc.append(SE in KHU_CFA[:7] and se in [6,7]) # keto 100 mg/kg
                drugc.append(SE in KHU_CFA[7:] and se in [6,7]) # keto 50 mg/kg
                pass
                
            if True: # morhpine analgesic effects
                drugc.append(SE in morphineGroup and se in [10,11,12]) # morphine
                drugc.append(SE in KHUsham and se in range(10,13)) # morphine
                pass
            
            if False:  # KHU_PSL_magnolin analgesic effects
                drugc.append(SE in KHU_PSL_magnolin and se in [8,9,10,11,12,13]) # morphine
                pass
            
            if False:  # PDmorphine analgesic effects
                drugc.append(SE in [325, 326] and se in [6,7,8,9])
                drugc.append(SE in [327, 328] and se in [12,13,14,15])
                drugc.append(SE in [329, 330] and se in [12,13,14,15,18,19,20,21])
                drugc.append(SE in [331] and se in [6,7,8,9,18,19,20,21])
                pass
            
            if [SE, se] in [[285, 4],[290, 5]]: continue # 시간짧음, movement 불일치
            
            if np.sum(np.array(painc)) > 0: group_pain_training.append([SE, se])   
            if np.sum(np.array(nonpainc)) > 0: group_nonpain_training.append([SE, se])
            if np.sum(np.array(drugc)) > 0: group_drug_training.append([SE, se])

total_list = list(set(list(np.array(group_pain_training)[:,0]) + list(np.array(group_nonpain_training)[:,0]) \
                  + list(np.array(group_drug_training)[:,0])))
    
total_list = total_list + PDpain + PDmorphine
total_list = list(set(total_list))

#%% keras setup

lr = 1e-3 # learning rate
n_hidden = int(2**7) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(2**7) # fully conneted laye node 갯수 # 8 # 원래 6 
    
l2_rate = 1e-3
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

def keras_setup(lr=0.01, dropout_rate1=0, batchnmr=False, seed=1, add_fn=None, layer_1=None, layer_2=None, l2=0.01):
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras

    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    # input1 = keras.layers.Input(shape=(FS, 1))
    
    # input1_1 = Bidirectional(LSTM(n_hidden, return_sequences=False))(input1)
    # input1_1 = Dense(int(n_hidden), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input1_1) # fully conneted layers, relu

    
    input2 = tf.keras.layers.Input(shape=(add_fn))
    input10 = input2
    input10 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2), activation='relu')(input10) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    input10 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2), activation='relu')(input10) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    input10 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2), activation='sigmoid')(input10) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    # input10 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(0.001), activation='relu')(input10) # fully conneted layers, relu

    # input_cocat = keras.layers.Concatenate(axis=1)([input1_1, input2_1])
    
    # input10 = Dense(int(layer_1), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input2_1) # fully conneted layers, relu
    # if batchnmr: input10 = BatchNormalization()(input10)
    # input10 = Dropout(dropout_rate1)(input10) # dropout
    
    # input10 = Dense(int(layer_1/2), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input10) # fully conneted layers, relu
    # if batchnmr: input10 = BatchNormalization()(input10)
    # input10 = Dropout(dropout_rate1)(input10) # dropout
    
    # input10 = Dense(int(layer_2), kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='sigmoid')(input10) # fully conneted layers, relu
    # if batchnmr: input10 = BatchNormalization()(input10)
    # input10 = Dropout(dropout_rate1)(input10) # dropout
    
    # input10 = Dense(2, kernel_initializer = init, activation='sigmoid')(input10) # fully conneted layers, relu

    merge_4 = Dense(3, kernel_initializer = init, activation='softmax')(input10) # fully conneted layers, relu

    model = tf.keras.models.Model(inputs=input2, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer

    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

def upsampling(X_tmp, Y_tmp, Z_tmp, verbose=0):
    X = np.array(X_tmp)
    Y = np.array(Y_tmp)
    Z = np.array(Z_tmp)
    while True:
        ixs = msFunction.msarray([3])
        for label in range(Y.shape[1]):
            ixs[label] = np.where(Y[:,label]==1)[0]
        
        nums = np.zeros((Y.shape[1])) * np.nan
        for label in range(Y.shape[1]):
            nums[label] = len(ixs[label])
        
        maxnum = np.max(nums)
        label_ix = list(range(Y.shape[1]))
        up_ix = list(set(label_ix) - set([np.argmax(nums)]))
        for i in range(len(up_ix)):
            addix = np.where(Y[:,up_ix[i]]==1)[0]
            minnum = len(addix)
            # if not(maxnum // minnum < 2):
            if maxnum // minnum > 1:
                X = np.append(X, X[addix], axis=0)
                Y = np.append(Y, Y[addix], axis=0)
                Z = np.append(Z, Z[addix], axis=0)
            elif maxnum // minnum == 1:
                rix = random.sample(list(addix), int(maxnum-minnum))
                X = np.append(X, X[rix], axis=0)
                Y = np.append(Y, Y[rix], axis=0)
                Z = np.append(Z, Z[rix], axis=0)
                
        tmp = np.mean(np.array(Y), axis=0)
        if np.max(tmp) == np.min(tmp): break
        if verbose or False: 
            print('data set num #', len(Y), np.mean(np.array(Y), axis=0))
    return X, Y, Z

model = keras_setup(lr=lr, seed=0, add_fn=2, layer_1=layer_1, layer_2=layer_1)
print(model.summary())

#%% pathset

settingID = 'model5_20220130_3class_morphine'
# settingID = 'model4.1.1_20211130_1_snuonly' 

SNU_chronicpain = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup + gabapentinGroup + oxaliGroup + glucoseGroup
KHU_chronicpain = KHU_CFA + morphineGroup + KHUsham + KHU_PSL_magnolin
KHU_pdpain = KHU_CFA + morphineGroup + KHUsham 

wantedlist = PSLgroup_khu + morphineGroup + KHUsham

# RESULT_SAVE_PATH = 'D:\\2p_pain\\weight_saves\\211129\\' + settingID + '\\'
RESULT_SAVE_PATH = 'C:\\mass_save\\20220102\\' + settingID + '\\'

if os.path.isdir('K:\\mscode_m2'): RESULT_SAVE_PATH = 'K:\\mscode_m2\\220220102\\' + settingID + '\\'

if not os.path.exists(RESULT_SAVE_PATH): os.mkdir(RESULT_SAVE_PATH)

#%% XYZgen2
THR = 0.3; THR2 = 0.2
repeat = 0
verbose = 0
mssave_final = []

start = time.time()

X, Y, Z = [], [], []
X_nonlabel, Z_nonlabel = [], []


target_sig = list(signalss2)
target_sig_df = list(signalss_df)

target_sig2 = list(movement_syn)

matrix = np.zeros((len(target_sig),MAXSE)) * np.nan

for SE in range(N):
    if SE in [179, 181]: continue
    
    selist = [0]
    if SE in morphineGroup + KHUsham + GBVX + PDpain + PDnonpain: selist = [0,1]
    if SE in KHU_CFA + PDmorphine + KHU_PSL_magnolin: selist = [0,1,2,3]
    
    if SE in list(range(247, 273)): selist = [0,1]

    # selist = [0] # 0
    
    # stand_mean = msFunction.msarray([3])
    for stanse in selist: # nonpain label 없으면 전부 컷됨
        if len(target_sig2[SE][stanse]) > 0:
            if np.isnan(np.mean(target_sig2[SE][stanse])): print('e1'); import sys; sys.exit()   
            bthr = behavss[SE][stanse][1]
            vix = np.where(target_sig2[SE][stanse] > bthr)[0]
            vix2 = np.where(target_sig2[SE][stanse] <= bthr)[0]

            def msstand(target_sig, stanse, vix, vix2):
                import numpy as np
                if len(vix) == 0: vix = vix2 
                sig = target_sig[SE][stanse][vix,:]
                sig2 = np.mean(sig, axis=0)
                roiNum = target_sig[SE][stanse].shape[1]
                stand1 = sig2  / np.sum(sig2) * roiNum
                
                sig = target_sig[SE][stanse][vix2,:]
                sig2 = np.mean(sig, axis=0)
                roiNum = target_sig[SE][stanse].shape[1]
                stand2 = sig2  / np.sum(sig2) * roiNum
                
                sig = target_sig[SE][stanse]
                sig2 = np.mean(sig, axis=0)
                roiNum = target_sig[SE][stanse].shape[1]
                stand3 = sig2  / np.sum(sig2) * roiNum
                return stand1, stand2, stand3
            
            stand1, stand2, stand3 = msstand(target_sig, stanse, vix, vix2)
            stand1_df, stand2_df, stand3_df = msstand(target_sig_df, stanse, vix, vix2)
            
            
            for se in range(len(target_sig[SE])):
                # # self는 비교하지 않음.
                ex0 = se == stanse
                
                # # 동일 headfix 상태에서 반복 이미징은 비교하지 않음.
                if stanse%2==0: t = stanse+1
                if stanse%2==1: t = stanse-1
                ex1 = se==t and len(selist) > 1
                
                if not(ex0 or ex1):
                # if not(se in selist):
                    if len(target_sig2[SE][se]) > 0:
                        if np.isnan(np.mean(target_sig2[SE][se])): print('e2'); import sys; sys.exit()   

                        bthr = behavss[SE][se][1]
                        f0 = np.mean(movement_syn[SE][se] > bthr)
                        
                        vix = np.where(target_sig2[SE][se] > bthr)[0]
                        vix2 = np.where(target_sig2[SE][se] <= bthr)[0]

                        def msexp(stand1, stand2, stand3, target_sig, se, vix, vix2):
                            import numpy as np
                            if len(vix) == 0: vix = vix2  
                            sig = target_sig[SE][se][vix,:]
                            sig2 = np.mean(sig, axis=0)
                            roiNum = target_sig[SE][stanse].shape[1]
                            exp = sig2  / np.sum(sig2) * roiNum
                            f1 = np.mean((exp - stand1) > THR)
                            f11 = np.mean((exp - stand1) < -THR2)
                            
                            sig = target_sig[SE][se][vix2,:]
                            sig2 = np.mean(sig, axis=0)
                            roiNum = target_sig[SE][stanse].shape[1]
                            exp = sig2  / np.sum(sig2) * roiNum
                            f2 = np.mean((exp - stand2) > THR)
                            f22 = np.mean((exp - stand2) < -THR2)
                            
                            # nonvix
                            sig = target_sig[SE][se]
                            sig2 = np.mean(sig, axis=0)
                            roiNum = target_sig[SE][stanse].shape[1]
                            exp = sig2  / np.sum(sig2) * roiNum
                            f3 = np.mean((exp - stand3) > THR)
                            f33 = np.mean((exp - stand3) < -THR2)
                            return f1, f2, f3, f11, f22, f33
                        
                        f1, f2, f3, f11, f22, f33 = msexp(stand1, stand2, stand3, target_sig, se, vix, vix2)
                        f6, f7, f8, f66, f77, f88 = msexp(stand1_df, stand2_df, stand3_df, target_sig_df, se, vix, vix2)
                        
                        f4 = np.mean(signalss2[SE][se])
                        if f4 > 150: print(SE, se, 'error check'); import sys; sys.exit()
                        
                        f5 = np.std(signalss2[SE][se]) # / np.mean(signalss[SE][se])
                        if f5 > 2000: f5 = 0; print(SE, se, 'error check'); import sys; sys.exit()
                        
                        label = [0, 0, 0]
                        if [SE, se] in group_nonpain_training: label = [1, 0, 0]
                        if [SE, se] in group_pain_training: label = [0, 1, 0]
                        if [SE, se] in group_drug_training: label = [0, 0, 1]
                         
                        
                        xtmp = [f1, f2, f3, f11, f22, f33, f6, f7, f8, f66, f77, f88]
                        
                        # if SE in nonlabels:
                        #     X_nonlabel.append(xtmp)
                        #     Z_nonlabel.append([SE, se])  
                            
                        if SE in total_list:
                            X.append(xtmp)
                            Y.append(label)
                            Z.append([SE, se])
                                
#% feature 추가 생성

savename = 'C:\\SynologyDrive\\2p_data\\' + 'inter_corr.pickle'
if os.path.isdir('K:\\mscode_m2'): savename = 'K:\\SynologyDrive\\2p_data\\' + 'inter_corr.pickle'

if not(os.path.isfile(savename)):
    inter_corr = np.zeros((N, MAXSE)) * np.nan
    for SE in tqdm(range(N)):
        for se in range(len(signalss_raw[SE])):
            xdata = np.array(signalss2[SE][se])
            roiNum = xdata.shape[1]
            rmatrix = np.zeros((roiNum,roiNum)) * np.nan
            for ROI in range(roiNum):
                for ROI2 in range(ROI+1, roiNum):
                    rmatrix[ROI,ROI2] = scipy.stats.pearsonr(xdata[:,ROI], xdata[:,ROI2])[0]
            inter_corr[SE,se] = np.nanmean(rmatrix)

    with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(inter_corr, f, pickle.HIGHEST_PROTOCOL)
        print(savename, '저장되었습니다.')

#% 데이터에 feature 추가
with open(savename, 'rb') as f:  # Python 3: open(..., 'wb')
    inter_corr = pickle.load(f)
        
for i in range(len(X)):
    SE = Z[i][0]; se = Z[i][1]
    X[i] = X[i] + [inter_corr[SE,se]]
        
                                                    
#%%
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

nonpain = np.where(Y[:,0]==1)[0]
pain = np.where(Y[:,1]==1)[0]

for f in range(X.shape[1]):
    plt.figure()
    plt.title(str(f))
    plt.hist(X[nonpain,f], bins=100, density=True, alpha=0.5)
    plt.hist(X[pain,f], bins=100, density=True, alpha=0.5)
    msFunction.msROC(X[nonpain,f], X[pain,f], figsw=True, repeat=2)
    
    # plt.figure()
    # plt.title(str(f))
    # msplot_mean = [np.mean(X[nonpain,f]), np.mean(X[pain,f])]
    # e = [np.std(X[nonpain,f]), np.std(X[pain,f])]
    # plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

#%% 검증
def ms_report(mssave):
    plt.figure()
    msplot = mssave[morphineGroup,2:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    msplot = mssave[KHUsham,2:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
def ms_report_2d(mssave, target):
    def ms_2dscatter(x,y,c):
        plt.scatter(np.nanmean(x), np.nanmean(y), c=c)
        plt.errorbar(np.nanmean(x), np.nanmean(y), yerr=scipy.stats.sem(x, nan_policy='omit'), fmt="o", c=c)
        plt.errorbar(np.nanmean(x), np.nanmean(y), xerr=scipy.stats.sem(y, nan_policy='omit'), fmt="o", c=c)

    
    plt.figure()
    x_pain, x_nonpain, x_drug = [], [], []
    y_pain, y_nonpain, y_drug = [], [], []
    for SE in target:
        for se in range(MAXSE):
            if [SE, se] in group_pain_training: 
                y_pain.append(mssave[SE,se,:][0])
                x_pain.append(mssave[SE,se,:][1])
                
            if [SE, se] in group_nonpain_training: 
                y_nonpain.append(mssave[SE,se,:][0])
                x_nonpain.append(mssave[SE,se,:][1])
                
            if [SE, se] in group_drug_training: 
                y_drug.append(mssave[SE,se,:][0])
                x_drug.append(mssave[SE,se,:][1])
                
    x = list(x_pain)
    y = list(y_pain)
    c = 'r'
    ms_2dscatter(x,y,c)
    
    x = list(x_nonpain)
    y = list(y_nonpain)
    c = 'b'
    ms_2dscatter(x,y,c)
    
    x = list(x_drug)
    y = list(y_drug)
    c = 'g'
    ms_2dscatter(x,y,c)
    
def ms_report_2d_mice(mssave, target):

    plt.figure()
    xydata = np.zeros((2,3,N,MAXSE)) * np.nan
    for SE in target:
        for se in range(MAXSE):
            if [SE, se] in group_pain_training: 
                xydata[1,0,SE,se] = mssave[SE,se,:][0]
                xydata[0,0,SE,se] = mssave[SE,se,:][1]
                
            if [SE, se] in group_nonpain_training: 
                xydata[1,1,SE,se] = mssave[SE,se,:][0]
                xydata[0,1,SE,se] = mssave[SE,se,:][1]
                
            if [SE, se] in group_drug_training: 
                xydata[1,2,SE,se] = mssave[SE,se,:][0]
                xydata[0,2,SE,se] = mssave[SE,se,:][1]
    
    x = msFunction.nanex(np.nanmean(xydata[0,0,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,0,:,:], axis=1))
    plt.scatter(x, y, c='r')
    x = msFunction.nanex(np.nanmean(xydata[0,1,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,1,:,:], axis=1))
    plt.scatter(x, y, c='b')
    x = msFunction.nanex(np.nanmean(xydata[0,2,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,2,:,:], axis=1))
    plt.scatter(x, y, c='g')
 
def ms_report_magnolin(mssave, target):
    plt.figure()
    xydata = np.zeros((2,5,N,MAXSE)) * np.nan
    for SE in target:
        for se in range(MAXSE):
            if se in [0,1,2,3]:
                xydata[1,0,SE,se] = mssave[SE,se,:][0]
                xydata[0,0,SE,se] = mssave[SE,se,:][1]
                
            if se in [4,5,6,7]:
                xydata[1,1,SE,se] = mssave[SE,se,:][0]
                xydata[0,1,SE,se] = mssave[SE,se,:][1]
                
            if se in [8,9]:
                xydata[1,2,SE,se] = mssave[SE,se,:][0]
                xydata[0,2,SE,se] = mssave[SE,se,:][1]
                
            if se in [10,11]:
                xydata[1,3,SE,se] = mssave[SE,se,:][0]
                xydata[0,3,SE,se] = mssave[SE,se,:][1]
                
            if se in [12,13]:
                xydata[1,4,SE,se] = mssave[SE,se,:][0]
                xydata[0,4,SE,se] = mssave[SE,se,:][1]
    
    x = msFunction.nanex(np.nanmean(xydata[0,0,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,0,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='b')
    x = msFunction.nanex(np.nanmean(xydata[0,1,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,1,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='r')
    x = msFunction.nanex(np.nanmean(xydata[0,2,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,2,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='g')
    x = msFunction.nanex(np.nanmean(xydata[0,3,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,3,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='k')
    x = msFunction.nanex(np.nanmean(xydata[0,4,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,4,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='y')
    
def ms_report_morphine(mssave):
    plt.figure()
    xydata = np.zeros((2,5,N,MAXSE)) * np.nan
    for SE in range(N):
        for se in range(MAXSE):
            if SE in KHUsham and se in [2,3,4,5]:
                xydata[1,0,SE,se] = mssave[SE,se,:][0]
                xydata[0,0,SE,se] = mssave[SE,se,:][1]
                
            if SE in KHUsham and se in [6,7,8,9]:
                xydata[1,1,SE,se] = mssave[SE,se,:][0]
                xydata[0,1,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [2,3,4,5]:
                xydata[1,2,SE,se] = mssave[SE,se,:][0]
                xydata[0,2,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [6,7,8,9]:
                xydata[1,3,SE,se] = mssave[SE,se,:][0]
                xydata[0,3,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [10,11,12]:
                xydata[1,4,SE,se] = mssave[SE,se,:][0]
                xydata[0,4,SE,se] = mssave[SE,se,:][1]
    
    x = msFunction.nanex(np.nanmean(xydata[0,0,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,0,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='b')
    x = msFunction.nanex(np.nanmean(xydata[0,1,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,1,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='r')
    x = msFunction.nanex(np.nanmean(xydata[0,2,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,2,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='g')
    x = msFunction.nanex(np.nanmean(xydata[0,3,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,3,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='k')
    x = msFunction.nanex(np.nanmean(xydata[0,4,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,4,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='y')

def ms_report_morphine(mssave):
    plt.figure()
    xydata = np.zeros((2,5,N,MAXSE)) * np.nan
    for SE in range(N):
        for se in range(MAXSE):
            if SE in KHUsham and se in [2,3,4,5]:
                xydata[1,0,SE,se] = mssave[SE,se,:][0]
                xydata[0,0,SE,se] = mssave[SE,se,:][1]
                
            if SE in KHUsham and se in [6,7,8,9]:
                xydata[1,1,SE,se] = mssave[SE,se,:][0]
                xydata[0,1,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [2,3,4,5]:
                xydata[1,2,SE,se] = mssave[SE,se,:][0]
                xydata[0,2,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [6,7,8,9]:
                xydata[1,3,SE,se] = mssave[SE,se,:][0]
                xydata[0,3,SE,se] = mssave[SE,se,:][1]
                
            if SE in morphineGroup and se in [10,11,12]:
                xydata[1,4,SE,se] = mssave[SE,se,:][0]
                xydata[0,4,SE,se] = mssave[SE,se,:][1]
    
    x = msFunction.nanex(np.nanmean(xydata[0,0,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,0,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='b')
    x = msFunction.nanex(np.nanmean(xydata[0,1,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,1,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='r')
    x = msFunction.nanex(np.nanmean(xydata[0,2,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,2,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='g')
    x = msFunction.nanex(np.nanmean(xydata[0,3,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,3,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='k')
    x = msFunction.nanex(np.nanmean(xydata[0,4,:,:], axis=1))
    y = msFunction.nanex(np.nanmean(xydata[1,4,:,:], axis=1))
    plt.scatter(np.mean(x), np.mean(y), c='y')  

def ms_report_cfa(mssave):
    KHU_CFA_100 = KHU_CFA[:7]
    KHU_CFA_50 = KHU_CFA[7:]

    target_group = list(KHU_CFA_50)

    plt.figure()
    msplot = mssave[target_group,0:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    
    target_group = list(KHU_CFA_100)

    plt.figure()
    msplot = mssave[target_group,0:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
def ms_report_snu_chronic(mssave):
    msplot = mssave[shamGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

    msplot = mssave[pslGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

    msplot = mssave[ipsaline_pslGroup + ipclonidineGroup,:][:,[1,3]]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='g')


    GBVX_nonpain_d3, GBVX_nonpain_d10 = [], []
    for SE in [164, 166, 167, 172, 174, 177, 179, 181]:
        d3c, d10c = [], []
        for se in range(12):
            d10c.append(SE in [164, 165, 166] and se in [2,3])
            
            d3c.append(SE in [167] and se in [4,5])
            d10c.append(SE in [167] and se in [6,7])
            
            d3c.append(SE in [172] and se in [4,5])
            d10c.append(SE in [172] and se in [8,9])
            
            d3c.append(SE in [174] and se in [4,5])
            
            d3c.append(SE in [177,179,181] and se in [2,3])
            d10c.append(SE in [177,179] and se in [6,7])

            if np.sum(np.array(d3c)) > 0: GBVX_nonpain_d3.append(mssave[SE,se])
            if np.sum(np.array(d10c)) > 0: GBVX_nonpain_d10.append(mssave[SE,se])
            
        
    GBVX_nonpain_d3 = msFunction.nanex(GBVX_nonpain_d3)
    GBVX_nonpain_d10 = msFunction.nanex(GBVX_nonpain_d10)

    msplot = np.zeros((50, 2)) * np.nan
    msplot[:len(GBVX_nonpain_d3),0] = GBVX_nonpain_d3
    msplot[:len(GBVX_nonpain_d10),1] = GBVX_nonpain_d10
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='k')

    #%SNU oxaliplatin
    plt.figure()
    msplot = mssave[oxaliGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

    msplot = mssave[glucoseGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    
def ms_report_khu_magnolin(mssave):    
    plt.figure()
    msplot = mssave[KHU_PSL_magnolin,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    
    #% mssave[KHU_PSL_magnolin,:] subject mean
    if False:
        subject_mean = [[0,1,2,3], [4,5,6,7], [8,9], [10,11], [12,13]]
        msmatrix = np.zeros((N, len(subject_mean))) * np.nan
        
        for i in range(len(subject_mean)):
            msmatrix[KHU_PSL_magnolin,i] = np.mean(mssave[KHU_PSL_magnolin,:][:, subject_mean[i]], axis=1)
            
        plt.figure()
        msplot = msmatrix[KHU_PSL_magnolin,:]
        msplot_mean = np.nanmean(msplot, axis=0)
        e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
        plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

def msplot_PD(mssave2):
    PDmorphineA = [325, 326]
    PDmorphineB = [327, 328]
    PDmorphineC = [329, 330]
    PDmorphineD = [331]
    
    visse = 9
    PDmorphine_matrix = msFunction.msarray([len(PDmorphine), visse])
    
    for ix, SE in enumerate(PDmorphine):
        for se in range(MAXSE):
            if SE in PDmorphine and se in [0,1,2,3]:
                PDmorphine_matrix[ix][0].append(mssave2[SE,se])
            
            if SE in PDmorphineA and se in [5,6]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [6,7]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [8,9]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [12,13,14,15]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
                
            if SE in PDmorphineB and se in [5,6]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [6,7,8,9]:
                PDmorphine_matrix[ix][2].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [10,11]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [12,13]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [14,15]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [18,19]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [20,21]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
                
            if SE in PDmorphineC and se in [5,6]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [6,7,8,9]:
                PDmorphine_matrix[ix][2].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [12,13]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [14,15]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [18,19]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [20,21]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
                
            if SE in PDmorphineD and se in [5,6]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [6,7]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [8,9]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [12,13,14,15]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [18,19]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [20,21]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])

    PDmorphine_matrix2 = np.zeros((len(PDmorphine), visse)) * np.nan
    for ix in range(len(PDmorphine)):
        for se in range(visse):
            PDmorphine_matrix2[ix,se] = np.mean(PDmorphine_matrix[ix][se])
    
    plt.figure()
    plt.plot(np.nanmean(PDmorphine_matrix2, axis=0))
    msplot = PDmorphine_matrix2
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    
    plt.figure()
    msplot = mssave2[PDpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    plt.plot(msplot_mean, c='r')
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    msplot = mssave2[PDnonpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    plt.plot(msplot_mean, c='b')
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

#%% total tr
X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)
vix = np.sum(Y, axis=1)>0
X_total = X[vix]; Y_total = Y[vix]; Z_total = Z[vix]
# X_total, Y_total, Z_total = upsampling(X_total, Y_total, Z_total)       
print(np.mean(Y_total, axis=0), np.sum(Y_total, axis=0))

layer_1 = 20; epochs = 3000 # 30
# 20...이엇나?

from sklearn.decomposition import PCA
pca_nc = 6
pca = PCA(n_components=pca_nc)
pca.fit(X_total)
X_total = pca.transform(X_total)

dropout_rate1 = 0.1
l2 = 0.1
batchnmr = True

model = keras_setup(lr=lr, seed=0, add_fn=X_total.shape[1], layer_1=layer_1, \
                    batchnmr=batchnmr, dropout_rate1=dropout_rate1, l2=l2)
print(model.summary())

#%% overfit
if False:
    print(len(Y_total), np.mean(Y_total, axis=0))
    hist = model.fit(X_total, Y_total, batch_size=2**11, epochs=epochs, verbose=1)
    
    if True:
        xhat = model.predict(X_total)[:,1]
        
        painset = np.where(Y_total[:,1]==1)[0]
        realpain = np.where(xhat>0.6)[0]
        
        eix = list(set(list(painset)) - set(list(realpain)))
        Y_total[eix,1] = 0
        vix = np.sum(Y_total, axis=1)>0
        X_total = X_total[vix]; Y_total = Y_total[vix]; Z_total = Z_total[vix]
        
        print(len(Y_total), np.mean(Y_total, axis=0))
                 
        model = keras_setup(lr=lr, seed=0, add_fn=X_total.shape[1], layer_1=layer_1, \
                            batchnmr=batchnmr, dropout_rate1=dropout_rate1, l2=l2)
        hist = model.fit(X_total, Y_total, batch_size=2**11, epochs=epochs, verbose=1)
        
    X2 = pca.transform(X)
    xhat = model.predict(X2)
    mssave_total = msFunction.msarray([N,MAXSE])
    for i in range(len(Z)):
        teSE = Z[i][0]
        tese = Z[i][1]
        mssave_total[teSE][tese].append(xhat[i])
    
    mssave2 = np.zeros((N,MAXSE,2)) * np.nan
    for row in range(N):
        for col in range(MAXSE):
            tmp = np.nanmean(mssave_total[row][col], axis=0)
            if not(np.isnan(np.mean(tmp))): mssave2[row, col, :] = tmp[:2]
            
    ms_report(mssave2[:,:,1])     
    msplot = mssave2[:,:,1][PSLgroup_khu,:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='orange')
    
    # ms_report_cfa(mssave2[:,:,1])
    # ms_report_khu_magnolin(mssave2[:,:,1])
    # msplot_PD(mssave2[:,:,1])
    
    savepath = RESULT_SAVE_PATH + 'eix.pickle'
    with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(eix, f, pickle.HIGHEST_PROTOCOL)
        print(savepath, '저장되었습니다.')
    import sys; sys.exit()   
    
else:
    #% dataload
    savepath = RESULT_SAVE_PATH + 'eix.pickle'
    with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
        eix = pickle.load(f)
        
#%% cv 생성
X = np.array(X); # X_nonlabel = np.array(X_nonlabel)
Y = np.array(Y)
Z = np.array(Z); # Z_nonlabel = np.array(Z_nonlabel)

vix = np.where(np.sum(Y, axis=1)>0)[0]
X_vix = np.array(X[vix]); Y_vix = np.array(Y[vix]); Z_vix = np.array(Z[vix])
tnum = len(Y_vix)

Y_vix[eix,1] = 0
vix = np.sum(Y_vix, axis=1)>0
X_vix = X_vix[vix]; Y_vix = Y_vix[vix]; Z_vix = Z_vix[vix]
print('excluded ratio', 1 - (len(Y_vix)/tnum))

def SEse_find(Z=None, Y=None, SE=None, se=None):
    if SE in wantedlist:
        Z = np.array(Z)
        vix = np.where(np.logical_and(Z[:,0]==SE, Z[:,1]==se))[0]
        if len(vix) > 0: return list(vix)
        else: return []
    else: return []

#%% intra subject
if False:
    cvlist = []
    for SE in wantedlist:
        for se in range(MAXSE):
            i = SEse_find(Z=Z_vix, SE=SE, se=se)
            if len(i) > 0: cvlist.append(i)
    print('len(cvlist)', len(cvlist))
#%% between subject
if True:
    cvlist = []
    for SE in wantedlist:
        cvlist_tmp = []
        for se in range(MAXSE):
            i = SEse_find(Z=Z_vix, SE=SE, se=se)
            if len(i) > 0: cvlist_tmp = cvlist_tmp + i
        cvlist.append(cvlist_tmp)
    print('len(cvlist)', len(cvlist))

#%% intra-subejct - day cv
if False:
    cvlist = []
    for SE in range(N):
        if (SE < 247 or SE in PSLgroup_khu) and not(SE in GBVX):
            for se in range(len(signalss_raw[SE])):
                i = SEse_find(Z=Z_vix, SE=SE, se=se)
                if len(i) > 0: cvlist.append(i)
                
        elif SE in GBVX: # nonpain만 사용하는경우 - baseline data가 안나오므로 무효할듯?
            i0 = SEse_find(Z=Z_vix, SE=SE, se=0)
            i1 = SEse_find(Z=Z_vix, SE=SE, se=1)
            i = i0+i1
            if len(i) > 0: 
                import sys; sys.exit()
                cvlist.append(i)
                
        elif SE >= 247 and SE < 273:
            for se in range(3):
                i = SEse_find(Z=Z_vix, SE=SE, se=se)
                if len(i) > 0: cvlist.append(i)
            
            i0 = SEse_find(Z=Z_vix, SE=SE, se=3)
            i1 = SEse_find(Z=Z_vix, SE=SE, se=4)
            i = i0+i1
            if len(i) > 0: cvlist.append(i)
            
        elif SE in PDpain + PDnonpain + KHU_CFA:
            for se in list(range(0,len(signalss_raw[SE]),2)):
                i0 = SEse_find(Z=Z_vix, SE=SE, se=se)
                i1 = SEse_find(Z=Z_vix, SE=SE, se=se+1)
                i = i0+i1
                if SE in [312, 313] and se == 8:
                    i0 = SEse_find(Z=Z_vix, SE=SE, se=8)
                    i1 = SEse_find(Z=Z_vix, SE=SE, se=9)
                    i2 = SEse_find(Z=Z_vix, SE=SE, se=10)
                    i = i0+i1+i2
                if len(i) > 0: cvlist.append(i)
                
        elif SE in morphineGroup + KHUsham:
            for se in list(range(2,len(signalss_raw[SE]),4)):
                i0 = SEse_find(Z=Z_vix, SE=SE, se=se)
                i1 = SEse_find(Z=Z_vix, SE=SE, se=se+1)
                i2 = SEse_find(Z=Z_vix, SE=SE, se=se+2)
                i3 = SEse_find(Z=Z_vix, SE=SE, se=se+3)
                i = i0+i1+i2+i3
                if len(i) > 0: cvlist.append(i)
                
        elif SE in PDmorphine: # nonpain만 사용하는경우
            for se in list(range(0, 4, 2)):
                i0 = SEse_find(Z=Z_vix, SE=SE, se=se)
                i1 = SEse_find(Z=Z_vix, SE=SE, se=se+1)
                i = i0+i1
                if len(i) > 0: cvlist.append(i)
                
        elif SE in KHU_PSL_magnolin: # nonpain만 사용하는경우
            for se in list(range(0, 8, 2)):
                i0 = SEse_find(Z=Z_vix, SE=SE, se=se)
                i1 = SEse_find(Z=Z_vix, SE=SE, se=se+1)
                i = i0+i1
                if len(i) > 0: cvlist.append(i)
                
        else: print(SE, 'is not allocated')

#%% cv training
tr_graph_save = msFunction.msarray([len(cvlist), 4])

# X2 = pca.transform(X)
model = keras_setup(lr=lr, seed=0, add_fn=pca_nc, layer_1=layer_1, \
                    batchnmr=batchnmr, dropout_rate1=dropout_rate1, l2=l2)
    
print(model.summary())
overwrite = True
repeat_save = []
for repeat in range(10): #, 100):
    ### outsample test
    print('repeat', repeat, 'data num', len(Y_vix), 'Y2 dis', np.mean(Y_vix, axis=0))
    mssave = msFunction.msarray([N,MAXSE])
    
    totallist = list(range(len(Y_vix)))
    for cv in range(0, len(cvlist)):
        telist = cvlist[cv]
        if len(telist) > 0:
            trlist = list(set(totallist)-set(telist))
            # print(len(totallist), len(trlist), len(telist))
            X_tr = X_vix[trlist]; X_te = X_vix[telist]
            Y_tr = Y_vix[trlist]; Y_te = Y_vix[telist]
            Z_tr = Z_vix[trlist]; Z_te = Z_vix[telist]
            
            pca = PCA(n_components=pca_nc)
            pca.fit(X_tr)
            X_tr = pca.transform(X_tr)
            X_te = pca.transform(X_te)
            
            # X_tr, Y_tr, Z_tr = upsampling(X_tr, Y_tr, Z_tr)        
            final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_' + str(telist[0]) + '_final.h5'
            if not(os.path.isfile(final_weightsave)) or overwrite:
                if True:
                    print(repeat, 'learning', cv, '/', len(cvlist))
                    print('tr distribution', np.mean(Y_tr, axis=0), np.sum(Y_tr, axis=0))
                    print('te distribution', np.mean(Y_te, axis=0))

                model = keras_setup(lr=lr, seed=0, add_fn=pca_nc, layer_1=layer_1, \
                                    batchnmr=batchnmr, dropout_rate1=dropout_rate1, l2=l2)
                verbose = 0
                if cv == 0: verbose = 1
                hist = model.fit(X_tr, Y_tr, batch_size=2**11, epochs=epochs, verbose=verbose)
                model.save_weights(final_weightsave)
                
            # test
            model.load_weights(final_weightsave)
            yhat = model.predict(X_te)
            
            for n in range(len(yhat)):
                teSE = Z_te[n][0]; tese = Z_te[n][1]
                mssave[teSE][tese].append(yhat[n])

            X2 = pca.transform(X)
            outlist = np.where(np.sum(Y, axis=1)==0)[0]
            yhat_out = model.predict(X2[outlist])
            for out_SE in wantedlist:
                for n in np.where(Z[outlist][:,0]==out_SE)[0]:
                    teSE = Z[outlist][n][0]; tese = Z[outlist][n][1]
                    mssave[teSE][tese].append(yhat_out[n])

    mssave2 = np.zeros((N,MAXSE,2)) * np.nan
    for row in range(N):
        for col in range(MAXSE):
            tmp = np.nanmean(mssave[row][col], axis=0)
            if not(np.isnan(np.mean(tmp))): mssave2[row, col, :] = tmp[:2]
            
    repeat_save.append(mssave2)
    
savepath = RESULT_SAVE_PATH + 'repeat_save.pickle'
with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(repeat_save, f, pickle.HIGHEST_PROTOCOL)
    print(savepath, '저장되었습니다.')

#% dataload
savepath = RESULT_SAVE_PATH + 'repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    
mssave = np.nanmean(np.array(repeat_save), axis=0)


#%%

# morphine
ms_report(mssave[:,:,1])
msplot = mssave[:,:,1][PSLgroup_khu,:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='orange')

# cfa
# ms_report_cfa(mssave[:,:,1])

# PD
# msplot_PD(mssave[:,:,1])

#%% total acc
msacc = []
for i in range(len(Z_vix)):
    SE = Z_vix[i][0]; se = Z_vix[i][1]
    if not np.isnan(mssave[SE,se]):
        msacc.append(Y_vix[i][1] == mssave[SE,se] > 0.5)
print('accuracy', np.mean(msacc))
    

#%%

# mssave10 = []
# for i in range(len(repeat_save)):
#     for j in range(i+1, len(repeat_save)):
#         mssave10.append(np.nanmean(np.abs((np.nanmean((repeat_save[j] - 0.5)/(repeat_save[i] - 0.5))-1)*100)))

# i = 1
# mssave = np.nanmean(np.array(repeat_save[i:i+1]), axis=0)
# ms_report(mssave) 
#%% KHUPSL

plt.figure()
msplot = mssave[morphineGroup,2:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[KHUsham,2:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

import sys; sys.exit()

psl_d3 = mssave[morphineGroup,2:6].flatten()
psl_d10 = mssave[morphineGroup,6:10].flatten()
sham_d3 = mssave[KHUsham,2:6].flatten()
sham_d10 = mssave[KHUsham,6:10].flatten()

msFunction.msROC(sham_d3, psl_d3)
msFunction.msROC(sham_d10, psl_d10)

#%% 평가2 - day간 mean or max

same_days = [[2,3], [4,5], [6,7], [8,9], [10,11,12]]
mssave2 = np.zeros((N, len(same_days))) * np.nan
for i in range(len(same_days)):
    mssave2[:,i] = np.nanmean(mssave[:,:,1][:, same_days[i]], axis=1)
    
plt.figure()
plt.plot(np.nanmean(mssave2[morphineGroup,:], axis=0), c='r')
plt.plot(np.nanmean(mssave2[KHUsham,:], axis=0), c='b')

msplot = mssave2[morphineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave2[KHUsham,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

#%% 평가2 - all day mean or max

same_days = [[2,3,4,5], [6,7,8,9], [10,11,12]]
mssave2 = np.zeros((N, len(same_days))) * np.nan
for i in range(len(same_days)):
    mssave2[:,i] = np.nanmean((mssave[:,:,1])[:, same_days[i]], axis=1)

for i in PSLgroup_khu:
    mssave2[i,:][:2] = mssave[:,:,1][i,1:3]

plt.figure()
msplot = mssave2[morphineGroup+PSLgroup_khu,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave2[KHUsham,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

nonpain = mssave2[KHUsham,:2]
pain = mssave2[morphineGroup+PSLgroup_khu,:2]
print(msFunction.msROC(nonpain, pain))


#%%

msplot = mssave[:,:,1][PSLgroup_khu,:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')


#%% KHU_CFA

KHU_CFA_100 = KHU_CFA[:7]
KHU_CFA_50 = KHU_CFA[7:]

##
target_group = list(KHU_CFA_50)
# plt.figure()
# plt.plot(np.nanmean(mssave[target_group,:], axis=0), c='r')

plt.figure()
msplot = mssave[target_group,0:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')


target_group = list(KHU_CFA_100)
# plt.figure()
# plt.plot(np.nanmean(mssave[target_group,:], axis=0), c='r')

plt.figure()
msplot = mssave[target_group,0:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
 
if False:
    filepath = 'C:\\mass_save\\model3\\fig\\'
    plt.savefig(filepath + 'KHU_PSL_session.png', dpi=1000)
    
    class0 = mssave[morphineGroup,2:10]
    class1 = mssave[KHUsham,2:10]
    accuracy, roc_auc, _ = msFunction.msROC(class1, class0); print(accuracy, roc_auc)
    
    plt.plot(np.nanmean(mssave[PSLgroup_khu, :], axis=0))
    
    #% KHU_CFA day merge
    same_days = [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13]]
    mssave2 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave2[:,i] = np.nanmean(mssave[:, same_days[i]], axis=1)
        
    Aprism = mssave2[target_group,:]
    plt.plot(np.nanmean(mssave2[target_group,:], axis=0))
    
    mssave_basenmr = mssave2[target_group,:] + 0.52
    for i in range(len(mssave_basenmr)):
        mssave_basenmr[i,:] = mssave_basenmr[i,:] / mssave_basenmr[i,2]
    
    
    Aprism_mssave3 = mssave_basenmr



#%%

psl_d3 = mssave2[morphineGroup,0]
psl_d10 = mssave2[morphineGroup,1]
psl_d10_morphine = mssave2[morphineGroup,2]
sham_d3 = mssave2[KHUsham,0]
sham_d10 = mssave2[KHUsham,1]
sham_d10_morphine = mssave2[KHUsham,2]

Aprism2 = pd.DataFrame(sham_d3)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(sham_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(sham_d10_morphine)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10_morphine)), ignore_index=True, axis=1)


Aprism = msFunction.msarray([6])
Aprism[0] += list(msFunction.nanex(sham_d3))
Aprism[1] += list(msFunction.nanex(sham_d10))
Aprism[2] += list(msFunction.nanex(sham_d10_morphine))
Aprism[3] += list(msFunction.nanex(psl_d3))
Aprism[4] += list(msFunction.nanex(psl_d10))
Aprism[5] += list(msFunction.nanex(psl_d10_morphine))

Aprism_info = np.zeros((len(Aprism),3))
for i in range(len(Aprism)):
    Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

nonpain = list(sham_d3) + list(sham_d10)
pain = list(psl_d3) + list(psl_d10)
msFunction.msROC(sham_d10, psl_d10)
#%% SNU PSL / GBVX

mssave = np.array(list(mssave_total))

msplot = mssave[shamGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave[pslGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[ipsaline_pslGroup + ipclonidineGroup,:][:,[1,3]]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='g')


GBVX_nonpain_d3, GBVX_nonpain_d10 = [], []
GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
for SE in [164, 166, 167, 172, 174, 177, 179, 181]:
    d3c, d10c = [], []
    for se in range(12):
        d10c.append(SE in [164, 166] and se in [2,3])
        
        d3c.append(SE in [167] and se in [4,5])
        d10c.append(SE in [167] and se in [6,7])
        
        d3c.append(SE in [172] and se in [4,5])
        d10c.append(SE in [172] and se in [8,9])
        
        d3c.append(SE in [174] and se in [4,5])
        
        d3c.append(SE in [177,179,181] and se in [2,3])
        d10c.append(SE in [177,179] and se in [6,7])

        if np.sum(np.array(d3c)) > 0: GBVX_nonpain_d3.append(mssave[SE,se])
        if np.sum(np.array(d10c)) > 0: GBVX_nonpain_d10.append(mssave[SE,se])
        
    
GBVX_nonpain_d3 = msFunction.nanex(GBVX_nonpain_d3)
GBVX_nonpain_d10 = msFunction.nanex(GBVX_nonpain_d10)

msplot = np.zeros((50, 2)) * np.nan
msplot[:len(GBVX_nonpain_d3),0] = GBVX_nonpain_d3
msplot[:len(GBVX_nonpain_d10),1] = GBVX_nonpain_d10
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='k')

#%% SNU oxaliplatin

msplot = mssave[oxaliGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave[glucoseGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

#%%

# Aprism = msFunction.msarray([6])
# Aprism[0] += list(msFunction.nanex()
# Aprism[1] += list(msFunction.nanex()
# Aprism[1] += list(msFunction.nanex())
# Aprism[2] += list(msFunction.nanex())

# Aprism[3] += list(msFunction.nanex())
# Aprism[4] += list(msFunction.nanex())
# Aprism[4] += list(msFunction.nanex())
# Aprism[5] += list(msFunction.nanex())

# Aprism2 = np.zeros((len(Aprism), 100)) * np.nan
# for i in range(len(Aprism)):
#     Aprism2[i,:len(Aprism[i])] = Aprism[i]

# Aprism3 = np.zeros((len(Aprism), 3)) * np.nan
# Aprism3[:,0] = np.nanmean(Aprism2, axis=1)
# Aprism3[:,1] = scipy.stats.sem(Aprism2, axis=1, nan_policy='omit').data
# Aprism3[:,2] = np.nansum(np.isnan(Aprism2)==0, axis=1)

psl_d3 = list(mssave[pslGroup,1]) + list(mssave[ipsaline_pslGroup + ipclonidineGroup,1])
psl_d10 = list(mssave[pslGroup,2]) + list(mssave[ipsaline_pslGroup + ipclonidineGroup,3])
psl_d3_GBVX = list(GBVX_nonpain_d3)
psl_d10_GBVX = list(GBVX_nonpain_d10)
sham_d3 = list(mssave[shamGroup,1])
sham_d10 = list(mssave[shamGroup,2])


Aprism2 = pd.DataFrame(sham_d3)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3_GBVX)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(sham_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10_GBVX)), ignore_index=True, axis=1)

Aprism = msFunction.msarray([6])
Aprism[0] += list(msFunction.nanex(sham_d3))
Aprism[1] += list(msFunction.nanex(sham_d10))
Aprism[2] += list(msFunction.nanex(psl_d3_GBVX))
Aprism[3] += list(msFunction.nanex(psl_d3))
Aprism[4] += list(msFunction.nanex(psl_d10))
Aprism[5] += list(msFunction.nanex(psl_d10_GBVX))

Aprism_info = np.zeros((len(Aprism),3))
for i in range(len(Aprism)):
    Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

nonpain = list(sham_d3) + list(sham_d10)
pain = list(psl_d3) + list(psl_d10)
msFunction.msROC(sham_d10, psl_d10)
    

#%% KHU formalin

highGroup3_1 = [247, 248, 250, 251, 257, 258, 259, 262]
highGroup3_2 = [252, 253, 256, 260, 261, 265, 266, 267, 269, 272]
highGroup3_3= [249, 255, 263, 264, 268, 270, 271]

plt.figure()
plt.plot(np.nanmean(mssave[highGroup3_1,:], axis=0))

np.nanmean(mssave[highGroup3_2,2])
np.nanmean(mssave[highGroup3_3,2])

# plt.plot(np.nanmean(mssave[highGroup3_2,:], axis=0))
# plt.plot(np.nanmean(mssave[highGroup3_3,:], axis=0))

Aprism_highgroup3_1 = mssave[highGroup3_1,:]
Aprism_highgroup3_2 = mssave[highGroup3_2,:]
Aprism_highgroup3_3 = mssave[highGroup3_3,:]

#%%



plt.figure()
plt.plot(np.nanmean(mssave[PDpain,:], axis=0))
plt.plot(np.nanmean(mssave[PDnonpain,:], axis=0))

plt.plot(np.nanmean(mssave[pdmorphine,:], axis=0))


# PD evaluation
meanmatrix = np.zeros((N,5))
for i in range(0, 5):
    for SE in PDpain + PDnonpain:
        meanmatrix[SE,i] = np.nanmean(mssave[SE,i*2:(i+1)*2])
        
AA_PDpain = meanmatrix[PDpain,:]
AA_PDnonpain = meanmatrix[PDnonpain,:]

plt.plot(np.nanmean(AA_PDpain, axis=0))
plt.plot(np.nanmean(AA_PDnonpain, axis=0))




#%%

   
mssave2 =  mssave2[:,:,1]
#%%



        #%%
for ix in range(len(PDmorphine)):
    PDmorphine_matrix2[ix,:] = PDmorphine_matrix2[ix,:] / PDmorphine_matrix2[ix,0]

#%%















