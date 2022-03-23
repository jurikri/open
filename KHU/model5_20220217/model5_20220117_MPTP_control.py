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

inter_corr = msdata_load['inter_corr']

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
                nonpainc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [0])
                nonpainc.append(SE in capsaicinGroup + CFAgroup and se in [0])
                nonpainc.append(SE in salineGroup and se in [0,1,2,3,4])
                
                nonpainc.append(SE in pslGroup and se in [0])
                nonpainc.append(SE in shamGroup and se in [0,1,2])
                
                nonpainc.append(SE in [141, 142, 143] and se in [0]) # PSL + i.p. saline - group1
                nonpainc.append(SE in [144, 145, 150, 152] and se in [0,1]) # PSL + i.p. saline - group2
                nonpainc.append(SE in [146, 158] and se in [0,1]) # PSL + i.p. saline - group2
                nonpainc.append(SE in [151,153,161,162] and se in [0,1]) # PSL + i.p. clonidine
                
                # oxali
                nonpainc.append(SE in oxaliGroup and se in [0, 1])
                nonpainc.append(SE in [188, 200] and se in [4, 5])
                nonpainc.append(SE in [192, 194, 196, 202, 220] and se in [6, 7])
                
                # glucose
                nonpainc.append(SE in glucoseGroup and se in [0,1,2,3,4,5,6,7])
                
                # GB/VX
                nonpainc.append(SE in GBVX and se in [0,1])
                
                if False: # GBVXsw
                    drugc.append(SE in [164,166] and se in [2,3,4,5])
                    drugc.append(SE in [165] and se in [2,3])
                    drugc.append(SE in [167] and se in [4,5,6,7])
                    drugc.append(SE in [172] and se in [4,5,8,9])
                    drugc.append(SE in [174] and se in [4,5])
                    drugc.append(SE in [177] and se in [2,3,4,5,6,7])
                    drugc.append(SE in [179] and se in [2,3,4,5,6,7,10,11])
                    drugc.append(SE in [181] and se in [2,3,6,7])
 
            if snu_acute:
                painc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [1])
                painc.append(SE in capsaicinGroup and se in [1])
                
            if snu_chronic:
                if CFAsw:
                    painc.append(SE in CFAgroup and se in [1,2])
                
                if PSLsw:
                    painc.append(SE in pslGroup and se in [1,2])
                    painc.append(SE in [70,71,75,76,79] and se in [3,4])
                    
                    # painc.append(SE in ipsaline_pslGroup and se in [1,3])
                    painc.append(SE in [141, 142, 143] and se in [1,2]) # PSL + i.p. saline - group1
                    painc.append(SE in [144, 145, 150, 152] and se in [2,3,6,7]) # PSL + i.p. saline - group2 (PSL)
                    painc.append(SE in [144, 145, 150, 152] and se in [4,5,8,9]) # PSL + i.p. saline - group2 (PSL + saline)
                    painc.append(SE in [146, 158] and se in [2,3]) # PSL + i.p. saline - group2 (PSL)
                    painc.append(SE in [146, 158] and se in [4,5]) # PSL + i.p. saline - group2 (PSL + saline)
                    
                    painc.append(SE in [151,153,161,162] and se in [2,3,6,7]) # PSL + i.p. clonidine (PSL)
                    # painc.append(SE in [151,153,161,162]and se in [4,5,8,9]) # PSL + i.p. clonidine (PSL + clonidine)

                    painc.append(SE in [179] and se in [8,9]) # GBVX group내의 pain
                    painc.append(SE in [181] and se in [4,5]) # GBVX group내의 pain
                
                if Oxalisw:
                    # oxali
                    painc.append(SE in oxaliGroup and se in [2,3])
                    painc.append(SE in [192, 194, 196, 202, 220, 198] and se in [4,5])
                    
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
                    drugc.append(SE in PDpain and se in list(range(2,10))) 

                    drugc.append(SE in PDmorphine and se in [4,5])
                    drugc.append(SE in [325, 326] and se in [10,11])
                    drugc.append(SE in [327, 328] and se in [10,11,16,17])
                    drugc.append(SE in [329, 330] and se in [10,11,16,17])
                    drugc.append(SE in [331] and se in [10,11,16,17])
                    drugc.append(SE in [339] and se in [10,11])
                    drugc.append(SE in [340, 341] and se in [10,11])
                    
                    # i.p. saline
                    drugc.append(SE in [325, 326] and se in [12,13,14,15])
                    drugc.append(SE in [327, 328] and se in [6,7,8,9])
                    drugc.append(SE in [329, 330] and se in [6,7,8,9,18,19,20,21])
                    drugc.append(SE in [331] and se in [12,13,14,15])
                    drugc.append(SE in [339] and se in [12,13,14,15])
                    drugc.append(SE in [340, 341] and se in [6,7,8,9])
                                       
                    
            if False: # keto analgesic effects
                drugc.append(SE in KHU_CFA[:7] and se in [6,7]) # keto 100 mg/kg
                drugc.append(SE in KHU_CFA[7:] and se in [6,7]) # keto 50 mg/kg
                pass
                
            if False: # morhpine analgesic effects
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
                drugc.append(SE in [339] and se in [6,7,8,9])
                drugc.append(SE in [340, 341] and se in [12,13,14,15])
                
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

settingID = 'model5_20220217_MPTP_control'
# settingID = 'model4.1.1_20211130_1_snuonly' 

SNU_chronicpain = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup + gabapentinGroup + oxaliGroup + glucoseGroup
KHU_chronicpain = KHU_CFA + morphineGroup + KHUsham + KHU_PSL_magnolin
KHU_pdpain = KHU_CFA + morphineGroup + KHUsham 
# PDpain + PDnonpain + PDmorphine
wantedlist = PDmorphine + PDpain + PDnonpain + morphineGroup + KHUsham

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
    if SE in PSLgroup_khu + morphineGroup + KHUsham + GBVX + PDpain + PDnonpain: selist = [0,1]
    if SE in KHU_CFA + PDmorphine + KHU_PSL_magnolin: selist = [0,1,2,3]
    
    if SE in [144, 145, 150, 152] + [146, 158] + [151,153,161,162]: selist = [0,1] # PSL + i.p. saline
    if SE in glucoseGroup: selist = [0,1]
    
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
for i in range(len(X)):
    SE = Z[i][0]; se = Z[i][1]
    if np.isnan(inter_corr[SE,se]):
        import sys; sys.exit()
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

#%% total tr

for lv in range(10):
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
    
    #% overfit
    if True:
        print(len(Y_total), np.mean(Y_total, axis=0))
        hist = model.fit(X_total, Y_total, batch_size=2**11, epochs=epochs, verbose=1)
        
        if True:
            xhat = model.predict(X_total)[:,1]
            painset = np.where(Y_total[:,1]==1)[0]
            
            # 10%로 최적화
            ms_thr = 0.6
            ms_thrsave = []
            for ms_thr in np.arange(0.4,0.7,0.01):
                badlabel = np.where(np.logical_and(xhat<ms_thr, Y_total[:,1]==1))[0]
                ms_ratio = len(badlabel) / len(Y_total)
                ms_thrsave.append([ms_thr, ms_ratio])
            ms_thrsave = np.array(ms_thrsave)   
            
            mix = np.argmin(np.abs(ms_thrsave[:,1] - 0.1))
            realpain = np.where(xhat>ms_thrsave[mix,0])[0]
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
                
        if False:
            import twophoton_pain_visualization as vis
            vis.msplot_PD(mssave2[:,:,1])     
        
        savepath = RESULT_SAVE_PATH + str(lv) +'_eix.pickle'
        with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(eix, f, pickle.HIGHEST_PROTOCOL)
            print(savepath, '저장되었습니다.')
            
    #% cv 생성
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
    
    #% between subject
    if True:
        cvlist = []
        for SE in wantedlist:
            cvlist_tmp = []
            for se in range(MAXSE):
                i = SEse_find(Z=Z_vix, SE=SE, se=se)
                if len(i) > 0: cvlist_tmp = cvlist_tmp + i
            cvlist.append([cvlist_tmp, SE])
        print('len(cvlist)', len(cvlist))
    cvlist = np.array(cvlist)
   
    #% cv training
    tr_graph_save = msFunction.msarray([len(cvlist), 4])
    
    # X2 = pca.transform(X)
    model = keras_setup(lr=lr, seed=0, add_fn=pca_nc, layer_1=layer_1, \
                        batchnmr=batchnmr, dropout_rate1=dropout_rate1, l2=l2)
        
    print(model.summary())
    overwrite = False
    repeat_save = []
    for repeat in range(5): #, 100):
        ### outsample test
        print(str(lv) + '_repeat', repeat, 'data num', len(Y_vix), 'Y2 dis', np.mean(Y_vix, axis=0))
        mssave = msFunction.msarray([N,MAXSE])
        
        totallist = list(range(len(Y_vix)))
        for cv in range(0, len(cvlist)):
            telist = cvlist[cv, 0]
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
                final_weightsave = RESULT_SAVE_PATH + str(lv)+'_'+str(repeat)+'_'+ str(telist[0]) + '_final.h5'
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
    
                # outsample
                X2 = pca.transform(X)
                outlist = np.where(np.sum(Y, axis=1)==0)[0]
                yhat_out = model.predict(X2[outlist])
                for out_SE in wantedlist:
                    for n in np.where(Z[outlist][:,0]==out_SE)[0]:
                        teSE = Z[outlist][n][0]; tese = Z[outlist][n][1]
                        mssave[teSE][tese].append(yhat_out[n])
                        
                # excludedsample
                vix = np.where(np.sum(Y, axis=1)>0)[0]
                X_ex = X2[vix][eix]; 
                Z_ex = Z[vix][eix]; 
                
                for n in np.where(Z_ex[:,0]==cvlist[cv, 1])[0]:
                    teSE = Z_ex[n][0]; tese = Z_ex[n][1]
                    yhat_ex = model.predict(np.array([X_ex[n]]))
                    mssave[teSE][tese].append(yhat_ex[0])
    
        mssave2 = np.zeros((N,MAXSE,3)) * np.nan
        for row in range(N):
            for col in range(MAXSE):
                tmp = np.nanmean(mssave[row][col], axis=0)
                if not(np.isnan(np.mean(tmp))): mssave2[row, col, :] = tmp
                
        repeat_save.append(mssave2)
        
    savepath = RESULT_SAVE_PATH + str(lv) + '_repeat_save.pickle'
    with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(repeat_save, f, pickle.HIGHEST_PROTOCOL)
        print(savepath, '저장되었습니다.')

#%% vis
#% dataload
mssave = []
for lv in range(10):
    savepath = RESULT_SAVE_PATH + str(lv) + '_repeat_save.pickle'
    if os.path.isfile(savepath):
        with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
            repeat_save = pickle.load(f)
        tmp = np.nanmean(np.array(repeat_save), axis=0)
        mssave.append(tmp)       
mssave = np.array(mssave)
print(mssave.shape)
mssave = np.nanmean(mssave, axis=0)
    
import twophoton_pain_visualization as vis
Aprism1, Aprism2 = vis.msplot_PD(mssave[:,:,1])
vis.ms_report(mssave[:,:,1])















