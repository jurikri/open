# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
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

MAXSE = 100
# set pathway
try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
    gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
except:
    try:
        savepath = 'C:\\titan_savepath\\'; os.chdir(savepath);
        gsync = 'C:\\Users\\skklab\\Google 드라이브\\google_syn\\'
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['behavss2']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로

signalss = np.array(msdata_load['signalss'])

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

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]
totaldataset = grouped_total_list

msset = msGroup['msset']
msset2 = msGroup['msset2']

del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup \
+ itSalineGroup + itClonidineGroup # for test only

pslset = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup
fset = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
baseonly = lowGroup + lidocaineGroup + restrictionGroup
#%

# def msROC(class0, class1):
#     import numpy as np
#     from sklearn import metrics
    
#     pos_label = 1; roc_auc = -np.inf; fig = None

#     class0 = np.array(class0); class1 = np.array(class1)
#     class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
    
#     anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
#     predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)       
#     fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
#     maxix = np.argmax((1-fpr) * tpr)
#     specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
#     accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
#     roc_auc = metrics.auc(fpr,tpr)
    
#     return accuracy, roc_auc


# def downsampling(msssignal, wanted_size):
#     downratio = msssignal.shape[0]/wanted_size
#     downsignal = np.zeros(wanted_size)
#     downsignal[:] = np.nan
#     for frame in range(wanted_size):
#         s = int(round(frame*downratio))
#         e = int(round(frame*downratio+downratio))
#         downsignal[frame] = np.mean(msssignal[s:e])
        
#     return np.array(downsignal)

# t4 = total activity, movement

# t4 = np.zeros((N,MAXSE)); movement = np.zeros((N,MAXSE))
# for SE in range(N):
#     for se in range(len(signalss[SE])):
#         t4[SE,se] = np.mean(signalss[SE][se])
#         movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not

# movement_syn = msFunction.msarray([N])
# for SE in range(N):
#     tmp = []
#     for se in range(len(signalss[SE])):
#         tmp.append(downsampling(bahavss[SE][se], signalss[SE][se].shape[0]))
#     movement_syn[SE] = tmp

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

            snu_base = True
            snu_acute = True
            snu_chronic = False
            
            khu_base = False
            khu_acute = False
            khu_chronic = False
            
            CFAsw = False

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
                nonpainc.append(SE in list(range(192,198)) + [202, 203, 220, 221]  and se in [3])
                nonpainc.append(SE in [188, 189, 200, 201] and se in [2])
                nonpainc.append(SE in glucoseGroup[:2] and se in [0,1,2])
                nonpainc.append(SE in glucoseGroup[2:] and se in [0,1,2,3])
            
            if snu_acute:
                painc.append(SE in highGroup + midleGroup + ketoGroup + highGroup2 and se in [1])
                if False:
                    painc.append(SE in capsaicinGroup and se in [1])
                
            if snu_chronic:
                if CFAsw:
                    painc.append(SE in CFAgroup and se in [1,2])
                
                painc.append(SE in pslGroup and se in [1,2])
                painc.append(SE in ipsaline_pslGroup and se in [1,3])
                
                painc.append(SE in [179] and se in [8,9]) # GBVX group내의 pain
                painc.append(SE in [181] and se in [4,5]) # GBVX group내의 pain
                
                if True:
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
                
                mslist = [2,3,4,5,6,7,8,9] # overfit check 용
                nonpainc.append(SE in KHUsham and se in mslist)
                
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
                    painc.append(SE in KHU_CFA[:7] and se in [10])
                painc.append(SE in morphineGroup and se in mslist)
                painc.append(SE in PSLgroup_khu and se in [1,2])
                painc.append(SE in KHU_PSL_magnolin and se in [4,5,6,7])
                
                if False:
                    painc.append(SE in PDpain and se in list(range(2,10)))
                    painc.append(SE in PDmorphine and se in [4,5])
                    painc.append(SE in [325, 326] and se in [10,11,12,13,14,15])
                    painc.append(SE in [327, 328] and se in [6,7,8,9,10,11, 16,17])
                    painc.append(SE in [329, 330] and se in [6,7,8,9,10,11, 16,17,18,19,20,21])
                    painc.append(SE in [331] and se in [10,11,12,13,14,15,16,17])
                    
                
            if False: # keto analgesic effects
                # nonpainc.append(SE in KHU_CFA[:7] and se in [6,7]) # keto 100 mg/kg
                # nonpainc.append(SE in KHU_CFA[7:] and se in [6,7]) # keto 50 mg/kg
                pass
                
            if False:  # morhpine analgesic effects
                # nonpainc.append(SE in morphineGroup and se in [10,11,12]) # morphine
                # nonpainc.append(SE in KHUsham and se in range(10,13)) # morphine
                pass
            
            if [SE, se] in [[285, 4],[290, 5]]: continue # 시간짧음, movement 불일치
            
            if np.sum(np.array(painc)) > 0: group_pain_training.append([SE, se])   
            if np.sum(np.array(nonpainc)) > 0: group_nonpain_training.append([SE, se])

total_list = list(set(list(np.array(group_pain_training)[:,0]) + list(np.array(group_nonpain_training)[:,0])))
total_list = total_list + KHU_PSL_magnolin
total_list = list(set(total_list))

#%% khu t4 check
if False:
    nonpain1, nonpain2, pain = [], [], []
    
    for SE in highGroup3:
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test: nonpain1.append(t4[SE,se])
            if [SE, se] in group_pain_test: pain.append(t4[SE,se])
            
    print(np.mean(nonpain1), np.mean(nonpain2), np.mean(pain))
    accuracy, roc_auc = msROC(nonpain1 + nonpain2, pain)
    print(accuracy, roc_auc)

# t4 만으로 formalin이 나뉠수 있음.
# 하지만 nonpain mov와 t4의 corr이 있으면, mov 정도에 의해 얼마든지 오측정 될 가능성이 있음.

# mov 정리해야함, 이부분은 나중에 처리 (외장하드에 있음)

#%% data prep

# def xarray_fn(X_tr=None, fn=None):
#     X_tr2 = msFunction.msarray([fn])
#     for h in range(fn):
#         tmp4 = []
#         for n in range(len(X_tr)):
#             tmp4.append(np.array(X_tr[n,h], dtype=float))
#         X_tr2[h] = np.array(tmp4)
#     return X_tr2

# def XYZgen(SE=None, se=None, msclass=None, testsw2=False):
#     X_tmp=[]; Y_tmp=[]; Z_tmp=[]; # ROI=None
#     msbins = np.arange(0, signalss[SE][se].shape[0]-FS+1, BINS) 
    
# #    if testsw2: 
# #    engram = list(range(signalss[SE][se].shape[1]))
    
# #    if roisw:
#     if testsw2:
#         for ROI in range(signalss[SE][se].shape[1]):
#             mssignal = np.array(signalss[SE][se][:,ROI])
#             for u in msbins:
#                     # feature 1
#                 ft1 = np.array(mssignal[u:u+FS])
#                 ft1 = np.reshape(ft1, (ft1.shape[0], 1))  
               
#                 X_tmp.append(ft1);
#                 Y_tmp.append(msclass); 
#                 Z_tmp.append([SE, se]) #; T_tmp += t4_save
#     else:
#         for u in msbins:
#             ft1 = np.mean(np.array(signalss[SE][se][u:u+FS,:]), axis=1)
#             ft1 = np.reshape(ft1, (ft1.shape[0], 1))  
           
#             X_tmp.append(ft1);
#             Y_tmp.append(msclass); 
#             Z_tmp.append([SE, se]) #; T_tmp += t4_save 
          
#     return X_tmp, Y_tmp, Z_tmp

# #Xsave, Ysave, Zsave = ms_sampling(forlist=[70])
# # X_tmp = X_tr; Y_tmp = Y_tr; Z_tmp = Z_tr
# def upsampling(X_tmp, Y_tmp, Z_tmp, offsw=False):
#     dup = 10
#     X = np.array(X_tmp)
#     Y = np.array(Y_tmp)
#     Z = np.array(Z_tmp)
#     if not(offsw):
#         while True:
#             n_ix = np.where(Y[:,0]==1)[0]
#             p_ix = np.where(Y[:,1]==1)[0]
#             print('sample distributions', 'nonpain', n_ix.shape[0], 'pain', p_ix.shape[0])
            
#             nnum = n_ix.shape[0]
#             pnum = p_ix.shape[0]
            
#             maxnum = np.max([nnum, pnum])
#             minnum = np.min([nnum, pnum])
            
#             print('ratio', maxnum / minnum)
#             if maxnum / minnum > dup:
#                 print('1/' + str(dup) + ' down')
#                 downix = np.where(Y[:,np.argmax([nnum, pnum])]==1)[0]
#                 rix = random.sample(list(downix), int(len(downix)/dup*(dup-1)))
                
#                 tlist = list(range(len(Y)))
#                 vlist = list(set(tlist)-set(rix))
                
#                 X = X[vlist]
#                 Y = Y[vlist]
#                 Z = Z[vlist]
#                 continue

#             addix = np.where(Y[:,np.argmin([nnum, pnum])]==1)[0]
#             if not(maxnum == minnum):
#                 if maxnum // minnum > 1:
#                     X = np.append(X, X[addix], axis=0)
#                     Y = np.append(Y, Y[addix], axis=0)
#                     Z = np.append(Z, Z[addix], axis=0)
#                 elif maxnum // minnum == 1:
#                     rix = random.sample(list(addix), maxnum-minnum)
#                     X = np.append(X, X[rix], axis=0)
#                     Y = np.append(Y, Y[rix], axis=0)
#                     Z = np.append(Z, Z[rix], axis=0)
#             else: break
#             print('data set num #', len(Y), np.mean(np.array(Y), axis=0))
    
#     # shuffle
#     X = np.array(X)
#     Y = np.array(Y)
#     Z = np.array(Z)
    
#     six = random.sample(list(range(len(X))), len(X))
    
#     X = X[six]
#     Y = Y[six]
#     Z = Z[six]

#     return X, Y, Z

def upsampling(X_tmp, Y_tmp, Z_tmp, verbose=0):
    X = np.array(X_tmp)
    Y = np.array(Y_tmp)
    Z = np.array(Z_tmp)
    while True:
        n_ix = np.where(Y[:,0]==1)[0]
        p_ix = np.where(Y[:,1]==1)[0]
        if verbose: print('sample distributions', 'nonpain', n_ix.shape[0], 'pain', p_ix.shape[0])
        
        nnum = n_ix.shape[0]
        pnum = p_ix.shape[0]
        
        maxnum = np.max([nnum, pnum])
        minnum = np.min([nnum, pnum])
        
        if verbose: print('ratio', maxnum / minnum)
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
        if verbose:  print('data set num #', len(Y), np.mean(np.array(Y), axis=0))
    return X, Y, Z


#%% XYZgen

X, Y, Z = [], [], [];
Xte, Zte = [], [];
# ROI = 0 # dummy
# testsw = False
# if seset is None: sefor = range(10)
# if not seset is None: sefor = seset; testsw = True
# SE = forlist[0]; se = sefor[0]
for SE in tqdm(range(230)):
    for se in range(len(signalss[SE])):
        msclass = [0, 0]
        if [SE, se] in group_pain_training: msclass = [0, 1] # for pain
        if [SE, se] in group_nonpain_training: msclass = [1, 0]  # for nonpain

        if not(msclass is None):
            # for training
            msbins = np.arange(0, signalss[SE][se].shape[0]-FS+1, BINS, dtype=int) 
            mssignal = np.mean(np.array(signalss[SE][se]), axis=1)
            for u in msbins:
                ft1 = np.array(mssignal[u:u+FS])
                ft1 = np.reshape(ft1, (ft1.shape[0], 1))  
               
                X.append(ft1);
                Y.append(msclass); 
                Z.append([SE, se, u, None])
                
            # for test
            msbins = np.arange(0, signalss[SE][se].shape[0]-FS+1, BINS, dtype=int) 
            for ROI in range(signalss[SE][se].shape[1]):
                mssignal = np.array(signalss[SE][se])[:,ROI]
                for u in msbins:
                    ft1 = np.array(mssignal[u:u+FS])
                    ft1 = np.reshape(ft1, (ft1.shape[0], 1))  
                   
                    Xte.append(ft1);
                    Zte.append([SE, se, u, ROI])
             
X, Y, Z = np.array(X), np.array(Y), np.array(Z)
Xte, Zte = np.array(Xte), np.array(Zte)

print('len(X)', len(X), '// np.mean(Y, axis=0)', np.mean(Y, axis=0))
#%% hyperparameter

# mslength = np.zeros((N,MAXSE)) * np.nan
# for SE in range(N):
#     for se in range(len(signalss[SE])):
#         signal = np.array(signalss[SE][se])
#         mslength[SE,se] = signal.shape[0]
        
FS = 497
print('full_sequence', FS, 'frames')

BINS = 10 # 최소 time frame 간격 # hyper

# learning intensity
epochs = 1 # 
lr = 1e-3 # learning rate

n_hidden = int(48) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(48) # fully conneted laye node 갯수 # 8 # 원래 6 

l2_rate = 0.3
dropout_rate1 = 0.2 # dropout rate
dropout_rate2 = 0.1 # 


#%% keras setup
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

def keras_setup(lr=1e-3, xin=None, seed=0, l2_rate=0, layer_1=10, n_hidden=10, dropout_rate1=0, dropout_rate2=0):
    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
    
    input1 = tf.keras.layers.Input(shape=(xin.shape)) # 각 병렬 layer shape에 따라 input 받음
    input2 = Bidirectional(LSTM(n_hidden))(input1) # biRNN -> 시계열에서 단일 value로 나감
    input2 = Dense(layer_1, kernel_initializer = init, activation='relu')(input2) # fully conneted layers, relu
    input2 = Dropout(dropout_rate1)(input2) # dropout
    
    added = input2

    merge_1 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(added) # fully conneted layers, relu
    merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
    merge_2 = Dense(2, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
    # merge_3 = Dense(2, input_dim=2)(merge_2) # regularization 삭제
    # merge_4 = Activation('softmax')(merge_3) # activation as softmax function
    
    merge_4 = Dense(2, kernel_initializer = init, activation='softmax')(merge_2)
    
    model = tf.keras.models.Model(inputs=input1, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup(lr=lr, seed=0, xin=np.zeros((100,1)))
print(model.summary())

#%% pathset

# q = project_list[0]; nix = 0
# # engram_save_teman = []
# for nix, q in enumerate(project_list):
#     settingID = q[0]; seed = q[1]  # ; seed2 = int(seed+1)
#     # continueSW = q[2]
    
#     print('settingID', settingID, 'seed', seed)

#     # set the pathway2
#     RESULT_SAVE_PATH =  gsync + 'kerasdata\\'
#     if not os.path.exists(RESULT_SAVE_PATH):
#         os.mkdir(RESULT_SAVE_PATH)

#     RESULT_SAVE_PATH = gsync + 'kerasdata\\' + settingID + '\\'
#     if not os.path.exists(RESULT_SAVE_PATH):
#         os.mkdir(RESULT_SAVE_PATH)
        
# ### wantedlist
#     runlist = highGroup3 + PSLgroup_khu + pslGroup + PSLgroup_khu
#     validlist =  PSLgroup_khu # highGroup3 # [PSLgroup_khu[2]] # [highGroup3[0]];
    
    
    
settingID = 'model2_0'


fset = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
# runlist = highGroup3 + PSLgroup_khu + pslGroup + PSLgroup_khu
wantedlist =  fset # highGroup3 # [PSLgroup_khu[2]] # [highGroup3[0]];


RESULT_SAVE_PATH = gsync + 'kerasdata\\' + settingID + '\\'
if os.path.isdir('K:\\mscode_m2'): RESULT_SAVE_PATH = 'K:\\mscode_m2\\220220102\\' + settingID + '\\'
if not os.path.exists(RESULT_SAVE_PATH): os.mkdir(RESULT_SAVE_PATH)

#%% learning

vix = np.where(np.sum(Y, axis=1)>0)[0]
X_vix = np.array(X[vix]); Y_vix = np.array(Y[vix]); Z_vix = np.array(Z[vix])

X2, Y2, Z2 = upsampling(X_vix, Y_vix, Z_vix)  
print('len(X2)', len(X2), '// np.mean(Y2, axis=0)', np.mean(Y2, axis=0))

rix = list(range(len(Y2)))
random.seed(0); random.shuffle(rix)
X2, Y2, Z2 = X2[rix], Y2[rix], Z2[rix]

model = keras_setup(lr=1e-3, xin=X2[0], seed=0, \
            l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)

hist = model.fit(X2, Y2, batch_size=2**9, epochs=200, verbose=1)


#%% cvtraining

model = keras_setup(lr=1e-3, xin=X_tr[0], seed=0, \
            l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
    
cvlist = highGroup + midleGroup + ketoGroup + highGroup2  
total = list(range(len(Y_tr)))

repeat = 0; cv = 0

mssave = msFunction.msarray([N,MAXSE])
start = time.time()   
for cv in range(len(cvlist)):
    print("time :", time.time() - start); seed=0; cnt=0; acc=0
    cvSE = cvlist[cv]
    
    telist = np.where(Z2[:,0]==cvSE)[0]
    trlist = list(set(total)-set(telist))
    
    X_tr = X2[trlist]; #X_te = X2[telist]
    Y_tr = Y2[trlist]; #Y_te = Y2[telist]
    Z_tr = Z2[trlist]; #Z_te = Z2[telist]
    
    telist_te = np.where(Zte[:,0]==cvSE)[0]; X_te = Xte[telist_te]
    
    final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_' + str([cvSE]) + '_final.h5'
    if not(os.path.isfile(final_weightsave)) or overwrite:
        print(repeat, 'learning', cv, '/', len(cvlist))
        print(len(total), len(telist), len(trlist))
        print('tr distribution', np.mean(Y_tr, axis=0), np.sum(Y_tr, axis=0))
        print('te distribution', np.mean(Y_te, axis=0))
        
        model = keras_setup(lr=1e-3, xin=X_tr[0], seed=seed, \
                    l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
        
        for epoch in range(1000000):
            cnt+=1
            if (cnt > 3000) or (acc < 0.70 and cnt > 300) or (acc < 0.51 and cnt > 50):
                seed += 1
                model = keras_setup(lr=1e-3, xin=X_tr[0], seed=seed, \
                            l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
                cnt, acc = 0, 0
                
            hist = model.fit(X_tr, Y_tr, batch_size=2**9, epochs=1, verbose=1)
            acc = list(np.array(hist.history['accuracy']))[-1]
 
            if acc > 0.91: model.save_weights(final_weightsave); break
    
    # test
    model.load_weights(final_weightsave)
    yhat = model.predict(X_te)
    for n in range(len(yhat)):
        teSE = Z_te[n][0]; tese = Z_te[n][1]
        mssave[teSE][tese].append(yhat[:,1][n])

    start = time.time()

#%% general model

final_weightsave = RESULT_SAVE_PATH + str(repeat) + '_' + 'general' + '_final.h5'
if not(os.path.isfile(final_weightsave)) or overwrite:
    model = keras_setup(lr=1e-3, xin=X2[0], seed=seed, \
                l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
    
    for epoch in range(1000000):
        cnt+=1
        if (cnt > 3000) or (acc < 0.70 and cnt > 300) or (acc < 0.51 and cnt > 50):
            seed += 1
            model = keras_setup(lr=1e-3, xin=X2[0], seed=seed, \
                        l2_rate=l2_rate, layer_1=layer_1, n_hidden=n_hidden, dropout_rate1=dropout_rate1, dropout_rate2=dropout_rate2)
            cnt, acc = 0, 0
            
        hist = model.fit(X2, Y2, batch_size=2**9, epochs=1, verbose=1)
        acc = list(np.array(hist.history['accuracy']))[-1]

        if acc > 0.91: model.save_weights(final_weightsave); break

# test
teslit = []
for cv in range(len(cvlist)):
    telist = list(telist) + list(np.where(Zte[:,0]==cvlist[cv])[0])

total = list(range(len(Xte)))
trlist = list(set(total)-set(telist))
print(len(total), len(telist), len(trlist))

X_test = Xte[trlist]; Z_test = Zte[trlist]

model.load_weights(final_weightsave)
print('start testing')
yhat = model.predict(X_test)
for n in range(len(yhat)):
    teSE = Z_test[n][0]; tese = Z_test[n][1]
    mssave[teSE][tese].append(yhat[:,1][n])


#%%
import sys; sys.exit()
    
upsampling(X_tmp, Y_tmp, Z_tmp, verbose=0)
 
    mslog = msFunction.msarray([N]); k=0
    model = keras_setup(lr=lr, seed=seed)
    initial_weightsave = RESULT_SAVE_PATH + 'initial_weight.h5'
    model.save_weights(initial_weightsave)
    
    savepath_pickle = RESULT_SAVE_PATH + 'resultsave.pickle'
    if os.path.isfile(savepath_pickle) and True:
        with open(savepath_pickle, 'rb') as f:  # Python 3: open(..., 'rb')
            mssave = pickle.load(f)
            
    elif not(os.path.isfile(savepath_pickle)):
        mssave = np.zeros((N, MAXSE)) * np.nan 

    for k in range(len(validlist)):
        stopsw = False
        vlist = validlist[k]
        if not(type(vlist)==list): vlist = [vlist]
        addset = []
        for w in [validlist[k]]:
            if not(stopsw):
                if validlist[k] in msset_total[:,0]: 
                    addset += list(msset_total[np.where(msset_total[:,0]==validlist[k])[0],:][0][1:])
                if validlist[k] in msset_total[:,1:].flatten(): stopsw = True
                
        if not(stopsw): 
            final_weightsave = RESULT_SAVE_PATH + str(validlist[k]) + '_final.h5'
            if not(os.path.isfile(final_weightsave)) or True:
                vlist += addset
                print('learning 시작합니다. validation mouse #', validlist[k])
            
                trlist = list(set(runlist) - set(vlist))

                # training set
                X_tr, Y_tr, Z_tr = ms_sampling(forlist = trlist)
                print('tr set num #', len(Y_tr), np.mean(np.array(Y_tr), axis=0))
                X_tr, Y_tr, Z_tr = upsampling(X_tr, Y_tr, Z_tr, offsw=False) # ratio 10 초과시 random down -> 1:1로 upsample, -> shuffle
                print('trainingset bias', np.mean(Y_tr, axis=0))
                
                # validation set
                X_te, Y_te, Z_te = ms_sampling(forlist = vlist, seset=[0,1])
                print('tr set num #', len(Y_te), np.mean(np.array(Y_te), axis=0))
                
                # model reset
                model = keras_setup(lr=lr, seed=seed)
                model.load_weights(initial_weightsave)
                
                hist = model.fit(X_tr, Y_tr, batch_size=2**9, epochs=200, verbose=1, validation_data = (X_te, Y_te))
                
                s_loss=[]; s_acc=[]; sval_loss=[]; sval_acc=[];
                s_loss += list(np.array(hist.history['loss']))
                s_acc += list(np.array(hist.history['accuracy']))
                sval_loss += list(np.array(hist.history['val_loss']))
                sval_acc += list(np.array(hist.history['val_accuracy']))
      
                # save
                mssave_tmp = {'s_loss': s_loss, 's_acc': s_acc, \
                              'sval_loss': sval_loss, 'sval_acc': sval_acc}

                model.save_weights(final_weightsave)
                
                savepath_log = RESULT_SAVE_PATH + str(k) + '_log.pickle'
                with open(savepath_log, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(mssave_tmp, f, pickle.HIGHEST_PROTOCOL)
                    print(savepath_log, '저장되었습니다.')
                
                for fignum in range(2):
                    plt.figure()
                    if fignum == 0:
                        figname = str(k) + '_loss_save.png'
                        plt.plot(s_loss)
                        plt.plot(sval_loss)
                    if fignum == 1:
                        figname = str(k) + '_acc_save.png'
                        plt.plot(s_acc)
                        plt.plot(sval_acc)
                    plt.savefig(RESULT_SAVE_PATH + figname)
                    plt.close()
                
                # test
                # valSE = vlist[0]
                for valSE in vlist:
                    for valse in range(len(signalss[valSE])):
                        X_te, Y_te, Z_te = ms_sampling(forlist = [valSE], seset=[valse], ROIsw=True)
                        predict = model.predict(X_te)
                        pain = np.mean(predict[:,1])
                        print()
                        print('te set num #', len(Y_te), 'test result SE', valSE, 'se', valse, 'pain >>', pain)
                        mssave[valSE, valse] = pain

    with open(savepath_pickle, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)
        print(savepath_pickle, '저장되었습니다.')


#%% prism 복붙용 변수생성

pain_time = msFunction.msarray([MAXSE])
nonpain_time = msFunction.msarray([MAXSE])

target = np.array(mssave)
for row in range(len(target)):
    target[row,:] = target[row,:] - target[row,0]

nonpain1, nonpain2, pain = [], [], []
for SE in range(N):
    if SE in PSLgroup_khu: # filter
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test:
                nonpain_time[se].append(target[SE,se])
            if [SE, se] in group_pain_test:
                pain_time[se].append(target[SE,se])

def to_prism(target):
    Aprism = pd.DataFrame([])
    for row in range(len(target)):
        Aprism = pd.concat((Aprism, pd.DataFrame(target[row])), ignore_index=True, axis=1)
    return Aprism

Aprism_nonpain = to_prism(nonpain_time)
Aprism_pain = to_prism(pain_time)

# ROC 판정용 - 직렬화
def to_linear(target):
    linear = []
    for row in range(len(target)):
        linear += target[row]
    return linear

nonpain = to_linear(nonpain_time)
pain = to_linear(pain_time)

print(np.mean(nonpain), np.mean(pain))
accuracy, roc_auc = msROC(nonpain, pain)
print(accuracy, roc_auc)

#%% PD 분석 - 후처리
PATH = 'D:\\mscore\\syncbackup\\Project\\박하늬선생님_PD_painimaging\\raw\\'
pickle_save_tmp = PATH + 'mspickle_PD.pickle'    

with open(pickle_save_tmp, 'rb') as f:  # Python 3: open(..., 'rb')
    signalss_PD = pickle.load(f)
    
mssave_PD = np.zeros((len(signalss_PD), MAXSE)) * np.nan 

for valSE in range(len(signalss_PD)):
    for valse in range(len(signalss_PD[SE])):
        X_te, Y_te, Z_te =  ms_sampling(forlist=[valSE], seset=[valse], signalss=signalss_PD, ROIsw=True, fixlabel=True)
        predict = model.predict(X_te)
        pain = np.mean(predict[:,1])
        print()
        print('te set num #', len(Y_te), 'test result SE', valSE, 'se', valse, 'pain >>', pain)
        mssave_PD[valSE, valse] = pain











