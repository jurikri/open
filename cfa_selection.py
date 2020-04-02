# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""
import os  
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import random
import time

from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from numpy.random import seed as nseed
import tensorflow as tf
from keras.layers import BatchNormalization


# set pathway
try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
     
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열

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
 
msset = msGroup['msset']
msset2 = msGroup['msset2']
del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup \
+ itSalineGroup + itClonidineGroup # for test only

pslset = pslGroup + shamGroup + adenosineGroup + itSalineGroup + itClonidineGroup
# In[]

def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

# t4 = total activity, movement 
t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
        # 개별 thr로 relu 적용되어있음. frame은 signal과 syn가 다름

##
# 절대값으로 resizing 하면안됨. session 마다 size가 다름을 고려해야함. 수정요망 . 
        # 현재 사용하지 않으므로, 나중으로 미루겠음.. 
        # 수정.. 되있음? 되있는듯
movement_syn = []
[movement_syn.append([]) for u in range(N)]

for SE in range(N):
    [movement_syn[SE].append([]) for u in range(5)]
    for se in range(5):
        movement_syn[SE][se] = downsampling(bahavss[SE][se], signalss[SE][se].shape[0])
 
##
       
grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

bins = 10 # 최소 time frame 간격

totaldataset = grouped_total_list
                  
def array_recover(X_like):
    X_like_toarray = []; X_like = np.array(X_like)
    for input_dim in range(msunit *fn):
        tmp = np.zeros((X_like.shape[0],X_like[0,input_dim].shape[0]))
        for row in range(X_like.shape[0]):
            tmp[row,:] = X_like[row,input_dim]
    
        X_like_toarray.append(tmp)
        
        X_like_toarray[input_dim] =  \
        np.reshape(X_like_toarray[input_dim], (X_like_toarray[input_dim].shape[0],X_like_toarray[input_dim].shape[1],1))
    
    return X_like_toarray

# data 생성
SE = 70; se = 1; label = 1; roiNum=None; GAN=False; Mannual=False; mannual_signal=None; passframesave=np.array([])
def dataGeneration(SE, se, label, roiNum=None, bins=bins, GAN=False, Mannual=False, \
                   mannual_signal=None, mannual_signal2=None, passframesave=np.array([])):    
    X = []; Y = []; Z = []
    if label == 0:
        label = [1, 0] # nonpain
    elif label == 1:
        label = [0, 1] # pain
#    elif label == 2:
#        label = [0, 0] # nonpain low
 
    if not(roiNum==None):
        s = roiNum; e = roiNum+1
    elif roiNum==None:
        s = 0; e = signalss[SE][se].shape[1]
    
    if Mannual:
        signal_full = mannual_signal
        mannual_signal2 = mannual_signal2
        
    signal1 = np.mean(signal_full[:,s:e], axis=1) # 단일 ROI만 선택하는 것임
    
    lastsave = np.zeros(msunit, dtype=int)    
    binlist = list(range(0, full_sequence-np.min(sequenceSize), bins))
    
    if len(binlist) == 0:
        binlist = [0]

    if passframesave.shape[0] != 0:
        binlist = passframesave

    t4_save = []
    for frame in binlist:   
        X_tmp = []; [X_tmp.append([]) for k in range(msunit * fn)] 

        for unit in range(msunit):
            if frame <= full_sequence - sequenceSize[unit]:
                X_tmp[unit] = (signal1[frame : frame + sequenceSize[unit]])
                lastsave[unit] = frame
                
                if unit == 0:
                    t4_save.append(np.mean(signal1[frame : frame + sequenceSize[unit]]))
                
            else:
                X_tmp[unit] = (signal1[lastsave[unit] : lastsave[unit] + sequenceSize[unit]])
                if unit == 0:
                    t4_save.append(np.mean(signal1[lastsave[unit] : lastsave[unit] + sequenceSize[unit]]))

        X.append(X_tmp)
        Y.append(label)
        Z.append([SE,se])

    return X, Y, Z

# reset..?
from keras.backend.tensorflow_backend import clear_session
import tensorflow.python.keras.backend as K

def reset_keras(classifier):
    sess = K.get_session()
    clear_session()
    sess.close()
    sess = K.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

# 최소길이 찾기
mslength = np.zeros((N,5)); mslength[:] = np.nan
for SE in range(N):
    if SE in totaldataset:
        for se in range(5):
            signal = np.array(signalss[SE][se])
            mslength[SE,se] = signal.shape[0]

full_sequence = int(np.nanmin(mslength))
#full_sequence = int(round(FPS*60)) # 20200115 test용, 최소 크기를 1분으로 고정
print('full_sequence', full_sequence, 'frames')

#signalss_cut = preprocessing(endpoint=int(full_sequence))

msunit = 1 # input으로 들어갈 시계열 길이 및 갯수를 정함. full_sequence기준으로 1/n, 2/n ... n/n , n/n

sequenceSize = np.zeros(msunit) # 각 시계열 길이들을 array에 저장
for i in range(msunit):
    sequenceSize[i] = int(full_sequence/msunit*(i+1))
sequenceSize = sequenceSize.astype(np.int)

print('full_sequence', full_sequence)
print('sequenceSize', sequenceSize)

  
###############
# hyperparameters #############
 
# learning intensity
epochs = 1 # epoch 종료를 결정할 최소 단위.
lr = 5e-4 # learning rate
fn = 1

n_hidden = int(8 * 3) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 3) # fully conneted laye node 갯수 # 8 # 원래 6 
# 6 for normal
# 10 for +cfa

#duplicatedNum = 1
#mspainThr = 0.27
#acitivityThr = 0.4
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regulariza3 # regularization 상수
l2_rate = 0.0
dropout_rate1 = 0.05 # dropout late
dropout_rate2 = 0.05 # 

#testsw = False  # test 하지 않고 model만 저장함. # cloud 사용량을 줄이기 위한 전략.. 
trainingsw = True # training 하려면 True 
statelist = ['exp'] # ['exp', 'con']  # random shuffled control 사용 유무
validation_sw = True # 시각화목적으로만 test set을 validset으로 배치함.
testsw2 = False
testsw3 = True
#if testsw2:
##    import os
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = ''
#    import tensorflow as tf

# 집 컴퓨터, test 전용으로 수정

acc_thr = 0.91 # 0.93 -> 0.94
batch_size = 2**10 # 5000
###############

# constant 
maxepoch = 3000
n_in =  1 # number of features
n_out = 2 # number of class # 20191104: 3 class로 시도
classratio = 1 # class under sampling ratio

project_list = []
 # proejct name, seed
#
#project_list.append(['control_test_segment_adenosine_set1', 100, None])
#project_list.append(['control_test_segment_adenosine_set2', 200, None])
#project_list.append(['control_test_segment_adenosine_set3', 300, None])
#project_list.append(['control_test_segment_adenosine_set4', 400, None])
#project_list.append(['control_test_segment_adenosine_set5', 500, None])
# 
#project_list.append(['0330_batchnorm_1', 100, None])
#project_list.append(['0330_batchnorm_2', 200, None])
#project_list.append(['0330_batchnorm_3', 300, None])
 
project_list.append(['0331_CFA_selection', 100, None])

q = project_list[0]
for nix, q in enumerate(project_list):

    print(nix, l2_rate)
    
    settingID = q[0]; seed = q[1]; seed2 = int(seed+1)
    continueSW = q[2]
    
    print('settingID', settingID, 'seed', seed, 'continueSW', continueSW)

    # set the pathway2
    RESULT_SAVE_PATH = './result/'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)

    RESULT_SAVE_PATH = './result/' + settingID + '//'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)
    
    if not os.path.exists(RESULT_SAVE_PATH + 'exp/'):
        os.mkdir(RESULT_SAVE_PATH + 'exp/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'exp_raw/'):
        os.mkdir(RESULT_SAVE_PATH + 'exp_raw/') 
    
    if not os.path.exists(RESULT_SAVE_PATH + 'control/'):
        os.mkdir(RESULT_SAVE_PATH + 'control/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'control_raw/'):
        os.mkdir(RESULT_SAVE_PATH + 'control_raw/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'model/'):
        os.mkdir(RESULT_SAVE_PATH + 'model/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'tmp/'):
        os.mkdir(RESULT_SAVE_PATH + 'tmp/')

    testset = []
    trainingset = list(totaldataset)
    for u in testset:
        try:
            trainingset.remove(u)
        except:
            pass
# In[]    
    # initiate
    
    set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup + highGroup2
    set1 = lowGroup + lidocaineGroup + restrictionGroup + salineGroup
    set3 = pslGroup + adenosineGroup + shamGroup + CFAgroup + chloroquineGroup + itSalineGroup + itClonidineGroup
    for msdel in msset_total[:,1]:
        set3.remove(msdel)
    
    reducing_test_list = []; reducing_ratio = 1
    random.seed(seed)
    reducing_test_list += random.sample(set1, int(round(len(set1)*reducing_ratio)))
    random.seed(seed)
    reducing_test_list += random.sample(set2, int(round(len(set2)*reducing_ratio)))
    random.seed(seed)
    reducing_test_list += random.sample(set3, int(round(len(set3)*reducing_ratio)))

    for msadd in msset_total[:,0]:
          if msadd in reducing_test_list:
              tmp = msset_total[[np.where(msset_total[:,0] == msadd)][0][0],1]
              reducing_test_list += list(tmp)
    print('selected mouse #', len(reducing_test_list))          
#    print(reducing_test_list)
    def ms_sampling(forlist=range(N), cfa_set=None):
        sampleNum = []; [sampleNum.append([]) for u in range(n_out)]
        
        datasetX = []; datasetY = []; datasetZ = []
        for classnum in range(n_out):
            datasetX.append([]); datasetY.append([]); datasetZ.append([])
            
        # nonpain     
        msclass = 0 # nonpain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in forlist:
            if SE in trainingset:
                if SE in reducing_test_list:
                    for se in range(5):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        set1 = highGroup + midleGroup + lowGroup + yohimbineGroup + ketoGroup + lidocaineGroup + restrictionGroup + highGroup2 
                        c1 = SE in set1 and se in [0,2]
                        c2 = SE in capsaicinGroup and se in [0,2]
#                        c3 = SE in pslGroup + adenosineGroup and se in [0]
#                        c4 = SE in shamGroup and se in [0,1,2]
#                        c5 = SE in salineGroup and se in [0,1,2,3,4]
                        c6 = SE in CFAgroup and se in [0]
#                        c7 = SE in chloroquineGroup and se in [0]
#                        c8 = SE in itSalineGroup and se in [0]
#                        c9 = SE in itClonidineGroup and se in [0]
  
#                        c13 = SE in chloroquineGroup and se in [1]
                                        
#                        if c1 or c2 or c3 or c4 or c5 or c6 or c7:
                        if c1 or c2 or c6:
#                        if c13: #
                            # msset 만 baseline을 제외시킴, total set 아님 
                            exceptbaseline = (SE in np.array(msset)[:,1:].flatten()) and se == 0 
                            if not exceptbaseline: # baseline을 공유하므로, 사용하지 않는다. 
                                mssignal = np.mean(signalss[SE][se], axis=1)
#                                mssignal2 = np.array(movement_syn[SE][se])
                                msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                                
                                for u in msbins:
                                    if type(cfa_set) != 'NoneType' and SE in CFAgroup:
                                        if not [SE, se, u] in cfa_set:
                                            continue
                                    
                                    mannual_signal = mssignal[u:u+full_sequence]
                                    mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                    
#                                    mannual_signal2 = mssignal2[u:u+full_sequence]
#                                    mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
    
                                    X, Y, Z = dataGeneration(SE, se, label=msclass, \
                                                   Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                                    
                                    X_tmp += X; Y_tmp += Y; Z_tmp += [Z[0] + [u]]#; T_tmp += t4_save 
                    
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
        sampleNum[msclass] = len(datasetX[msclass])
        print('nonpain_sampleNum', sampleNum[msclass])
        
        msclass = 1 # pain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in forlist:
            if SE in trainingset:
                if SE in reducing_test_list:
                    for se in range(5):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup + highGroup2
                        c11 = SE in set2 and se in [1]
                        c12 = SE in CFAgroup and se in [1,2]
#                        c13 = SE in chloroquineGroup and se in [1]
                          
                        if c11 or c12: # 
                            if not(0.15 < movement[SE,se]):
                                print(SE, se, 'movement 부족, pain session에서 제외.')
                                continue
                        
                            mssignal = np.mean(signalss[SE][se], axis=1)
#                            mssignal2 = np.array(movement_syn[SE][se])
                            msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                            
                            for u in msbins:
                                if type(cfa_set) != 'NoneType' and SE in CFAgroup:
                                    if not [SE, se, u] in cfa_set:
                                        continue

                                mannual_signal = mssignal[u:u+full_sequence]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                
#                                mannual_signal2 = mssignal2[u:u+full_sequence]
#                                mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                                
                                X, Y, Z = dataGeneration(SE, se, label=msclass, \
                                               Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                                X_tmp += X; Y_tmp += Y; Z_tmp += [Z[0] + [u]] #; T_tmp += t4_save 
                                
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp                        
        sampleNum[msclass] = len(datasetX[msclass])
        print('pain_sampleNum', sampleNum[msclass])          
       
        return datasetX, datasetY, datasetZ

    def upsampling(X_elite, Y_elite, Z_elite):
        
        X = np.array(X_elite)
        Y = np.array(Y_elite)
        Z = np.array(Z_elite)
    
        nonpain_ix = np.where(Y[:,0]==1)[0]
        pain_ix = np.where(Y[:,1]==1)[0]
        print('sample distributions', 'ix 재확인', nonpain_ix.shape[0], pain_ix.shape[0])
        
        data0 = np.array(Y)[nonpain_ix]
        data1 = np.array(Y)[pain_ix]
        Ylist_class_save = [nonpain_ix, pain_ix]
        
        big = np.max([data0.shape[0], data1.shape[0]])
        small = np.min([data0.shape[0], data1.shape[0]])
        bigix = np.argmax([data0.shape[0], data1.shape[0]])
        samllix = np.argmin([data0.shape[0], data1.shape[0]])
        
        mul = big // small
        mod = big % small
        
        Ylist_class_big = np.array(Ylist_class_save[bigix])
        Ylist_class_small = np.array(Ylist_class_save[samllix])
        
        rix = list(range(Ylist_class_small.shape[0]))
        random.seed(seed); rix2 = random.sample(rix, mod)
        
        # 큰거 먼저 넣고, 작은거 반복해서 append
        trX=[]; [trX.append([]) for u in range(len(X))]
        for u in range(len(X)):
            trX[u] = np.array(X)[u][Ylist_class_big]
        trY = np.array(Y)[Ylist_class_big]
        trZ = np.array(Z)[Ylist_class_big]
    #    trZ = np.array(ml_dataset[2])[cvrix][badix]
        print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])
        
        for m in range(mul): 
            for u in range(len(X)):
                trX[u] = np.append(trX[u], np.array(X[u])[Ylist_class_small], axis=0)
            trY = np.append(trY, np.array(Y)[Ylist_class_small], axis=0)
            trZ = np.append(trZ, np.array(Z)[Ylist_class_small], axis=0)
            print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])
            
        for u in range(len(X)):
            trX[u] = np.append(trX[u], np.array(X[u])[Ylist_class_small][rix2], axis=0)
        trY = np.append(trY, np.array(Y)[Ylist_class_small][rix2], axis=0)
        trZ = np.append(trZ, np.array(Z)[Ylist_class_small][rix2], axis=0)
        
        # random shuffle
        trX2=[]; [trX2.append([]) for u in range(len(X))]
        rix3 = list(range(len(trY)))
        random.seed(seed); random.shuffle(rix3)
        for u in range(len(X)):
    #        print(np.array(trX[u]).shape)
            trX2[u] = np.array(trX[u])[rix3]
    #        print(np.array(trX[u])[rix3].shape)
        trY = np.array(trY)[rix3]
        trZ = np.array(trZ)[rix3]

        print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])
        return trX2, trY, trZ
    # In[]
    X_save2, Y_save2, Z_save2 = ms_sampling(forlist=CFAgroup)
    
    X = np.array(X_save2[0]); Y = np.array(Y_save2[0]); Z = np.array(Z_save2[0])
    for i in range(1,n_out):
        X = np.concatenate((X,X_save2[i]), axis = 0)
        Y = np.concatenate((Y,Y_save2[i]), axis = 0)
        Z = np.concatenate((Z,Z_save2[i]), axis = 0)

    X = array_recover(X)
    Y = np.array(Y); Y = np.reshape(Y, (Y.shape[0], n_out))
    indexer = np.array(Z)

    inputsize = np.zeros(msunit *fn, dtype=int) 
    for unit in range(msunit *fn):
        inputsize[unit] = X[unit].shape[1] # size 정보는 계속사용하므로, 따로 남겨놓는다.
        
    def keras_setup(lr=lr):
        #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        
        dt = datetime.now()
        idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

        #init = initializers.glorot_normal(seed=None)

        init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
        
        input1 = []; [input1.append([]) for i in range(msunit *fn)] # 최초 input layer
        input2 = []; [input2.append([]) for i in range(msunit *fn)] # input1을 받아서 끝까지 이어지는 변수
        
        for unit in range(msunit *fn):
            input1[unit] = keras.layers.Input(shape=(inputsize[unit], n_in)) # 각 병렬 layer shape에 따라 input 받음
            input2[unit] = Bidirectional(LSTM(n_hidden))(input1[unit]) # biRNN -> 시계열에서 단일 value로 나감
            input2[unit] = Dense(layer_1, kernel_initializer = init, \
                  activation='relu')(input2[unit]) # fully conneted layers, relu
            input2[unit] = Dropout(dropout_rate1)(input2[unit]) # dropout
        
        if msunit *fn == 1:
            added = input2[0]
        elif not(msunit *fn == 1):
            added = keras.layers.Add()(input2) # 병렬구조를 여기서 모두 합침
        merge_1 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate),\
                        activation='relu')(added) # fully conneted layers, relu
#        merge_1 = BatchNormalization()(merge_1)
#        merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
#        merge_2 = Dense(n_out, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), \
#                        activation='relu')(merge_2) # fully conneted layers, sigmoid
##        merge_2 = BatchNormalization()(merge_2)
        
        merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
        merge_2 = Dense(n_out, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), \
                        activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
#        merge_2 = BatchNormalization()(merge_2)
        
        merge_3 = Dense(n_out, input_dim=n_out)(merge_2) # regularization 삭제
        merge_4 = Activation('softmax')(merge_3) # activation as softmax function
        
        model = keras.models.Model(inputs=input1, outputs=merge_4) # input output 선언
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
        
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        return model, idcode
    
    model, idcode = keras_setup()        
    initial_weightsave = RESULT_SAVE_PATH + 'model//' + 'initial_weight.h5'
    model.save_weights(initial_weightsave)
    print(model.summary())
    
    def valid_generation(mousenumlist, only_se=None):
        X_tmp = []; Y_tmp = []; valid = None
        for mousenum in mousenumlist:
            test_mouseNum = mousenum
            
            sessionNum = 5
            if test_mouseNum in se3set:
                sessionNum = 3
            
    #            SE = test_mouseNum
            for se in range(sessionNum):
                if only_se != None and only_se != se:
                    continue
                init = False
                if only_se != None:
                    msclass = 1; init = True # 무적권 pain으로 취급
                elif only_se == None:
                    SE = test_mouseNum
                    set1 = highGroup + midleGroup + lowGroup + yohimbineGroup + ketoGroup + lidocaineGroup + restrictionGroup + highGroup2 
                    c1 = SE in set1 and se in [0,2]
                    c2 = SE in capsaicinGroup and se in [0,2]
                    c3 = SE in pslGroup + adenosineGroup and se in [0]
                    c4 = SE in shamGroup and se in [0,1,2]
                    c5 = SE in salineGroup and se in [0,1,2,3,4]
                    c6 = SE in CFAgroup and se in [0]
                    c7 = SE in chloroquineGroup and se in [0]
                    c8 = SE in itSalineGroup and se in [0]
                    c9 = SE in itClonidineGroup and se in [0,1,2]
    
                    set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
                    c101 = SE in set2 and se in [1]
                    c102 = SE in capsaicinGroup and se in [1]
                    c103 = SE in pslGroup and se in [1,2]
                    c104 = SE in itSalineGroup and se in [1,2]
                                       
                    if c1 or c2 or c3 or c4 or c5 or c6 or c7 or c8 or c9:
                        msclass = 0; init = True
                    elif c101 or c102 or c103 or c104: #
                        msclass = 1; init = True
                        
                    if SE == 132 and se == 2:
                        msclass = 1; init = True
                    if SE == 129 and se == 2:
                        continue
                 
                if init:
#                    print(SE, msclass )
                    binning = list(range(0,(signalss[test_mouseNum][se].shape[0]-full_sequence), bins))
                    if signalss[test_mouseNum][se].shape[0] == full_sequence:
                        binning = [0]
                    binNum = len(binning)
                    
    #                    mssignal2 = np.array(movement_syn[test_mouseNum][se])
                    for i in range(binNum): # range(np.min([2, binNum])):    
                    # each ROI
                        signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                        ROInum = signalss_PSL_test.shape[1]
                        
    #                        mannual_signal2 = mssignal2[binning[i]:binning[i]+full_sequence]
    #                        mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                        
                        for ROI in range(ROInum):
                            mannual_signal = signalss_PSL_test[:,ROI]
                            mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
    
    #                            print(mannual_signal2.shape)
    
                            Xtest, Ytest, _= dataGeneration(test_mouseNum, se, label=msclass, \
                                           Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                            
                            X_tmp += Xtest; Y_tmp += Ytest
                                     
        if np.array(Y_tmp).shape[0] != 0:      
            Xtest = array_recover(X_tmp); 
            Y_tmp = np.array(Y_tmp); Y_tmp = np.reshape(Y_tmp, (Y_tmp.shape[0], n_out))
            valid = tuple([Xtest, Y_tmp])
            
        print('sample num...', len(Y_tmp), \
              'valdiation set distributions...', np.round(np.mean(Y_tmp, axis=0), 4))
        
        return valid        
    
    if False: # 시각화 
        # 20190903, VS code로 옮긴뒤로 에러나는 중, 해결필요
        print(model.summary())
        
        from contextlib import redirect_stdout
        
        with open('modelsummary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
                
        from keras.utils import plot_model
        plot_model(model, to_file='model.png')
             
    ##
        
    print('acc_thr', acc_thr, '여기까지 학습합니다.')
    print('maxepoch', maxepoch)

    # In[]
    picklesavename = RESULT_SAVE_PATH + 'mssave.pickle'
    with open(picklesavename, 'rb') as f:  # Python 3: open(..., 'rb')
        mssave = pickle.load(f)
    
    while True:
        Ylist = list(range(len(Y)))
        msmin = 50 # 조합 최소 갯수    
        random.seed(None)
        rn = random.randrange(msmin, len(Ylist))
        rn = 50
        print('sample min', msmin, 'sample max', len(Ylist), 'rn', rn)
        rlist = random.sample(Ylist, rn)
        X_elite=[]; [X_elite.append([]) for u in range(len(X))]
        for u in range(len(X)):
            X_elite[u] = np.array(X[u])[rlist]
        Y_elite = np.array(Y)[rlist]
        Z_elite = np.array(indexer)[rlist]
    
        trX, trY, trZ = upsampling(X_elite, Y_elite, Z_elite)
           
        testlist = highGroup + midleGroup
        testlist.remove(8); testlist.remove(26)
        valid = valid_generation(testlist, only_se=None)
        validX, validY, _ = upsampling(valid[0], valid[1], valid[1])
        valid = tuple([validX, validY])
        
        reset_keras(model)
        nseed(seed)
        tf.random.set_seed(seed)   
        model, idcode = keras_setup() 
        model.load_weights(initial_weightsave) 
     
        starttime = time.time(); current_acc = -np.inf; cnt=0
        s_loss=[]; s_acc=[]; sval_loss=[]; sval_acc=[] 
        grade_acc = 0.6
        while current_acc < acc_thr and cnt < 500: # 0.93: # 목표 최대 정확도, epoch limit
            if (cnt > maxepoch/epochs) or \
            (current_acc < 0.70 and cnt > 300/epochs) or (current_acc < 0.51 and cnt > 100/epochs):
                break

            current_weightsave = RESULT_SAVE_PATH + '_tmp_model_weights.h5'    
            isfile1 = os.path.isfile(current_weightsave)
      
            if isfile1 and cnt > 0:
                reset_keras(model)
                model, idcode = keras_setup(lr=lr)
                model.load_weights(current_weightsave)
                
            hist = model.fit(trX, trY, batch_size=batch_size, epochs=epochs)
            cnt += 1; model.save_weights(current_weightsave)
            if cnt % 20 == 0 and cnt != 0:
                print('cnt', cnt)
                             
            s_loss += list(np.array(hist.history['loss']))
            s_acc += list(np.array(hist.history['accuracy']))
                                     
            if s_acc[-1] > grade_acc:
                print(grade_acc)
                grade_acc += 0.05
                
                hist = model.fit(trX, trY, batch_size = batch_size, epochs = epochs, \
                                 validation_data = valid)
                cnt += 1; model.save_weights(current_weightsave)
            
                s_loss += list(np.array(hist.history['loss']))
                s_acc += list(np.array(hist.history['accuracy']))
                sval_loss += list(np.array(hist.history['val_loss']))
                sval_acc += list(np.array(hist.history['val_accuracy']))
            
                if s_acc[-1] - 0.04 > sval_acc[-1]:
                    print('overfit 판단, 종료')
                    break
            
            
            # 종료조건: 
            current_acc = s_acc[-1] 
        
#        if sval_acc[-1] > 0.55:
        mssave.append([grade_acc-0.05, trZ, s_loss, s_acc, sval_loss, sval_acc, cnt])
        print('len(mssave)', len(mssave))
        with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)

# In[]      # mean signal 처리
#np.mean(np.array(mssave)[:,-1])

index_value_save = np.zeros(np.max(indexer, axis=0)+1)
index_value_save[:] = np.nan

mssave2 = np.array(mssave)

for i in range(len(mssave2)):
    acctmp = mssave2[i][5][-1]
#    print(acctmp)
    for j in mssave2[i][1]:
        tmp = index_value_save[j[0], j[1], j[2]]
        index_value_save[j[0], j[1], j[2]] = np.nanmean([tmp, acctmp])
#        print(index_value_save[j[0], j[1], j[2]])
        
print('np.nanmean(index_value_save)', np.nanmean(index_value_save))

plt.hist(index_value_save.flatten())
elite_cfa = np.where(index_value_save>0.70)
elite_cfa = np.array(elite_cfa); elite_cfa2=[]
for i in range(elite_cfa.shape[1]):
    elite_cfa2.append(list(elite_cfa[:,i]))
# In[] Formalin or F + CFA로 psl test

# traning set
X_save2, Y_save2, Z_save2 = ms_sampling(forlist=(highGroup + midleGroup + CFAgroup), cfa_set=elite_cfa2)
X_save2, Y_save2, Z_save2 = ms_sampling(forlist=(highGroup + midleGroup), cfa_set=None)
    

X = np.array(X_save2[0]); Y = np.array(Y_save2[0]); Z = np.array(Z_save2[0])
for i in range(1,n_out):
    X = np.concatenate((X,X_save2[i]), axis = 0)
    Y = np.concatenate((Y,Y_save2[i]), axis = 0)
    Z = np.concatenate((Z,Z_save2[i]), axis = 0)

X = array_recover(X)
Y = np.array(Y); Y = np.reshape(Y, (Y.shape[0], n_out))
Z = np.array(Z)

trX, trY, trZ = upsampling(X, Y, Z)

# val set
testlist = pslGroup
valid = valid_generation(testlist, only_se=None)
#validX, validY, _ = upsampling(valid[0], valid[1], valid[1])
#valid = tuple([validX, validY])
# In[]

lr = 1e-3 # learning rate

n_hidden = int(8 * 6) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 6) #

l2_rate = 0.3
dropout_rate1 = 0.2 # dropout late
dropout_rate2 = 0.1 # 

# model reset
reset_keras(model)
nseed(seed)
tf.random.set_seed(seed)   
model, idcode = keras_setup() 
model.load_weights(initial_weightsave) 

# traning 
starttime = time.time(); current_acc = -np.inf; cnt=0
s_loss=[]; s_acc=[]; sval_loss=[]; sval_acc=[] 
grade_acc = [0.6,0.7,0.8,0.85,0.9,0.95]
gix = 0
while current_acc < acc_thr and cnt < 500: # 0.93: # 목표 최대 정확도, epoch limit
    if (cnt > maxepoch/epochs) or \
    (current_acc < 0.70 and cnt > 300/epochs) or (current_acc < 0.51 and cnt > 100/epochs):
        break

    current_weightsave = RESULT_SAVE_PATH + '_tmp_model_weights.h5'    
    isfile1 = os.path.isfile(current_weightsave)
  
    if isfile1 and cnt > 0:
        reset_keras(model)
        model, idcode = keras_setup(lr=lr)
        model.load_weights(current_weightsave)
        
    hist = model.fit(trX, trY, batch_size=batch_size, epochs=epochs)
    cnt += 1; model.save_weights(current_weightsave)
    if cnt % 20 == 0 and cnt != 0:
        print('cnt', cnt)
                     
    s_loss += list(np.array(hist.history['loss']))
    s_acc += list(np.array(hist.history['accuracy']))
                             
    if s_acc[-1] > grade_acc[gix]:
        print(grade_acc[gix])
        gix += 1
        
        hist = model.fit(trX, trY, batch_size = batch_size, epochs = epochs, \
                         validation_data = valid)
        cnt += 1; model.save_weights(current_weightsave)
    
        s_loss += list(np.array(hist.history['loss']))
        s_acc += list(np.array(hist.history['accuracy']))
        sval_loss += list(np.array(hist.history['val_loss']))
        sval_acc += list(np.array(hist.history['val_accuracy']))
    
#        if s_acc[-1] - 0.04 > sval_acc[-1]:
#            print('overfit 판단, 종료')
#            break
        
    # 종료조건: 
    current_acc = s_acc[-1] 

final_weightsave = RESULT_SAVE_PATH + 'model/' + 'final_my_model_weights_final.h5'
model.save_weights(final_weightsave) 
        
dummy_table = np.zeros((N,5))
for test_mouseNum in testlist:
    
    reset_keras(model)
    model, idcode = keras_setup(lr=0)
    model.load_weights(current_weightsave) # subset은 상위 mouse의 final 을 load해야 할것이다.. 확인은 안해봄..
    
    sessionNum = 5
    if test_mouseNum in se3set:
        sessionNum = 3
    for se in range(sessionNum): 
        valid = valid_generation([test_mouseNum], only_se=se)
        print('학습아님.. test 중입니다.', 'SE', test_mouseNum, 'se', se)
        hist = model.fit(valid[0], valid[1], batch_size=batch_size, epochs=1)
#                        # lr = 0 으로 학습안됨. validation이 이 방법이 훨씬 빨라서 사용함.. 
        dummy_table[test_mouseNum, se] = hist.history['accuracy'][-1]

# 최적화용 저장
picklesavename =  RESULT_SAVE_PATH + 'exp_raw/' + 'formalin_capsaicin.pickle'
with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(dummy_table, f, pickle.HIGHEST_PROTOCOL)
    print(picklesavename, '저장되었습니다.')  















#



