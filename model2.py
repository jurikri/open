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
    gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
except:
    try:
        savepath = 'C:\\titan_savepath\\'; os.chdir(savepath);
        gsync = 'C:\\Users\\skklab\\Google 드라이브\\google_syn\\'
        
#        os.path.isfile('C:\\Windows\\addins\\FXSEXT.ecf')
#        os.path.isfile('C:\\titan_savepath\\result\\test.txt')
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
ipsaline_pslGroup = msGroup['ipsaline_pslGroup']
ipclonidineGroup = msGroup['ipclonidineGroup']
gabapentinGroup = msGroup['gabapentinGroup']

msset = msGroup['msset']
msset2 = msGroup['msset2']

del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup \
+ itSalineGroup + itClonidineGroup # for test only

pslset = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup
fset = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
baseonly = lowGroup + lidocaineGroup + restrictionGroup
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
lr = 1e-3 # learning rate
fn = 1

n_hidden = int(8 * 6) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 6) # fully conneted laye node 갯수 # 8 # 원래 6 
# 6 for normal
# 10 for +cfa

#duplicatedNum = 1
#mspainThr = 0.27
#acitivityThr = 0.4
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regulariza3 # regularization 상수
l2_rate = 0.3
dropout_rate1 = 0.20 # dropout late
dropout_rate2 = 0.10 # 

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
batch_size = 2**9 # 5000
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
#project_list.append(['control_test_segment_adenosine_set5', 500, None]) # 4번 까지 완료, 밤에 돌리자 0331 
 
project_list.append(['foramlin_only_1', 100, None])
project_list.append(['foramlin_only_2', 200, None]) 
project_list.append(['foramlin_only_3', 300, None]) 
project_list.append(['foramlin_only_4', 400, None]) 
project_list.append(['foramlin_only_5', 500, None]) 
 
 
#project_list.append(['0330_batchnorm_1', 100, None])
#project_list.append(['0330_batchnorm_2', 200, None])
#project_list.append(['0330_batchnorm_3', 300, None])

q = project_list[0]
for nix, q in enumerate(project_list):
#    if nix == 1:
#        l2_rate = 0.1
#    if nix == 2:
#        l2_rate = 0
    
    print(nix, l2_rate)
    
    settingID = q[0]; seed = q[1]; seed2 = int(seed+1)
    continueSW = q[2]
    
    print('settingID', settingID, 'seed', seed, 'continueSW', continueSW)

    # set the pathway2
    RESULT_SAVE_PATH =  savepath + 'result\\'
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
    set3 = pslGroup + adenosineGroup + shamGroup + CFAgroup + chloroquineGroup + itSalineGroup + itClonidineGroup + ipsaline_pslGroup + ipclonidineGroup + \
    gabapentinGroup
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
    def ms_sampling():
        sampleNum = []; [sampleNum.append([]) for u in range(n_out)]
        
        datasetX = []; datasetY = []; datasetZ = []
        for classnum in range(n_out):
            datasetX.append([]); datasetY.append([]); datasetZ.append([])
            
        # nonpain     
        msclass = 0 # nonpain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in range(N):
            if SE in trainingset:
                if SE in reducing_test_list:
                    for se in range(5):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        set1 = highGroup + midleGroup + lowGroup + yohimbineGroup + ketoGroup + lidocaineGroup + restrictionGroup + highGroup2 
                        c1 = SE in set1 and se in [0]
                        c2 = SE in capsaicinGroup and se in [0]
#                        c3 = SE in pslGroup + adenosineGroup and se in [0]
#                        c4 = SE in shamGroup and se in [0,1,2]
                        c5 = SE in salineGroup and se in [0,1,2,3,4]
                        c6 = SE in CFAgroup and se in [0]
                        c7 = SE in chloroquineGroup and se in [0]
#                        c8 = SE in itSalineGroup and se in [0]
#                        c9 = SE in itClonidineGroup and se in [0]
  
#                        c13 = SE in chloroquineGroup and se in [1]
                                        
                        if c1 or c2 or c5 or c6 or c7:
#                        if c13: #
                            # msset 만 baseline을 제외시킴, total set 아님 
                            exceptbaseline = (SE in np.array(msset)[:,1:].flatten()) and se == 0 
                            if not exceptbaseline: # baseline을 공유하므로, 사용하지 않는다. 
                                mssignal = np.mean(signalss[SE][se], axis=1)
#                                mssignal2 = np.array(movement_syn[SE][se])
                                msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                                
                                for u in msbins:
                                    mannual_signal = mssignal[u:u+full_sequence]
                                    mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                    
#                                    mannual_signal2 = mssignal2[u:u+full_sequence]
#                                    mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
    
                                    X, Y, Z = dataGeneration(SE, se, label=msclass, \
                                                   Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                                    
                                    X_tmp += X; Y_tmp += Y; Z_tmp += Z #; T_tmp += t4_save 
                    
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
        
        sampleNum[msclass] = len(datasetX[msclass])
        print('nonpain_sampleNum', sampleNum[msclass])
        
        msclass = 1 # pain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in range(N):
            if SE in trainingset:
                if SE in reducing_test_list:
                    for se in range(5):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2
                        c11 = SE in set2 and se in [1]
#                        c12 = SE in CFAgroup and se in [1,2]
#                        c13 = SE in chloroquineGroup and se in [1]
                          
                        if c11: # 
                            if not(0.15 < movement[SE,se]):
                                print(SE, se, 'movement 부족, pain session에서 제외.')
                                continue
                        
                            mssignal = np.mean(signalss[SE][se], axis=1)
#                            mssignal2 = np.array(movement_syn[SE][se])
                            msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                            
                            for u in msbins:
                                mannual_signal = mssignal[u:u+full_sequence]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                
#                                mannual_signal2 = mssignal2[u:u+full_sequence]
#                                mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                                
                                X, Y, Z = dataGeneration(SE, se, label=msclass, \
                                               Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                                X_tmp += X; Y_tmp += Y; Z_tmp += Z #; T_tmp += t4_save 
                    
        datasetX[msclass] = np.array(X_tmp)
        datasetY[msclass] = np.array(Y_tmp)
        datasetZ[msclass] = np.array(Z_tmp)
        sampleNum[msclass] = len(datasetX[msclass]); print('pain_sampleNum', sampleNum[msclass])
            
        diff = sampleNum[0] - sampleNum[1]
        print('painsample #', diff, '부족, duplicate로 채움')
        msclass = 1
        
        for u in range(int(diff/sampleNum[1])):
            datasetX[msclass] = np.concatenate((np.array(datasetX[msclass]), np.array(X_tmp)), axis=0)
            datasetY[msclass] = np.concatenate((np.array(datasetY[msclass]), np.array(Y_tmp)), axis=0)
            datasetZ[msclass] = np.concatenate((np.array(datasetZ[msclass]), np.array(Z_tmp)), axis=0)
              
        remain = diff % sampleNum[1]    
              
        random.seed(seed)
        rix = random.sample(range(len(X_tmp)), remain)
                  
        datasetX[msclass] = np.concatenate((np.array(datasetX[msclass]), np.array(X_tmp)[rix]), axis=0)
        datasetY[msclass] = np.concatenate((np.array(datasetY[msclass]), np.array(Y_tmp)[rix]), axis=0)
        datasetZ[msclass] = np.concatenate((np.array(datasetZ[msclass]), np.array(Z_tmp)[rix]), axis=0)

        return datasetX, datasetY, datasetZ
    
    X_save2, Y_save2, Z_save2 = ms_sampling()
#    if continueSW != None:
#        X_save2, Y_save2, Z_save2, t4_save = ms_sampling_continue()
#    painindex_classs = np.concatenate((painindex_class0, painindex_class1), axis=0)
    #  datasetX = X_save; datasetY = Y_save; datasetZ = Z_save
    
    for i in range(n_out):
        print('class', str(i),'sampling 이후', np.array(X_save2[i]).shape[0])

    X = np.array(X_save2[0]); Y = np.array(Y_save2[0]); Z = np.array(Z_save2[0])
    for i in range(1,n_out):
        X = np.concatenate((X,X_save2[i]), axis = 0)
        Y = np.concatenate((Y,Y_save2[i]), axis = 0)
        Z = np.concatenate((Z,Z_save2[i]), axis = 0)

    X = array_recover(X)
    Y = np.array(Y); Y = np.reshape(Y, (Y.shape[0], n_out))
    indexer = np.array(Z)

    # control: label을 session만 유지하면서 무작위로 섞음
    Y_control = np.array(Y)
    for SE in range(N):
        for se in range(5):
            cbn = [SE, se]
            
            identical_ix = np.where(np.sum(indexer==cbn, axis=1)==2)[0]
            if identical_ix.shape[0] != 0:
                random.seed(None)  # control의 경우 seed 없음
                dice = random.choice([[1,0], [0,1]])
                Y_control[identical_ix] = dice
                
    # cross validation을 위해, training / test set split            
    # mouselist는 training set에 사용된 list임.
    # training set에 사용된 mouse의 마릿수 만큼 test set을 따로 만듦
    
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
        merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
        merge_2 = Dense(n_out, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), \
                        activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
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
    
    def valid_generation(mousenumlist, only_se=None, meantest=False):
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
                    binning = list(range(0,(signalss[test_mouseNum][se].shape[0]-full_sequence), bins))
                    if signalss[test_mouseNum][se].shape[0] == full_sequence:
                        binning = [0]
                    binNum = len(binning)
                    
    #                    mssignal2 = np.array(movement_syn[test_mouseNum][se])
                    for i in range(binNum):    
                    # each ROI
                        signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                        ROInum = signalss_PSL_test.shape[1]
                        
                        if not(meantest):
                            for ROI in range(ROInum):
                                mannual_signal = signalss_PSL_test[:,ROI]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
        
        #                            print(mannual_signal2.shape)
        
                                Xtest, Ytest, _= dataGeneration(test_mouseNum, se, label=msclass, \
                                               Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                                
                                X_tmp += Xtest; Y_tmp += Ytest
                                
                        elif meantest:
                            mannual_signal = np.mean(signalss_PSL_test, axis=1)
                            mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
    
                            Xtest, Ytest, _= dataGeneration(test_mouseNum, se, label=msclass, \
                                           Mannual=True, mannual_signal=mannual_signal) #, mannual_signal2=mannual_signal2)
                            
                            X_tmp += Xtest; Y_tmp += Ytest
                                     
        if np.array(Y_tmp).shape[0] != 0:      
            Xtest = array_recover(X_tmp); 
            Y_tmp = np.array(Y_tmp); Y_tmp = np.reshape(Y_tmp, (Y_tmp.shape[0], n_out))
            valid = tuple([Xtest, Y_tmp])
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
    
#    print(reducing_test_list)
    # training set 재설정
    trainingset = trainingset
    print('trainingset #, pre', len(trainingset))
    etc = ipsaline_pslGroup + ipclonidineGroup + gabapentinGroup

    forlist = list(trainingset)
    for SE in forlist:
        c1 = np.sum(indexer[:,0]==SE) == 0 # 옥으로 전혀 선택되지 않았다면 test set으로 빼지 않음
        if c1 and SE in trainingset:
            trainingset.remove(SE)
            print('removed', SE)
            
        c2 = msset_total[:,0]
        if SE in c2:
            try:
                for u in np.array(msset_total)[np.where(np.array(msset_total)[:,0] == SE)[0][0],:][1:]:
                    trainingset.remove(u)
                    print('subset 포함을 위한 제거', u)
            except:
                print('예외처리', SE)
                
        if SE in chloroquineGroup and SE in trainingset: # chloroquineGroup을 training에만 사용하고, test하지 않음
            trainingset.remove(SE)
            print('chloroquineGroup 평가하지 않음', SE)

    mouselist = trainingset # 사용 중지 
    mouselist.sort()
    print('mouselist #', len(mouselist))
    
#    if savepath == 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\':
#    mouselist = list(np.sort(np.array(mouselist))[::-1]) # runlist reverse
    
    if not(len(etc) == 0):
        mouselist.append(etc[0])
    
    # 학습할 set 결정, 따로 조작하지 않을 땐 mouselist로 설정하면 됨.
#    tmp = []
#    for t in mouselist:
#        if not t in pslset + capsaicinGroup + CFAgroup:
#            tmp.append(t)
            
    wanted = [etc[0]]
    # pslset + capsaicinGroup + CFAgroup
#    wanted = np.sort(wanted)
    mannual = [] # 절대 아무것도 넣지마 

    print('mouselist', mouselist)
    print('etc', etc)
    for i in wanted:
        try:
            mannual.append(np.where(np.array(mouselist)==i)[0][0])
        except:
            print(i, 'is excluded.', 'etc group에서 확인')
            
    print('wanted', np.array(mouselist)[mannual])
            
    # save_hyper_parameters 기록남기기
    save_hyper_parameters = []
    save_hyper_parameters.append(['settingID', settingID])
    save_hyper_parameters.append(['epochs', epochs])
    save_hyper_parameters.append(['lr', lr])
    save_hyper_parameters.append(['n_hidden', n_hidden])
    save_hyper_parameters.append(['layer_1', layer_1])
    save_hyper_parameters.append(['l2_rate', l2_rate])
    save_hyper_parameters.append(['dropout_rate1', dropout_rate1])
    save_hyper_parameters.append(['dropout_rate2', dropout_rate2])
    save_hyper_parameters.append(['acc_thr', acc_thr])
    save_hyper_parameters.append(['batch_size', batch_size])
    save_hyper_parameters.append(['seed', seed])
#    save_hyper_parameters.append(['classratio', classratio])
    save_hyper_parameters.append(['mouselist', mouselist])
    save_hyper_parameters.append(['full_sequence', full_sequence])
    
    savename4 = RESULT_SAVE_PATH + 'model/' + '00_model_save_hyper_parameters.csv'
    
    if not (os.path.isfile(savename4)):
        print(settingID, 'prameter를 저장합니다. prameter를 저장합니다. prameter를 저장합니다.')
        csvfile = open(savename4, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        for row in range(len(save_hyper_parameters)):
            csvwriter.writerow(save_hyper_parameters[row])
        csvfile.close()
        
    # In[]

    sett = 0; ix = 0
    for ix, sett in enumerate(mannual):
        final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
        print('final_weightsave', final_weightsave)
        
        os.path.isfile('C:\\titan_savepath\\result\\foramlin_only_1\\model\\0_my_model_weights_final.h5')
        os.path.isfile('C:\\Windows\\addins\\FXSEXT.ecf')
        os.path.isfile('‪C:\\titan_savepath\\test.txt')
        os.path.isfile('C:\\test.txt')
        
       
        
        exist_model = os.path.isfile(final_weightsave)
        print('training을 위한 model 존재 유무', exist_model)

        if not(exist_model) and trainingsw: # trainingsw
            print('mouse #', [mouselist[sett]], '학습된 model 없음. 새로시작합니다.')
                  
            reset_keras(model)
            nseed(seed)
            tf.random.set_seed(seed)   
            model, idcode = keras_setup() 
            model.load_weights(initial_weightsave) 

            # cv, training / dev set 생성, dev로 model 최적화를 따로 진행하지 않음. 단지 학습상황 확인용으로만 사용됨.
            
            # pre allocation
            X_training = []; [X_training.append([]) for i in range(msunit *fn)] # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
            X_valid = []; [X_valid.append([]) for i in range(msunit *fn)]
            
            # trainig set, delist는 test에 해당함. 쥐를 기준으로 한마리씩 뻄 
            delist = np.where(indexer[:,0]==mouselist[sett])[0]
            
            if mouselist[sett] in np.array(msset_total)[:,0]:
                for u in np.array(msset_total)[np.where(np.array(msset_total)[:,0] == mouselist[sett])[0][0],:][1:]:
                    print(mouselist[sett], 'subset으로써 추가 제거됩니다.', u)
                    delist = np.concatenate((delist, np.where(indexer[:,0]==u)[0]), axis=0)
            
            for unit in range(msunit *fn): # input은 msunit 만큼 병렬구조임. for loop으로 각자 계산함
                X_training[unit] = np.delete(np.array(X[unit]), delist, 0)
#                        X_valid[unit] = np.array(X[unit])[delist]
        
            Y_training_list = np.delete(np.array(Y), delist, 0)
                   
            print('mouse #', [mouselist[sett]])
            print('sample distributions.. ', np.round(np.mean(Y_training_list, axis = 0), 4))
            
            # training bias 방지를 위해 동일하게 shuffle 
            np.random.seed(seed)
            shuffleix = list(range(X_training[0].shape[0]))
#            np.random.shuffle(shuffleix)
            shuffleix = random.sample(shuffleix, int(round(len(shuffleix)/1))) # reducing
#                    print(shuffleix)
   
            tr_y_shuffle = Y_training_list[shuffleix]
            tr_x = []
            for unit in range(msunit *fn):
                tr_x.append(X_training[unit][shuffleix])


            # 특정 training acc를 만족할때까지 epoch를 epochs단위로 지속합니다.
            current_acc = -np.inf; cnt = 0
            hist_save_loss = []
            hist_save_acc = []
            hist_save_val_loss = []
            hist_save_val_acc = []
            
            testlist = [mouselist[sett]]
            if mouselist[sett] in np.array(msset_total)[:,0]:
                for u in np.array(msset_total)[np.where(np.array(msset_total)[:,0] == mouselist[sett])[0][0],:][1:]:
                    testlist.append(u)
            
            if not(len(etc) == 0):
                if etc[0] == mouselist[sett]:
                    print('test ssesion, etc group 입니다.') 
                    testlist = list(etc)
            
            
            starttime = time.time()
            grade_acc = 0.86
            while current_acc < acc_thr: # 0.93: # 목표 최대 정확도, epoch limit
                acc_thr_sw = False
                if (cnt > maxepoch/epochs) or (current_acc < 0.70 and cnt > 300/epochs) or ( current_acc < 0.51 and cnt > 50/epochs):
                    seed += 1
                    reset_keras(model)
                    nseed(seed)
                    tf.random.set_seed(seed)   
                    model, idcode = keras_setup() 
#                    model.load_weights(initial_weightsave)      
                    current_acc = -np.inf; cnt = -1
                    print('seed 변경, model reset 후 처음부터 다시 학습합니다.')

                current_weightsave = RESULT_SAVE_PATH + 'tmp/'+ str(idcode) + '_' + str(mouselist[sett]) + '_my_model_weights.h5'
                
                isfile1 = os.path.isfile(current_weightsave)
#                if isfile1 and cnt > 0:
#                    model.load_weights(current_weightsave)
#                    print('mouse #', [mouselist[sett]], cnt, '번째 이어서 학습합니다.')
#                else:
#                    print('학습 진행중인 model 없음. 새로 시작합니다')
                
                if mouselist[sett] == etc[0]:
                    valid = valid_generation(etc, only_se=None)
                else:
                    valid = valid_generation([mouselist[sett]], only_se=None)
                  
                if isfile1 and cnt > 0:
                    reset_keras(model)
                    model, idcode = keras_setup(lr=lr)
                    model.load_weights(current_weightsave)
                    print('mouse #', [mouselist[sett]], cnt, '번째 이어서 학습합니다.')
                    
                hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs)
                cnt += 1
                hist_save_loss += list(np.array(hist.history['loss']))
                hist_save_acc += list(np.array(hist.history['accuracy']))
                if hist_save_acc[-1] > grade_acc:
                    acc_thr_sw = True
                    print(grade_acc)
                    grade_acc += 0.02
                                          
                model.save_weights(current_weightsave)
                
                # 종료조건: 
                current_acc = hist_save_acc[-1] 
                
                
                if acc_thr_sw:
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
                    tmp = 'l2_rate_' + str(l2_rate) +  '_current_acc_' + str(round(current_acc,3)) 
                    picklesavename =  RESULT_SAVE_PATH + 'exp_raw/' + 'valid_' + tmp + '.pickle'
                    with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(dummy_table, f, pickle.HIGHEST_PROTOCOL)
                        print(picklesavename, '저장되었습니다.')  
                           
            model.save_weights(final_weightsave)   
            print('mouse #', [mouselist[sett]], 'traning 종료, final model을 저장합니다.')

            # hist 저장      
            plt.figure();
            mouseNum = mouselist[sett]
            hist_save_loss_plot = np.array(hist_save_loss)/np.max(hist_save_loss)
            
            plt.plot(hist_save_loss_plot, label= '# ' + str(mouseNum) + ' loss')
            plt.plot(hist_save_acc, label= '# ' + str(mouseNum) + ' acc')
            plt.legend()
            plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_trainingSet_result.png')
            plt.close()

            savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_trainingSet_result.csv'
            csvfile = open(savename, 'w', newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(hist_save_acc)
            csvwriter.writerow(hist_save_loss)
            spendingtime = time.time() - starttime
            csvwriter.writerow([spendingtime, spendingtime/60, spendingtime/60**2])
            csvfile.close()

#            if validation_sw:
#                plt.figure();
#                mouseNum = mouselist[sett]
#                hist_save_val_loss_plot = np.array(hist_save_val_loss)/np.max(hist_save_val_loss)
#                
#                plt.plot(hist_save_val_loss_plot, label= '# ' + str(mouseNum) + ' loss')
#                plt.plot(hist_save_val_acc, label= '# ' + str(mouseNum) + ' acc')
#                plt.legend()
#                plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_validationSet_result.png')
#                plt.close()
#
#                savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_validationSet_result.csv'
#                csvfile = open(savename, 'w', newline='')
#                csvwriter = csv.writer(csvfile)
#                csvwriter.writerow(hist_save_val_acc)
#                csvwriter.writerow(hist_save_val_loss)
#                csvfile.close()

        ####### test 구문 입니다. ##########        
        # testlist는 위에 작성한 dev set의 testlist 변수를 그대로 이어 받는다.
        # 단 etc에 경우 teslist를 따로 만든다.
        testlist = [mouselist[sett]]
        if mouselist[sett] in np.array(msset_total)[:,0]:
            for u in np.array(msset_total)[np.where(np.array(msset_total)[:,0] == mouselist[sett])[0][0],:][1:]:
                testlist.append(u)
        
        if not(len(etc) == 0):
            if etc[0] == mouselist[sett]:
                print('test ssesion, etc group 입니다.') 
                testlist = list(etc)
        
        final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
        isfile2 = os.path.isfile(final_weightsave)
        print(final_weightsave)
        print('test를 위한 model 존재 유무', isfile2)    
        
        if isfile2 and testsw3:
            for test_mouseNum in testlist:
                picklesavename = RESULT_SAVE_PATH + 'exp_raw/' + 'testsw3_' + str(test_mouseNum) + '.pickle'
                if not(os.path.isfile(picklesavename)) or False: # 만들어야될게 없으면 실행 or overwrite
                    dummy_table = np.zeros((N,5))
                    reset_keras(model)
                    model, idcode = keras_setup(lr=0)
                    model.load_weights(final_weightsave) # subset은 상위 mouse의 final 을 load해야 할것이다.. 확인은 안해봄..
                    
                    sessionNum = 5
                    if test_mouseNum in se3set:
                        sessionNum = 3
                    for se in range(sessionNum): 
                        valid = valid_generation([test_mouseNum], only_se=se)
                        print('학습아님.. test 중입니다.', 'SE', test_mouseNum, 'se', se)
                        hist = model.fit(valid[0], valid[1], batch_size=batch_size, epochs=1)
                        # lr = 0 으로 학습안됨. validation이 이 방법이 훨씬 빨라서 사용함.. 
                        
                        dummy_table[test_mouseNum, se] = hist.history['accuracy'][-1]

                    with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(dummy_table, f, pickle.HIGHEST_PROTOCOL)
                        print(picklesavename, '저장되었습니다.')
                        
                # meantest
                picklesavename2 = RESULT_SAVE_PATH + 'exp_raw/' + 'testsw3_' + str(test_mouseNum) + 'mean.pickle'
                if not(os.path.isfile(picklesavename2)) or False: # 만들어야될게 없으면 실행 or overwrite
                    dummy_table = np.zeros((N,5))
                    reset_keras(model)
                    model, idcode = keras_setup(lr=0)
                    model.load_weights(final_weightsave) # subset은 상위 mouse의 final 을 load해야 할것이다.. 확인은 안해봄..
                    
                    sessionNum = 5
                    if test_mouseNum in se3set:
                        sessionNum = 3
                    for se in range(sessionNum): 
                        valid = valid_generation([test_mouseNum], only_se=se, meantest=True)
                        print('학습아님.. test 중입니다.', 'SE', test_mouseNum, 'se', se)
                        hist = model.fit(valid[0], valid[1], batch_size=batch_size, epochs=1)
                        # lr = 0 으로 학습안됨. validation이 이 방법이 훨씬 빨라서 사용함.. 
                        
                        dummy_table[test_mouseNum, se] = hist.history['accuracy'][-1]

                    with open(picklesavename2, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(dummy_table, f, pickle.HIGHEST_PROTOCOL)
                        print(picklesavename2, '저장되었습니다.')
                        

        ####### test - binning 구문 입니다. ##########, test version 2
        # model load는 cv set 시작에서 무조건 하도록 되어 있음.
        if isfile2 and testsw2:
            model.load_weights(final_weightsave) # training / test는 흔히 별개로 처리되곤하기 때문에, 다시 로드한다.
            
            for test_mouseNum in testlist:
                testbin = None
                picklesavename = RESULT_SAVE_PATH + 'exp_raw/' + 'PSL_result_' + str(test_mouseNum) + '.pickle'
                picklesavename2 = RESULT_SAVE_PATH + 'exp_raw/' + 'PSL_result_mean_' + str(test_mouseNum) + '.pickle'
                
                isfile3 = os.path.isfile(picklesavename)
                if isfile3:
                    print('PSL_result_' + str(test_mouseNum) + '.pickle', '이미 존재합니다. skip')

                if not(isfile3):
                    PSL_result_save = []
                    [PSL_result_save.append([]) for i in range(N)]
                    PSL_result_save_mean = []
                    [PSL_result_save_mean.append([]) for i in range(N)]
                    
                    for SE2 in range(N):
                        [PSL_result_save[SE2].append([]) for i in range(5)]
                        [PSL_result_save_mean[SE2].append([]) for i in range(5)]
                    # PSL_result_save변수에 무조건 동일한 공간을 만들도록 설정함. pre allocation 개념
                    
                    sessionNum = 5
                    if test_mouseNum in se3set:
                        sessionNum = 3
                    
                    for se in range(sessionNum):
                        
                        binning = list(range(0,(signalss[test_mouseNum][se].shape[0]-full_sequence), bins))
                        binNum = len(binning)
                        
                        if signalss[test_mouseNum][se].shape[0] == full_sequence:
                            binNum = 1
                            binning = [0]
                                                       
                        [PSL_result_save[test_mouseNum][se].append([]) for i in range(binNum)]
                        [PSL_result_save_mean[test_mouseNum][se].append([]) for i in range(binNum)]
                        
                        i = 54; ROI = 0
                        for i in range(binNum):    
                            # each ROI
                            signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                            ROInum = signalss_PSL_test.shape[1]
                            
                            [PSL_result_save[test_mouseNum][se][i].append([]) for k in range(ROInum)]
                            
                            mannual_signal2 = movement_syn[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                            mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                            for ROI in range(ROInum):
                                mannual_signal = signalss_PSL_test[:,ROI]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                
                                X, _, _, _ = dataGeneration(test_mouseNum, se, label=0, \
                                       Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                                    
                                X_array = array_recover(X)
                                print(test_mouseNum, se, 'BINS', i ,'/', binNum, 'ROI', ROI)
                                prediction = model.predict(X_array)
                                PSL_result_save[test_mouseNum][se][i][ROI] = prediction
                                
                            # ROI mean
                            mean_signal = np.mean(signalss_PSL_test, axis=1)
                            mean_signal = np.reshape(mean_signal, (mean_signal.shape[0], 1))

                            X, _, _, _ = dataGeneration(test_mouseNum, se, label=0, \
                                       Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                                
                            X_array = array_recover(X)
                            prediction = model.predict(X_array)
                            PSL_result_save_mean[test_mouseNum][se][i] = prediction
                            
                    with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(PSL_result_save, f, pickle.HIGHEST_PROTOCOL)
                        print(picklesavename, '저장되었습니다.')
                        
                    with open(picklesavename2, 'wb') as f:  # Python 3: open(..., 'wb')
                        pickle.dump(PSL_result_save_mean, f, pickle.HIGHEST_PROTOCOL)
                        print(picklesavename2, '저장되었습니다.')
# In[]      # mean signal 처리
       

# In[]










#



