# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""
import os  # 경로 관리
# library import
import pickle # python 변수를 외부저장장치에 저장, 불러올 수 있게 해줌
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime # 시관 관리 
import csv
import random
#import tensorflow as tf
#from tensorflow.keras import regularizers

from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam


# set pathway
try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

# check the save pathway
try:
    df2 = [['SE', 'se', '%']]
    df2.append([1, 1, 1])
    csvfile = open('mscsvtest.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in range(len(df2)):
        csvwriter.writerow(df2[row])
    
    csvfile.close()
except:
    print('저장경로가 유효하지 않습니다.')

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    

#with open('pointSave.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
#    pointSave = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
#behavss2 = msdata_load['behavss2'] # 투포톤과 syn 맞춰진 버전 
#movement = msdata_load['movement'] # 움직인정보를 평균내서 N x 5 matrix에 저장
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

def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

# In[]

movement_syn = []
[movement_syn.append([]) for u in range(N)]
#bahavss[0][1].shape[0]/120 = downsize_movement
downsize_signals = signalss[0][1].shape[0]/120
for SE in range(N):
    [movement_syn[SE].append([]) for u in range(5)]
    for se in range(5):
        downsize = int(round(signalss[SE][se].shape[0]/downsize_signals))
        movement_syn[SE][se] = downsampling(bahavss[SE][se], downsize)

for SE in range(N):
#    print('downsizing', SE)
    for se in range(5):
        signaltmp = []
        downsize = int(round(signalss[SE][se].shape[0]/downsize_signals))
        for roi in range(signalss[SE][se].shape[1]):
            signaltmp.append(downsampling(signalss[SE][se][:,roi], downsize))
#        print(SE, se, downsize, signaltmp[se].shape[0])
        signalss[SE][se] = np.transpose(np.array(signaltmp))
        
print('dwonsize check', signalss[4][1].shape)


#plt.plot(signalss[10][1])
    
#        print(np.mean(movement_syn[SE][se]))
##plt.plot(movement_syn[1][1])
#import sys
#sys.exit()    
#

# In[]

msset = msGroup['msset']
del msGroup['msset']

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup
pslset = pslGroup + shamGroup + adenosineGroup

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

bins = 5 # 최소 time frame 간격

totaldataset = grouped_total_list

# 최소길이 찾기
mslength = np.zeros((N,5)); mslength[:] = np.nan
for SE in range(N):
    if SE in totaldataset:
        for se in range(5):
#            if [SE, se] in longlist:
            signal = np.array(signalss[SE][se])
            mslength[SE,se] = signal.shape[0]

full_sequence = int(np.nanmin(mslength))
print('full_sequence', full_sequence, 'frames')

# In[]
        
#shortlist = []; longlist = []
#for SE in range(N):
#    if SE in totaldataset:
#        for se in range(5):
#            length = np.array(signalss[SE][se]).shape[0]
#            if length > 180*FPS:
#                longlist.append([SE,se])
#            elif length < 180*FPS:
#                shortlist.append([SE,se])
#            else:
#                print('error')                   

#msset = [[70,72],[71,84],[75,85],[76,86], [79,88]]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
def array_recover(X_like): #[samplenum][bins][segment(8)][안에는 each np.array]
    X_like_toarray = []; X_like = np.array(X_like)
    
    for sampleNum in range(X_like.shape[0]):
        sampletmp = []
        for binss in range(X_like.shape[1]):
            for segement in range(msunit *fn):
                sampletmp.append(X_like[sampleNum][binss][segement])
                
        X_like_toarray.append(sampletmp)
    X_like_toarray = np.array(X_like_toarray)
    Xs = X_like_toarray.shape
    X_like_toarray = np.reshape(X_like_toarray, (Xs[0], Xs[1], 1))

    return X_like_toarray

# data 생성
SE = 70; se = 1; label = 1; roiNum=None; GAN=False; Mannual=False; mannual_signal=None; passframesave=np.array([])
# signal1 = np.mean(signalss[SE][se],axis=1)[:full_sequence]
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
        
#    elif not(Mannual):
#        signal_full = np.array(signalss_cut[SE][se])
        
    signal1 = np.mean(signal_full[:,s:e], axis=1) # 단일 ROI만 선택하는 것임
    signal2 = np.mean(mannual_signal2[:,s:e], axis=1)
    
#    del signal1
#    signal1 = np.array(signal2)  # signal1을 movement로 intercept # movement를 signal1로 작업할 떄만 사용
    
#    if GAN:
#        signal_full = np.array(GAN_data[SE][se])
#        signal_full_roi = np.mean(signal_full[:,s:e], axis=1)
    
    lastsave = np.zeros(msunit, dtype=int)
    lastsave2 = np.zeros(msunit, dtype=int) # 체크용
    
    binlist = list(range(0, full_sequence-np.min(sequenceSize), bins))

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
#                print(frame, unit, lastsave[unit])
                if unit == 0:
                    t4_save.append(np.mean(signal1[lastsave[unit] : lastsave[unit] + sequenceSize[unit]]))
                    
        if fn > 1:
            branchNum = 1
            signal2 = np.array(signal2)
            for unit in range(msunit):
                if frame <= full_sequence - sequenceSize[unit]:
                    X_tmp[unit+msunit*branchNum] = (signal2[frame : frame + sequenceSize[unit]])
                    lastsave[unit] = frame
                    
                    if unit == 0:
                        t4_save.append(np.mean(signal2[frame : frame + sequenceSize[unit]]))
                    
                else:
                    X_tmp[unit+msunit*branchNum]  = (signal2[lastsave[unit] : lastsave[unit] + sequenceSize[unit]])
    #                print(frame, unit, lastsave[unit])
                    if unit == 0:
                        t4_save.append(np.mean(signal2[lastsave[unit] : lastsave[unit] + sequenceSize[unit]]))   
                    
    ############
        X.append(X_tmp)
        Y.append(label)
        Z.append([SE,se])

    return X, Y, Z, t4_save

#signalss_cut = preprocessing(endpoint=int(full_sequence))

msunit = 6 # input으로 들어갈 시계열 길이 및 갯수를 정함. full_sequence기준으로 1/n, 2/n ... n/n , n/n

sequenceSize = np.zeros(msunit) # 각 시계열 길이들을 array에 저장
for i in range(msunit):
    sequenceSize[i] = int(full_sequence/msunit*(i+1))
sequenceSize = sequenceSize.astype(np.int)

print('full_sequence', full_sequence)
print('sequenceSize', sequenceSize)

  
###############
# hyperparameters #############
 
# learning intensity
epochs = 100 # epoch 종료를 결정할 최소 단위.
lr = 1e-3 # learning rate
fn = 1

n_hidden = int(8 * 3) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 3) # fully conneted laye node 갯수 # 8

parallel = len(list(range(0, full_sequence-np.min(sequenceSize), bins)))
#duplicatedNum = 1
#mspainThr = 0.27
#acitivityThr = 0.4
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regularization
l2_rate = 0.25 # regularization 상수
dropout_rate1 = 0.20 # dropout late
dropout_rate2 = 0.10 # 

#testsw = False  # test 하지 않고 model만 저장함. # cloud 사용량을 줄이기 위한 전략.. 
trainingsw = True # training 하려면 True 
statelist = ['exp'] # ['exp', 'con']  # random shuffled control 사용 유무
validation_sw = True # 시각화목적으로만 test set을 validset으로 배치함.
testsw2 = False
#if testsw2:
##    import os
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = ''
#    import tensorflow as tf

# 집 컴퓨터, test 전용으로 수정
acc_thr = 0.95 # 0.93 -> 0.94
batch_size = 3000 # 5000

c1 = savepath == 'D:\\painDecorder\\save\\tensorData\\' or savepath == 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'
if True and c1:
    trainingsw = True
    testsw2 = False
    batch_size = 200

###############

# constant 
maxepoch = 5000
n_in =  1 # number of features
n_out = 2 # number of class # 20191104: 3 class로 시도
classratio = 1 # class under sampling ratio

project_list = []
 # proejct name, seed
#project_list.append(['1223_formalin_movement_1', 100, None])
#project_list.append(['1223_formalin_movement_2', 200, None])
#project_list.append(['1226_adenosine_1', 100, None])
#project_list.append(['1226_adenosine_2', 200, None])
#project_list.append(['1226_adenosine_3', 300, None])
#project_list.append(['1226_adenosine_4', 400, None])
#project_list.append(['1226_adenosine_5', 500, None])
 
#project_list.append(['0107_first_1', 100, None])
#project_list.append(['0107_first_2', 200, None])
#project_list.append(['0107_first_3', 300, None])
#project_list.append(['0107_first_4', 400, None])
#project_list.append(['0107_first_5', 500, None])

project_list.append(['0114_double_merge', 100, None])

q = project_list[0]
for q in project_list:
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
                for se in range(5):      
                    # pain Group에 들어갈 수 있는 모든 경우의 수 
                    set1 = highGroup + midleGroup + lowGroup + yohimbineGroup + ketoGroup + lidocaineGroup + highGroup2 
                    c1 = SE in set1 and se in [0,2]
                    c2 = SE in capsaicinGroup and se in [0,2]
                    c3 = SE in pslGroup + adenosineGroup and se in [0]
                    c4 = SE in shamGroup and se in [0,1,2]
                    c5 = SE in salineGroup and se in [0,1,2,3,4]
                                    
                    if c1 or c2 or c3 or c4 or c5:
                        exceptbaseline = (SE in np.array(msset)[:,1:].flatten()) and se == 0
                        if not exceptbaseline: # baseline을 공유하므로, 사용하지 않는다. 
                            mssignal = np.mean(signalss[SE][se], axis=1)
                            mssignal2 = np.array(movement_syn[SE][se])
                            msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                            
                            for u in msbins:
                                mannual_signal = mssignal[u:u+full_sequence]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                
                                mannual_signal2 = mssignal2[u:u+full_sequence]
                                mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))

                                X, Y, Z, t4_save = dataGeneration(SE, se, label=msclass, \
                                               Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                                
                                X_tmp.append(X); Y_tmp.append(Y[0]); Z_tmp.append(Z[0])
                    
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
        
        sampleNum[msclass] = len(datasetX[msclass])
        print('nonpain_sampleNum', sampleNum[msclass])
        
        msclass = 1 # pain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in range(N):
            if SE in trainingset:
                for se in range(5):      
                    # pain Group에 들어갈 수 있는 모든 경우의 수 
                    set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup + highGroup2
                    c11 = SE in set2 and se in [1]
                    c12 = SE in pslGroup and se in [1,2]
                    
                    if c11 or c12: # 
                        mssignal = np.mean(signalss[SE][se], axis=1)
                        mssignal2 = np.array(movement_syn[SE][se])
                        msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                        
                        for u in msbins:
                            mannual_signal = mssignal[u:u+full_sequence]
                            mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                            
                            mannual_signal2 = mssignal2[u:u+full_sequence]
                            mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))

                            X, Y, Z, _ = dataGeneration(SE, se, label=msclass, \
                                           Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                            X_tmp.append(X); Y_tmp.append(Y[0]); Z_tmp.append(Z[0])
                    
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

#    # control: label을 session만 유지하면서 무작위로 섞음
#    Y_control = np.array(Y)
#    for SE in range(N):
#        for se in range(5):
#            cbn = [SE, se]
#            
#            identical_ix = np.where(np.sum(indexer==cbn, axis=1)==2)[0]
#            if identical_ix.shape[0] != 0:
#                random.seed(None)  # control의 경우 seed 없음
#                dice = random.choice([[1,0], [0,1]])
#                Y_control[identical_ix] = dice
                
    # cross validation을 위해, training / test set split            
    # mouselist는 training set에 사용된 list임.
    # training set에 사용된 mouse의 마릿수 만큼 test set을 따로 만듦
    
    inputsize = np.zeros(msunit *fn, dtype=int) 
    for unit in range(msunit *fn):
        inputsize[unit] = X[0][unit,0].shape[0] # size 정보는 계속사용하므로, 따로 남겨놓는다.
        
    def keras_setup():
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        
        dt = datetime.now()
        idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

        init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
        
        input1 = []; [input1.append([]) for i in range(parallel)] 
        input2 = []; [input2.append([]) for i in range(parallel)] 
        added = []; [added.append([]) for i in range(parallel)]
        merge_1 = []; [merge_1.append([]) for i in range(parallel)]
        merge_2 = []; [merge_2.append([]) for i in range(parallel)]
        merge_3 = []; [merge_3.append([]) for i in range(parallel)]
        merge_4 = []; [merge_4.append([]) for i in range(parallel)]
        model = []; [model.append([]) for i in range(parallel)]
        
        for u in range(parallel):
            [input1[u].append([]) for i in range(msunit *fn)] # 최초 input layer
            [input2[u].append([]) for i in range(msunit *fn)] # input1을 받아서 끝까지 이어지는 변수
            
            for unit in range(msunit *fn):
                input1[u][unit] = keras.layers.Input(shape=(inputsize[unit], n_in)) # 각 병렬 layer shape에 따라 input 받음
                input2[u][unit] = Bidirectional(LSTM(n_hidden))(input1[u][unit]) # biRNN -> 시계열에서 단일 value로 나감
                input2[u][unit] = Dense(layer_1, kernel_initializer = init, activation='relu')(input2[u][unit]) # fully conneted layers, relu
                input2[u][unit] = Dropout(dropout_rate1)(input2[u][unit]) # dropout
        
            added[u] = keras.layers.Add()(input2[u]) # 병렬구조를 여기서 모두 합침

            merge_1[u] = Dense(layer_1, kernel_initializer = init, activation='relu')(added[u]) # fully conneted layers, relu
            merge_2[u] = Dropout(dropout_rate2)(merge_1[u]) # dropout
            merge_2[u] = Dense(n_out, kernel_initializer = init, activation='relu')(merge_2[u]) # fully conneted layers, sigmoid
            merge_3[u] = Dense(n_out, input_dim=n_out, kernel_regularizer=regularizers.l2(l2_rate))(merge_2[u]) # regularization
#            merge_4[u] = Activation('softmax')(merge_3[u]) # activation as softmax function
            model[u] = keras.models.Model(inputs=input1[u], outputs=merge_3[u]) # input output 선언

        combined = keras.layers.concatenate([model[0].output, model[1].output])
        z = Dense(layer_1, kernel_initializer = init, activation='relu')(combined)
        z = Dense(2, kernel_initializer = init, activation='sigmoid')(z)
        z = Activation('softmax')(z)
        
        input_tmp = []
        for u in range(parallel):
            for k in range(len(model[u].input)):
                input_tmp.append(model[u].input[k])

        model_merge = keras.models.Model(inputs=input_tmp, outputs=z) # input output 선언
        model_merge.compile(loss='categorical_crossentropy', \
                            optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), \
                            metrics=['accuracy']) # optimizer
        
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        return model_merge, idcode
    
    model, idcode = keras_setup()        
    initial_weightsave = RESULT_SAVE_PATH + 'model//' + 'initial_weight.h5'
    model.save_weights(initial_weightsave)
    
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
    
    # training set 재설정
    trainingset = trainingset; etc = []
    forlist = list(trainingset)
    for SE in forlist:
        c1 = np.sum(indexer[:,0]==SE) == 0 # 옥으로 전혀 선택되지 않았다면 test set으로 빼지 않음
        if c1 and SE in trainingset:
            trainingset.remove(SE)
            print('removed', SE)
            
            if not SE in np.array(msset).flatten():
                etc.append(SE)
            
        c2 = np.array(msset)[:,0]
        if SE in c2:
            for u in np.array(msset)[np.where(np.array(msset)[:,0] == SE)[0][0],:][1:]:
                trainingset.remove(u)

    mouselist = trainingset
    mouselist.sort()
    
#    if savepath == 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\':
#    mouselist = list(np.sort(np.array(mouselist))[::-1]) # runlist reverse
    
    if not(len(etc) == 0):
        mouselist.append(etc[0])
    
    # 학습할 set 결정, 따로 조작하지 않을 땐 mouselist로 설정하면 됨.
    
    wanted = pslset
#    wanted = np.sort(wanted)
    mannual = [] # 절대 아무것도 넣지마 

    print('mouselist', mouselist)
    print('etc', etc)
    for i in wanted:
        try:
            mannual.append(np.where(np.array(mouselist)==i)[0][0])
        except:
            print(i, 'is excluded.', 'etc group에서 확인')
            
#    mannual = list(np.sort(np.array(mannual))[::-1]) # runlist reverse
    print('wanted', np.array(mouselist)[mannual])
            
#    np.random.seed(seed2)
#    shuffleix = list(range(len(mannual)))                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#    np.random.shuffle(shuffleix)
#    print('shuffleix', shuffleix)
#    mannual = np.array(mannual)[shuffleix]
#    print('etc ix', np.where(np.array(mouselist)== etc)[0])
#     구지 mannual을 두고 다시 indexing 하는 이유는, 인지하기 편하기 때문임. 딱히 안써도 됨
    
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

    sett = 0; ix = 0; state = 'exp' # for test
    for state in statelist:
        for ix, sett in enumerate(mannual):
            # training 구문입니다.
            exist_model = False; recent_model = False

            # training된 model이 있는지 검사
            if state == 'exp':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
        #        print('exp')
            elif state == 'con':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final_control.h5'
        #        print('con')

            print('final_weightsave', final_weightsave)

            try:
                model.load_weights(final_weightsave) 
                exist_model = True
                print('exist_model', exist_model)
            except:
                exist_model = False
                print('exist_model', exist_model, 'load 안됨')

            # 없다면, 2시간 이내에 training이 시작되었는지 검사
            if not(exist_model) and trainingsw:
                if state == 'exp':
                    loadname = RESULT_SAVE_PATH + 'tmp/' + str([mouselist[sett]]) + '_log.csv'
                elif state == 'con':
                    loadname = RESULT_SAVE_PATH + 'tmp/' + str([mouselist[sett]]) + '_log_control.csv'

                try:
                    mscsv = []       
                    f = open(loadname, 'r', encoding='utf-8')
                    rdr = csv.reader(f)
                    for line in rdr:
                        mscsv.append(line)
                    f.close()    
                    mscsv = np.array(mscsv)

                    dt = datetime.now()
                    idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

                    sameday = int(idcode) == int(float(mscsv[0][0]))
                    hour_diff = ((idcode - int(idcode)) - (float(mscsv[0][0]) - int(float(mscsv[0][0])))) * 100
                    if sameday:
                        print('mouse #', [mouselist[sett]], '은', hour_diff, '시간전에 학습을 시작했습니다.')
                        if hour_diff < 2.0:
                            recent_model = True
                        elif hour_diff >= 2.0:
                            recent_model = False    
                    recent_model = False # 임시로 종료   
                except:
                    recent_model = False

                # control은 추가로, exp plot이 되어있는지 확인
                if state == 'con':
                    try:
                        loadname2 = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_' + 'exp' + '_trainingSet_result.csv'
                        f = open(loadname2, 'r', encoding='utf-8')
                        f.close()
                    except:
                        print(mouselist[sett], 'exp pair 없음, control 진행을 멈춥니다.')
                        recent_model = True
                # 학습된 모델도 없고, 최근에 진행중인것도 없으니 학습 시작합니다.    
                if not(recent_model):
                    print('mouse #', [mouselist[sett]], '학습된', state, 'model 없음. 새로시작합니다.')
                    model.load_weights(initial_weightsave)
                    dt = datetime.now()
                    idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)
                        
                    # 나중에 idcode는 없애던지.. 해야될듯 
                    
                    df2 = [idcode]
                    csvfile = open(loadname, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(df2)         
                    csvfile.close() 

#                    X_training = []; # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
#                    X_valid = []; [X_valid.append([]) for i in range(msunit *fn)]
#                    Y_training_list = []
#                    Y_training_control_list = []
#                    Y_training = np.array(Y); Y_training_control = np.array(Y_control)# 여기서 뺸다
                    
                    # cross validation을 위해 test set을 제거함
                    delist = np.where(indexer[:,0]==mouselist[sett])[0]
                    
                    if mouselist[sett] in np.array(msset)[:,0]:
                        for u in np.array(msset)[np.where(np.array(msset)[:,0] == mouselist[sett])[0][0],:][1:]:
                            delist = np.concatenate((delist, np.where(indexer[:,0]==u)[0]), axis=0)
                    
                    X_training = np.delete(np.array(X), delist, 0)
                    X_valid = np.array(X)[delist]
                    Y_training_list = np.delete(np.array(Y), delist, 0)
                    Y_valid = np.array(Y)[delist]
                    
                    
                    
                
                                    
                    print('학습시작시간을 기록합니다.', df2)        
                    print('mouse #', [mouselist[sett]])
                    print('sample distributions.. ', np.round(np.mean(Y_training_list, axis = 0), 4))
                    
                    # bias 방지를 위해 동일하게 shuffle 
                    np.random.seed(seed)
                    shuffleix = list(range(X_training.shape[0]))
                    np.random.shuffle(shuffleix) 
#                    print(shuffleix)
   
                    tr_y = Y_training_list[shuffleix]
                    tr_x = X_training[shuffleix]; tr_x2 = []
                    for binss_merge in range(tr_x[0].shape[0]):
                        xtmp = []
                        for sample in range(tr_x.shape[0]):
                            xtmp2 = np.array(tr_x[sample][binss_merge,0])
                            xtmp.append(np.reshape(xtmp2, (xtmp2.shape[0],1)))
                        tr_x2.append(xtmp)
                        
                    X_valid2 = []
                    for binss_merge in range(X_valid[0].shape[0]):
                        xtmp = []
                        for sample in range(X_valid.shape[0]):
                            xtmp2 = np.array(X_valid[sample][binss_merge,0])
                            xtmp.append(np.reshape(xtmp2, (xtmp2.shape[0],1)))
                        X_valid2.append(xtmp)
                        
                    valid = tuple([X_valid2, Y_valid])


                    # 특정 training acc를 만족할때까지 epoch를 epochs단위로 지속합니다.
                    current_acc = -np.inf; cnt = -1
                    hist_save_loss = []
                    hist_save_acc = []
                    hist_save_val_loss = []
                    hist_save_val_acc = []
                                
                    
                    while current_acc < acc_thr: # 0.93: # 목표 최대 정확도, epoch limit
                        print('stop 조건을 표시합니다')
                        print('current_acc', current_acc, current_acc < acc_thr)

                        if cnt > maxepoch/epochs:
                            seed += 1
                            model, idcode = keras_setup()        
                            initial_weightsave = RESULT_SAVE_PATH + 'model//' + 'initial_weight.h5'
                            model.save_weights(initial_weightsave)
                            dt = datetime.now()
                            idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)
                            current_acc = -np.inf; cnt = -1
                            print('seed 변경, model reset 후 처음부터 다시 학습합니다.')

                        cnt += 1; print('cnt', cnt, 'current_acc', current_acc)

                        if state == 'exp':
                            current_weightsave = RESULT_SAVE_PATH + 'tmp/'+ str(idcode) + '_' + str(mouselist[sett]) + '_my_model_weights.h5'
                        elif state == 'con':
                            current_weightsave = RESULT_SAVE_PATH + 'tmp/'+ str(idcode) + '_' + str(mouselist[sett]) + '_my_model_weights_control.h5'

                        try:
                            model.load_weights(current_weightsave)
                            print('mouse #', [mouselist[sett]], cnt, '번째 이어서 학습합니다.')

                        except:
                            print('학습 진행중인 model 없음. 새로 시작합니다')

                        # control 전용, control_epochs 구하기
                        if state == 'con':
                            mscsv = []
                            f = open(loadname2, 'r', encoding='utf-8')
                            rdr = csv.reader(f)
                            for line in rdr:
                                mscsv.append(line)
                            f.close()    
                            mscsv = np.array(mscsv)
                            control_epochs = mscsv.shape[1]

                        hist = model.fit(tr_x2, tr_y, batch_size = batch_size, epochs = epochs, validation_data = valid)
                        hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                        hist_save_val_loss += list(np.array(hist.history['val_loss']))
                        hist_save_val_acc += list(np.array(hist.history['val_accuracy'])) 


                        model.save_weights(current_weightsave)
                        
                        # 종료조건: 
                        current_acc = np.min(hist_save_acc[-int(epochs*0.2):]) 
                        
#                        if state == 'con':
#                            current_acc = np.inf

#                        if cnt > 2 and current_acc < 0.7:
#                            # 700 epochs 후에도 학습이 안되고 있다면 초기화
#                            print('고장남.. 초기화')
#                            cnt = np.inf
                    
                    model.save_weights(final_weightsave)   
                    print('mouse #', [mouselist[sett]], 'traning 종료, final model을 저장합니다.')

                    # hist 저장      
                    plt.figure();
                    mouseNum = mouselist[sett]
                    plt.plot(hist_save_loss, label= '# ' + str(mouseNum) + ' loss')
                    plt.plot(hist_save_acc, label= '# ' + str(mouseNum) + ' acc')
                    plt.legend()
                    plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_trainingSet_result.png')
                    plt.close()

                    savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_trainingSet_result.csv'
                    csvfile = open(savename, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(hist_save_acc)
                    csvwriter.writerow(hist_save_loss)
                    csvfile.close()

                    if validation_sw and state == 'exp':
                        plt.figure();
                        mouseNum = mouselist[sett]
                        plt.plot(hist_save_val_loss, label= '# ' + str(mouseNum) + ' loss')
                        plt.plot(hist_save_val_acc, label= '# ' + str(mouseNum) + ' acc')
                        plt.legend()
                        plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_validationSet_result.png')
                        plt.close()

                        savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_validationSet_result.csv'
                        csvfile = open(savename, 'w', newline='')
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(hist_save_val_acc)
                        csvwriter.writerow(hist_save_val_loss)
                        csvfile.close()

            ####### test 구문 입니다. ##########        
            
            # 단일 cv set에서 대해서 기본적인 test list는 cv training 에서 제외된 data가 된다.
            # 단, training에 전혀 사용하지 않는 set = etc set에 대해서는 모든 training set으로 cv training 후,
            # 모든 etc set에 대해서 test 하므로, 이 경우 test list는 모든 etc set이 된다. 
            
            testlist = []
            testlist = [mouselist[sett]]
            
            if mouselist[sett] in np.array(msset)[:,0]:
                for u in np.array(msset)[np.where(np.array(msset)[:,0] == mouselist[sett])[0][0],:][1:]:
                    testlist.append(u)
 
            if not(len(etc) == 0):
                if etc[0] == mouselist[sett]:
                    print('test ssesion, etc group 입니다.') 
                    testlist = list(etc)
            
            if state == 'exp':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
            elif state == 'con':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final_control.h5'

            trained_fortest = False
            print(final_weightsave)
            try:
                model.load_weights(final_weightsave)
                trained_fortest =  True
                print('trained_fortest', trained_fortest)
            except:
                trained_fortest = False
                print('trained_fortest', trained_fortest)
        
            ####### test - binning 구문 입니다. ##########, test version 2
            # model load는 cv set 시작에서 무조건 하도록 되어 있음.
            if trained_fortest and testsw2:   
                for test_mouseNum in testlist:
                    testbin = None
                    picklesavename = RESULT_SAVE_PATH + 'exp_raw/' + 'PSL_result_' + str(test_mouseNum) + '.pickle'
                    try:
                        with open(picklesavename, 'rb') as f:  # Python 3: open(..., 'rb')
                            tmp = pickle.load(f)
                            testbin = False
                            print('PSL_result_' + str(test_mouseNum) + '.pickle', '이미 존재합니다. skip')
                    except:
                        testbin = True
                    
                    if testbin:
                        PSL_result_save = []
                        [PSL_result_save.append([]) for i in range(N)]
                        
                        for SE2 in range(N):
                            [PSL_result_save[SE2].append([]) for i in range(5)]
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
                            
                            i = 54; ROI = 0
                            for i in range(binNum):         
                                signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                                ROInum = signalss_PSL_test.shape[1]
                                
                                [PSL_result_save[test_mouseNum][se][i].append([]) for k in range(ROInum)]
                                for ROI in range(ROInum):
                                    mannual_signal = signalss_PSL_test[:,ROI]
                                    mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                    
                                    mannual_signal2 = movement_syn[test_mouseNum][se][binning[i]:binning[i]+full_sequence] # 반복이지만.. 편의상
                                    mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                
                                    X, _, _, _ = dataGeneration(test_mouseNum, se, label=0, \
                                           Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                                        
                                    X_array = array_recover(X)
                                    print(test_mouseNum, se, 'BINS', i ,'/', binNum, 'ROI', ROI)
                                    prediction = model.predict(X_array)
                                    PSL_result_save[test_mouseNum][se][i][ROI] = prediction
                    
    #                    msdata = {'PSL_result_save' : PSL_result_save}
                        
                        with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump(PSL_result_save, f, pickle.HIGHEST_PROTOCOL)
                            print(picklesavename, '저장되었습니다.')
# In[]      # mean signal 처리
            if trained_fortest and testsw2:
                for test_mouseNum in testlist:
                    testbin = None
                    picklesavename = RESULT_SAVE_PATH + 'exp_raw/' + 'PSL_result_mean_' + str(test_mouseNum) + '.pickle'
                    try:
                        with open(picklesavename, 'rb') as f:  # Python 3: open(..., 'rb')
                            tmp = pickle.load(f)
                            testbin = False
                            print('PSL_result_mean_' + str(test_mouseNum) + '.pickle', '이미 존재합니다. skip')
                    except:
                        testbin = True
                        
#                    testbin = True # 수정 될대까지 오버라이트
                    if testbin:
                        PSL_result_save_mean = []
                        [PSL_result_save_mean.append([]) for i in range(N)]
                        
                        for SE2 in range(N):
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
                                                           
                            [PSL_result_save_mean[test_mouseNum][se].append([]) for i in range(binNum)]
                            
                            i = 54; ROI = 0; msreport = []
                            for i in range(binNum):         
                                signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+full_sequence]
                                
                                mannual_signal = np.mean(signalss_PSL_test, axis=1)
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                
                                mannual_signal2 = movement_syn[test_mouseNum][se][binning[i]:binning[i]+full_sequence] # 반복이지만.. 편의상
                                mannual_signal2 = np.reshape(mannual_signal2, (mannual_signal2.shape[0], 1))
                            
                                X, _, _, _ = dataGeneration(test_mouseNum, se, label=0, \
                                           Mannual=True, mannual_signal=mannual_signal, mannual_signal2=mannual_signal2)
                                    
                                X_array = array_recover(X)
#                                print(test_mouseNum, se, 'BINS', i ,'/', binNum, 'mean signal')
                                prediction = model.predict(X_array)
                                PSL_result_save_mean[test_mouseNum][se][i] = prediction
                                msreport.append(np.mean(prediction[:,1]))
                                
                            print(test_mouseNum, se, np.mean(msreport))
     
                    
                        with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump(PSL_result_save_mean, f, pickle.HIGHEST_PROTOCOL)
                            print(picklesavename, '저장되었습니다.')


# In[]


















