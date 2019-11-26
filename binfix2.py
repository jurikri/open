# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 13:22:18 2019

@author: msbak
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 12:27:10 2019

@author: msbak
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""
import os  # 경로 관리
# library import
import pickle # python 변수를 외부저장장치에 저장, 불러올 수 있게 해줌

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
behavss2 = msdata_load['behavss2'] # 투포톤과 syn 맞춰진 버전 
movement = msdata_load['movement'] # 움직인정보를 평균내서 N x 5 matrix에 저장
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

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

bins = 10 # 최소 time frame 간격     

totaldataset = highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup + \
salineGroup + pslGroup + shamGroup
        
shortlist = []; longlist = []
for SE in range(N):
    if SE in totaldataset:
        for se in range(5):
            length = np.array(signalss[SE][se]).shape[0]
            if length > 180*FPS:
                longlist.append([SE,se])
            elif length < 180*FPS:
                shortlist.append([SE,se])
            else:
                print('error')                   

msset = [[70,72],[71,84],[75,85],[76,86]]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
def array_recover(X_like):
    X_like_toarray = []; X_like = np.array(X_like)
    for input_dim in range(msunit):
        tmp = np.zeros((X_like.shape[0],X_like[0,input_dim].shape[0]))
        for row in range(X_like.shape[0]):
            tmp[row,:] = X_like[row,input_dim]
    
        X_like_toarray.append(tmp)
        
        X_like_toarray[input_dim] =  \
        np.reshape(X_like_toarray[input_dim], (X_like_toarray[input_dim].shape[0],X_like_toarray[input_dim].shape[1],1))
    
    return X_like_toarray

# data 생성
SE = 70; se = 1; label = 1; roiNum=None; GAN=False; Mannual=False; mannual_signal=None; passframesave=np.array([])
def dataGeneration(SE, se, label, roiNum=None, bins=bins, GAN=False, Mannual=False, mannual_signal=None, passframesave=np.array([])):    
    X = []; Y = []; Z = []

    if label == 0:
        label = [1, 0] # nonpain
    elif label == 1:
        label = [0, 1] # pain
#    elif label == 2:
#        label = [0, 0, 1] # nonpain low
 
    if not(roiNum==None):
        s = roiNum; e = roiNum+1
    elif roiNum==None:
        s = 0; e = signalss[SE][se].shape[1]
    
    if Mannual:
        signal_full = mannual_signal
        
    elif not(Mannual):
        signal_full = np.array(signalss_cut[SE][se])
        
    signal_full_roi = np.mean(signal_full[:,s:e], axis=1) # 단일 ROI만 선택하는 것임
    
    if GAN:
        signal_full = np.array(GAN_data[SE][se])
        signal_full_roi = np.mean(signal_full[:,s:e], axis=1)
    
    lastsave = np.zeros(msunit, dtype=int)
    lastsave2 = np.zeros(msunit, dtype=int) # 체크용
    
    binlist = list(range(0, full_sequence-np.min(sequenceSize), bins))

    if passframesave.shape[0] != 0:
        binlist = passframesave

    t4_save = []
    for frame in binlist:   
        X_tmp = []; [X_tmp.append([]) for k in range(msunit)] 

        for unit in range(msunit):
            if frame <= full_sequence - sequenceSize[unit]:
                X_tmp[unit] = (signal_full_roi[frame : frame + sequenceSize[unit]])
                lastsave[unit] = frame
                
                if unit == 0:
                    t4_save.append(np.mean(signal_full_roi[frame : frame + sequenceSize[unit]]))
                
            else:
                X_tmp[unit] = (signal_full_roi[lastsave[unit] : lastsave[unit] + sequenceSize[unit]])
#                print(frame, unit, lastsave[unit])
                if unit == 0:
                    t4_save.append(np.mean(signal_full_roi[lastsave[unit] : lastsave[unit] + sequenceSize[unit]]))
                
        if False: # 시각화로 체크 위치만
            msimg = np.zeros((msunit*10, full_sequence))
            
            for unit in range(msunit):
                if frame <= full_sequence - sequenceSize[unit]:
                    msimg[unit*10:(unit+1)*10, frame : frame + sequenceSize[unit]] = 1
                    lastsave2[unit] = frame
                    
                else:
                    msimg[unit*10:(unit+1)*10, lastsave2[unit] : lastsave2[unit] + sequenceSize[unit]] = 1
                    
            plt.figure()
            plt.imshow(msimg)
            
         # signal 자체를 시각화로 체크 
         # frame은 계속 돌려야 하기 때문에, if문을 개별적으로 설정함
        if False:
            if True and (frame == 0 or frame == 100 or frame == 300 or frame == 410):
                plt.figure(frame, figsize=(8,3))
                
            for unit in range(msunit):
                if frame <= full_sequence - sequenceSize[unit]:
                    lastsave2[unit] = frame 
                    start = frame
                    end = frame + sequenceSize[unit]
                    
                else: 
                    start = lastsave2[unit]
                    end = lastsave2[unit] + sequenceSize[unit]
                    
                if True and (frame == 0 or frame == 100 or frame == 300 or frame == 410):
                    if unit == 0:
                        ax1 = plt.subplot(msunit, 1, unit+1)
                        tmp = np.mean(signalss[SE][se], axis=1)
                        tmp[:start] = np.nan; tmp[end:] = np.nan
                        ax1.plot(tmp)
                        ax1.axes.get_xaxis().set_visible(False)
                        ax1.axes.get_yaxis().set_visible(False) 
                    else:
                        ax2 = plt.subplot(msunit, 1, unit+1, sharex = ax1)
                        tmp = np.mean(signalss[SE][se], axis=1)
                        tmp[:start] = np.nan; tmp[end:] = np.nan
                        ax2.plot(tmp)
                        ax2.axes.get_xaxis().set_visible(False)
                        ax2.axes.get_yaxis().set_visible(False)
                        
                    plt.savefig(str(frame) + '.png')
                    

        X.append(X_tmp)
        Y.append(label)
        Z.append([SE,se])

    return X, Y, Z, t4_save

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

#signalss_cut = preprocessing(endpoint=int(full_sequence))

msunit = 8 # input으로 들어갈 시계열 길이 및 갯수를 정함. full_sequence기준으로 1/n, 2/n ... n/n , n/n

sequenceSize = np.zeros(msunit) # 각 시계열 길이들을 array에 저장
for i in range(msunit):
    sequenceSize[i] = int(full_sequence/msunit*(i+1))
sequenceSize = sequenceSize.astype(np.int)

print('full_sequence', full_sequence)
print('sequenceSize', sequenceSize)

  
###############
# hyperparameters #############
 
# learning intensity
epochs = 10 # epoch 종료를 결정할 최소 단위.
lr = 1e-3 # learning rate

n_hidden = int(8 * 1) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 1) # fully conneted laye node 갯수 # 8

#duplicatedNum = 1
#mspainThr = 0.27
#acitivityThr = 0.4
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regularization
l2_rate = 0.25 # regularization 상수
dropout_rate1 = 0.20 # dropout late
dropout_rate2 = 0.10 # 나중에 0.1보다 줄여서 test 해보자

#testsw = False  # test 하지 않고 model만 저장함. # cloud 사용량을 줄이기 위한 전략.. 
trainingsw = True # training 하려면 True 
statelist = ['exp'] # ['exp', 'con']  # random shuffled control 사용 유무
validation_sw = True # 시각화목적으로만 test set을 validset으로 배치함.
testsw2 = True
#if testsw2:
##    import os
#    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#    os.environ['CUDA_VISIBLE_DEVICES'] = ''
#    import tensorflow as tf

# 집 컴퓨터, test 전용으로 수정
if savepath == 'D:\\painDecorder\\save\\tensorData\\' or savepath == 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\':
    trainingsw = False
    testsw2 = True

acc_thr = 0.94 # 0.93 -> 0.94
batch_size = 2000 # 5000
###############

# constant 
maxepoch = 300
n_in =  1 # number of features
n_out = 2 # number of class # 20191104: 3 class로 시도
classratio = 1 # class under sampling ratio

project_list = []
 # proejct name, seed
project_list.append(['1126_binfix2_saline', 3, None])
#project_list.append(['1118_direct_2_continue1', 3, '1118_direct_2'])
#project_list.append(['1122_driect_cut_continue1', 4, '1122_driect_cut'])
#project_list.append(['1015_binfix_2', 2])
#project_list.append(['1029_adding_essential_1', 1])
#project_list.append(['0903_seeding_4', 4])
#project_list.append(['0903_seeding_5', 5])

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

    # 각 class의 data 입력조건설정
#    formalin_painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
#    nonpainGroup = salineGroup + lidocaineGroup
#    all_painGroup = formalin_painGroup + capsaicinGroup + pslGroup
    
     # 학습 순서 무작위 배치, seed none으로 설정함.
     
    # 여기서 totaldataset의 정의: 전체 data set범위
    
    testset = []
    trainingset = list(totaldataset)
    for u in testset:
        try:
            trainingset.remove(u)
        except:
            pass
    
#    fortmp = np.array(trainingset)
#    for u in fortmp:
#        if not u in np.array(longlist)[:,0]:
##            print(u)
#            trainingset.remove(u)
##            print(trainingset)

    # initiate
    def ms_sampling():
        sampleNum = []; [sampleNum.append([]) for u in range(n_out)]
        
        datasetX = []; datasetY = []; datasetZ = []
#    X_save2 = []; Y_save2 = []; Z_save2 = [];
        for classnum in range(n_out):
            datasetX.append([]); datasetY.append([]); datasetZ.append([])
        
        # lable로 최소 아픔 thr 를 결정해보자 
        msclass = 1 # pain
        X_tmp = []; Y_tmp = []; Z_tmp = []
        for SE in range(N):
            if SE in trainingset:
                for se in range(5):      
                    # pain Group에 들어갈 수 있는 모든 경우의 수 
                    c1 = SE in highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup
                    c2 = se in [1]

                    if c1 and c2: # 
                        mssignal = np.mean(signalss[SE][se], axis=1)
                        msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                        
                        for u in msbins:
                            mannual_signal = mssignal[u:u+full_sequence]
                            mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                            X, Y, Z, _ = dataGeneration(SE, se, label=msclass, \
                                           Mannual=True, mannual_signal=mannual_signal)
                            X_tmp += X; Y_tmp += Y; Z_tmp += Z #; T_tmp += t4_save 
                    
        datasetX[msclass] = np.array(X_tmp)
        datasetY[msclass] = np.array(Y_tmp)
        datasetZ[msclass] = np.array(Z_tmp)
        sampleNum[msclass] = len(datasetX[msclass]); print('pain_sampleNum', sampleNum[msclass])
        
        # nonpain      
        for activityThr in np.arange(1.4,3,0.005):
            msclass = 0 # nonpain
            X_tmp = []; Y_tmp = []; Z_tmp = []; T_tmp = []
            for SE in range(N):
                if SE in trainingset:
                    for se in range(5):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        c1 = SE in highGroup + midleGroup + yohimbineGroup + ketoGroup and se in [0,2,4]
                        c2 = SE in capsaicinGroup and se in [0,2]
                        c3 = SE in pslGroup and se in [0]
                        c4 = SE in shamGroup and se in [0,1,2]
                        c5 = SE in salineGroup and se in [0,1,2,3,4]
                        
                        if c1 or c2 or c3 or c4 or c5:
                            mssignal = np.mean(signalss[SE][se], axis=1)
                            msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                            
                            for u in msbins:
                                mannual_signal = mssignal[u:u+full_sequence]
                                mannual_signal = np.reshape(mannual_signal, (mannual_signal.shape[0], 1))
                                X, Y, Z, t4_save = dataGeneration(SE, se, label=msclass, \
                                               Mannual=True, mannual_signal=mannual_signal)
                                
                                X_tmp += X; Y_tmp += Y; Z_tmp += Z; T_tmp += t4_save 
                        
            datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
            
            msix1 = (np.array(T_tmp) > activityThr) # 임의 
                  
            datasetX[msclass] = np.array(datasetX[msclass])[msix1]
            datasetY[msclass] = np.array(datasetY[msclass])[msix1] 
            datasetZ[msclass] = np.array(datasetZ[msclass])[msix1]
#            print('after filtering, total pain samples #', len(datasetX[msclass]))
#            sampleNum[1] = round(len(datasetX[1]))
#            lowactivityNum[1] = len(datasetX[1])
            
            
            
            sampleNum[msclass] = len(datasetX[msclass])
            
            print(activityThr, 'nonpain_sampleNum', sampleNum[msclass], '수동 최적화.. 확인 !')
            
            if sampleNum[msclass] < sampleNum[1]:
                break
    
        return datasetX, datasetY, datasetZ
    
    def ms_sampling_continue():
        sampleNum = []; [sampleNum.append([]) for u in range(n_out)]
        
        datasetX = []; datasetY = []; datasetZ = []
        for classnum in range(n_out):
            datasetX.append([]); datasetY.append([]); datasetZ.append([])
            
        # activity가 낮은 pain sample 제거
        # continue의 경우 복잡해 지니깐, 최종 숫자로만 생각하자.
            
#        thr = 0.2 # 옥석 thr
#        mstmp = -np.inf
        for thr in np.arange(0,1,0.0025):
            msclass = 1 # for pain
            X_tmp = []; Y_tmp = []; Z_tmp = []; T_tmp = []
            for SE in trainingset:
                if SE in trainingset:
                    loadpath5 = savepath + 'result\\' + continueSW + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
                    with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                        PSL_result_save = pickle.load(f)
                    
                    for se in range(len(PSL_result_save[SE])):
                        c1 = SE in pslGroup and se in [1,2] # pain 조건
                        c2 = [SE, se] in longlist
                        if c1 and c2:
                            for BIN in range(len(PSL_result_save[SE][se])):
                                tmp = np.array(PSL_result_save[SE][se][BIN])[:,:,1]
                                
                                roiix = np.array(range(tmp.shape[0]))[np.mean(tmp,axis=1) > thr]
                                timix = np.array(range(tmp.shape[1]))[np.mean(tmp,axis=0) > thr]
                                 
                                mannual_signal = np.array(signalss[SE][se])[0+(BIN*10):full_sequence+(BIN*10)]
                                mannual_signal = np.mean(mannual_signal[:,[roiix]], axis = 2)
                                
                                if not(len(timix) == 0):
                                    X, Y, Z, t4_save = \
                                    dataGeneration(SE, se, label=msclass, Mannual=True, \
                                                   mannual_signal=mannual_signal, passframesave=timix*bins)
                                    X_tmp += X; Y_tmp += Y; Z_tmp += Z; T_tmp += t4_save
     
            datasetX[msclass] = np.array(X_tmp)
            datasetY[msclass] = np.array(Y_tmp)
            datasetZ[msclass] = np.array(Z_tmp)
            sampleNum[msclass] = len(datasetX[msclass]); 
        
            print(sampleNum[msclass], thr)
        
            if thr == 0:
                print('pain_sampleNum', sampleNum[msclass], 'thr', thr)
#                mstmp = sampleNum[msclass]
                
            if sampleNum[msclass] < lowactivityNum[1]*0.9:
                break
                
        print('pain_sampleNum', sampleNum[msclass], 'thr', thr)
                    
        print('nonpain thr를 계산합니다.')
        for nonpainthr in np.arange(0,1,0.0025):
            msclass = 0 # nonpain
            X_tmp = []; Y_tmp = []; Z_tmp = []
            for SE in trainingset:
                if SE in trainingset:
                    loadpath5 = savepath + 'result\\' + continueSW + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
                    with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                        PSL_result_save = pickle.load(f)
                    for se in range(len(PSL_result_save[SE])):      
                        # pain Group에 들어갈 수 있는 모든 경우의 수 
                        c1 = SE in pslGroup and se in [0]
                        c2 = [SE, se] in longlist
                        c3 = SE in shamGroup and se in [0,1,2]
                        c4 = SE in highGroup and se in [0]
                        
                        if (c1 or c3 or c4) and c2:
                            for BIN in range(len(PSL_result_save[SE][se])):
                                tmp = np.array(PSL_result_save[SE][se][BIN])[:,:,1]
                            
                                roiix = np.array(range(tmp.shape[0]))[np.mean(tmp,axis=1) > nonpainthr]
                                timix = np.array(range(tmp.shape[1]))[np.mean(tmp,axis=0) > nonpainthr]
                                
                                mannual_signal = np.array(signalss[SE][se])[0+(BIN*10):full_sequence+(BIN*10)]
                                mannual_signal = np.mean(mannual_signal[:,[roiix]], axis = 2)
                                
                                if not(len(timix) == 0):
                                    X, Y, Z, _ = \
                                    dataGeneration(SE, se, label=msclass, Mannual=True, \
                                                   mannual_signal=mannual_signal, passframesave=timix*bins)
                                    X_tmp += X; Y_tmp += Y; Z_tmp += Z
                        
                        
            datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
            sampleNum[msclass] = len(datasetX[msclass]);
            
            print(nonpainthr, sampleNum[0])
                
            if sampleNum[0] < sampleNum[1]:
                break
        
        print('nonpain_sampleNum', sampleNum[msclass], 'nonpainthr', nonpainthr)
        
        msclass = np.argmax(sampleNum)
        print('higher # class is...', msclass)
        np.random.seed(seed2)
        print('random seed', seed2)
        shuffleix = list(range(len(datasetX[msclass])))
        shuffleix = np.array(random.sample(shuffleix, sampleNum[np.argmin(sampleNum)]))
   
        datasetX[msclass] = np.array(datasetX[msclass])[shuffleix]
        datasetY[msclass] = np.array(datasetY[msclass])[shuffleix]
        datasetZ[msclass] = np.array(datasetZ[msclass])[shuffleix]
  
        return datasetX, datasetY, datasetZ, t4_save
    
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
                dice = random.choice([[1,0],[0,1]])
                Y_control[identical_ix] = dice
                
    # cross validation을 위해, training / test set split            
    # mouselist는 training set에 사용된 list임.
    # training set에 사용된 mouse의 마릿수 만큼 test set을 따로 만듦

    inputsize = np.zeros(msunit, dtype=int) 
    for unit in range(msunit):
        inputsize[unit] = X[unit].shape[1] # size 정보는 계속사용하므로, 따로 남겨놓는다.
        
    def keras_setup(model = None):
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        
        dt = datetime.now()
        idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

        #init = initializers.glorot_normal(seed=None)
        
        try:
            model.reset_states()
            keras.backend.clear_session()
            print('올라와있는 model이 있었기 때문에, 초기화 하였습니다.')
        except:
            pass 
            # print('reset할 기존 model 없음')
        
        init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
        
        input1 = []; [input1.append([]) for i in range(msunit)] # 최초 input layer
        input2 = []; [input2.append([]) for i in range(msunit)] # input1을 받아서 끝까지 이어지는 변수
        
        for unit in range(msunit):
            input1[unit] = keras.layers.Input(shape=(inputsize[unit], n_in)) # 각 병렬 layer shape에 따라 input 받음
            input2[unit] = Bidirectional(LSTM(n_hidden))(input1[unit]) # biRNN -> 시계열에서 단일 value로 나감
            input2[unit] = Dense(layer_1, kernel_initializer = init, activation='relu')(input2[unit]) # fully conneted layers, relu
            input2[unit] = Dropout(dropout_rate1)(input2[unit]) # dropout
        
        added = keras.layers.Add()(input2) # 병렬구조를 여기서 모두 합침
        merge_1 = Dense(layer_1, kernel_initializer = init, activation='relu')(added) # fully conneted layers, relu
        merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
        merge_2 = Dense(n_out, kernel_initializer = init, activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
        merge_3 = Dense(n_out, input_dim=n_out, kernel_regularizer=regularizers.l2(l2_rate))(merge_2) # regularization
        merge_4 = Activation('softmax')(merge_3) # activation as softmax function
        
        model = keras.models.Model(inputs=input1, outputs=merge_4) # input output 선언
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
        
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        return model, idcode
    
    model, idcode = keras_setup()    
    
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
            etc.append(SE)
            
        c2 = np.array(msset)[:,0]
        if SE in c2:
            for u in np.array(msset)[np.where(np.array(msset)[:,0] == SE)[0][0],:][1:]:
                trainingset.remove(u)
            
    mouselist = list(trainingset)
    mouselist.sort()
    
    if not(len(etc) == 0):
        mouselist.append(etc[0])
    
    # 학습할 set 결정, 따로 조작하지 않을 땐 mouselist로 설정하면 됨.
    wanted = [etc[0]] + shamGroup + pslGroup
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
        
#    # validation 개선용
#    valsave = []; ixsave= []
#    for SE in range(N):
#        if SE in grouped_total_list and SE not in (restrictionGroup + lowGroup):
#            for se in range(3):
#                tmp = pointSave[SE][se]
#                for BIN in range(len(tmp)):
#                    valsave.append(tmp[BIN])
#                    ixsave.append([SE,se,BIN])
    
    
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
                    model, idcode = keras_setup(model) # 시작과 함께 weight reset 됩니다.

                    df2 = [idcode]
                    csvfile = open(loadname, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(df2)         
                    csvfile.close() 

#                     validation set을 사용할경우 준비합니다.
#                    if validation_sw and state == 'exp': # control은 validation을 볼 필요가없다.
#                        init = True
#                        totalROI = signalss[mouselist[sett]][0].shape[1]#; painIndex = 1
#                        X_all = []; [X_all.append([]) for i in range(msunit)]
#                        for se in range(3):
#                            if [mouselist[sett], se] in longlist:
#                                label = 0
#                                if mouselist[sett] in pslGroup and se in [1,2]:
#                                    label = 1
#    
#                                for ROI in range(totalROI):
#                                    unknown_data, Y_val, Z, _ = \
#                                    dataGeneration(mouselist[sett], se, roiNum=ROI, label=label)
#                                    Z = np.array(Z); tmpROI = np.zeros((Z.shape[0],1)); tmpROI[:,0] = ROI
#                                    Z = np.concatenate((Z, tmpROI), axis = 1) # Z에 SE, se + ROI 정보까지 저장
#    
#                                    unknown_data_toarray = array_recover(unknown_data)
#    
#                                    if init:
#                                        for k in range(msunit):
#                                            X_all[k] = np.array(unknown_data_toarray[k])    
#                                        Z_all = np.array(Z); Y_all = np.array(Y_val)
#                                        init = False
#    
#                                    elif not(init):
#                                        for k in range(msunit):
#                                            X_all[k] = np.concatenate((X_all[k],unknown_data_toarray[k]), axis=0); 
#                                        Z_all = np.concatenate((Z_all,Z), axis=0); Y_all = np.concatenate((Y_all, np.array(Y_val)), axis=0)
#                                            
#                                        # Z는 안쓰는데,, 걍 복붙이라 남아있는듯? 
#                        valid = tuple([X_all, Y_all])

                    # training set을 준비합니다. cross validation split 
                    
                    X_training = []; [X_training.append([]) for i in range(msunit)] # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
                    X_valid = []; [X_valid.append([]) for i in range(msunit)]
                    Y_training_list = []
                    Y_training_control_list = []
#                    Y_training = np.array(Y); Y_training_control = np.array(Y_control)# 여기서 뺸다

                    delist = np.where(indexer[:,0]==mouselist[sett])[0]
                    
                    if mouselist[sett] in np.array(msset)[:,0]:
                        for u in np.array(msset)[np.where(np.array(msset)[:,0] == mouselist[sett])[0][0],:][1:]:
                            delist = np.concatenate((delist, np.where(indexer[:,0]==u)[0]), axis=0)
                    
                    for unit in range(msunit): # input은 msunit 만큼 병렬구조임. for loop으로 각자 계산함
                        X_training[unit] = np.delete(np.array(X[unit]), delist, 0)
                        X_valid[unit] = np.array(X[unit])[delist]
                
                    Y_training_list = np.delete(np.array(Y), delist, 0)
                    Y_training_control_list = np.delete(np.array(Y_control), delist, 0)
                    Y_valid = np.array(Y)[delist]
                    
                    valid = tuple([X_valid, Y_valid])
                    
                    print('학습시작시간을 기록합니다.', df2)        
                    print('mouse #', [mouselist[sett]])
                    print('sample distributions.. ', np.round(np.mean(Y_training_list, axis = 0), 4))
                    
                    # bias 방지를 위해 동일하게 shuffle 
                    np.random.seed(seed)
                    shuffleix = list(range(X_training[0].shape[0]))
                    np.random.shuffle(shuffleix) 
#                    print(shuffleix)
   
                    tr_y_shuffle = Y_training_list[shuffleix]
                    tr_y_shuffle_control = Y_training_control_list[shuffleix]

                    tr_x = []
                    for unit in range(msunit):
                        tr_x.append(X_training[unit][shuffleix])


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
                            model, idcode = keras_setup(model)
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
                        
#                        # validation이 가치가없으므로 끔 
#                        validation_sw = False
                        
  
                        if validation_sw and Y_valid.shape[0] != 0 and state == 'exp':
                            #1
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1, validation_data = valid)
                            hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                            hist_save_val_loss += list(np.array(hist.history['val_loss']))
                            hist_save_val_acc += list(np.array(hist.history['val_accuracy'])) 
                            
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = int(epochs/2)-1)
                            hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                            
                            #2
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1, validation_data = valid)
                            hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                            hist_save_val_loss += list(np.array(hist.history['val_loss']))
                            hist_save_val_acc += list(np.array(hist.history['val_accuracy'])) 
                            
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = int(epochs/2)-1)
                            hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                            
                        elif (not(validation_sw) or Y_valid.shape[0] == 0) and state == 'exp': 
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs) #, validation_data = valid)
                            hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                        elif state == 'con':
                            hist = model.fit(tr_x, tr_y_shuffle_control, batch_size = batch_size, epochs = control_epochs)

                        model.save_weights(current_weightsave)
                        
                        # 종료조건: 
                        current_acc = np.min(hist_save_acc[-int(epochs*0.2):]) 
                        
                        if state == 'con':
                            current_acc = np.inf

                        if cnt > 7 and current_acc < 0.6:
                            # 700 epochs 후에도 학습이 안되고 있다면 초기화
                            print('고장남.. 초기화')
                            cnt = np.inf

                    # 학습 model 최종 저장
                    #5: 마지막으로 validation 찍음
                    if validation_sw and Y_valid.shape[0] != 0 and state == 'exp':
                        hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1, validation_data = valid)
                        hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                        hist_save_val_loss += list(np.array(hist.history['val_loss']))
                        hist_save_val_acc += list(np.array(hist.history['val_accuracy']))
                    elif (not(validation_sw) or Y_valid.shape[0] == 0) and state == 'exp': 
                        hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1) #, validation_data = valid)
                        hist_save_loss += list(np.array(hist.history['loss'])); hist_save_acc += list(np.array(hist.history['accuracy']))
                    
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
                        if test_mouseNum in capsaicinGroup or test_mouseNum in pslGroup or test_mouseNum in shamGroup:
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
                                    signal_full_roi = np.mean(signalss_PSL_test[:,ROI:ROI+1], axis=1)
                                
                                    lastsave = np.zeros(msunit, dtype=int)
                                    X_ROI = []
                                    
                                    binlist = list(range(0, full_sequence-np.min(sequenceSize), bins))
                
                                    for frame in binlist:   
                                        X_tmp = []; [X_tmp.append([]) for k in range(msunit)] 
                                            
                                        for unit in range(msunit):
                                            if frame <= full_sequence - sequenceSize[unit]:
                                                X_tmp[unit] = (signal_full_roi[frame : frame + sequenceSize[unit]])
                                                lastsave[unit] = frame
                                                
                                            else:
                                                X_tmp[unit] = (signal_full_roi[lastsave[unit] : lastsave[unit] + sequenceSize[unit]])
                                #                print(frame, unit, lastsave[unit])
                                
                                        X_ROI.append(X_tmp)
                                        
                                    X_array = array_recover(X_ROI)
                                    print(test_mouseNum, se, 'BINS', i ,'/', binNum, 'ROI', ROI)
                                    prediction = model.predict(X_array)
                                    PSL_result_save[test_mouseNum][se][i][ROI] = prediction
                    
    #                    msdata = {'PSL_result_save' : PSL_result_save}
                        
                        with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump(PSL_result_save, f, pickle.HIGHEST_PROTOCOL)
                            print(picklesavename, '저장되었습니다.')


 

















