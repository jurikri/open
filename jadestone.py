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
# library import
import pickle # python 변수를 외부저장장치에 저장, 불러올 수 있게 해줌
import os  # 경로 관리
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
    

with open('pointSave.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    pointSave = pickle.load(f)
    
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
# preprocessing setup

# preprecessing 사용자정의함수 선언
def preprocessing(endpoint=False , mannualsw=False):
    # mannual setting
    SE = 0; se = 1
    signalss_semi = []
    for SE in range(N):
        signalss_semi.append([])
        for se in range(5):
            signal = np.array(signalss[SE][se])
            s = 0
            
            if not(endpoint):
                e = signal.shape[0] 
            elif endpoint:
                e = endpoint # 497 # 첫 497만  쓴다.
             
            mstmp = signal[s:e,:]
            signalss_semi[SE].append(mstmp)
            
    return signalss_semi

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
SE = 0; se = 1; label = 1; roiNum=None; GAN=False; Mannual=False; mannual_signal=None
def dataGeneration(SE, se, label, roiNum=None, bins=bins, GAN=False, Mannual=False, mannual_signal=None):    
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
        signal_full = np.array(signalss_497[SE][se])
        
    signal_full_roi = np.mean(signal_full[:,s:e], axis=1) # 단일 ROI만 선택하는 것임
    
    if GAN:
        signal_full = np.array(GAN_data[SE][se])
        signal_full_roi = np.mean(signal_full[:,s:e], axis=1)
    
    lastsave = np.zeros(msunit, dtype=int)
    lastsave2 = np.zeros(msunit, dtype=int) # 체크용
    
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

    return X, Y, Z

# 최소길이 찾기
msvalue = []
for SE in range(N):
    for se in range(5):
        signal = np.array(signalss[SE][se])
        msvalue.append(signal.shape[0])

full_sequence = np.min(msvalue)
print('full_sequence', full_sequence, 'frames')

signalss_497 = preprocessing(endpoint=int(full_sequence))

msunit = 6 # input으로 들어갈 시계열 길이 및 갯수를 정함. full_sequence기준으로 1/n, 2/n ... n/n , n/n

sequenceSize = np.zeros(msunit) # 각 시계열 길이들을 array에 저장
for i in range(msunit):
    sequenceSize[i] = int(full_sequence/6*(i+1))
sequenceSize = sequenceSize.astype(np.int)

print('full_sequence', full_sequence)
print('sequenceSize', sequenceSize)
        
# test version 2 저장용 최소길이 사전 계산
lensave = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        mslen = np.array(signalss[SE][se]).shape[0]
        binlist = list(range(0, mslen-497, bins))
  
        lensave[SE,se] = len(binlist)

print('in data set, time duration', set(lensave.flatten()))
print('다음과 같이 나누어서 처리합니다')
msshort = 2-1+42; mslong = 55-1+42
print('msshort', msshort, ', mslong', mslong)

# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# #############

# training set에 사용될 group 설정
# painGroup, nonpainGroup 변수를 이용해서 설정해야 뒤에 data generation과 연동된다.
# etc는 trainig set으로는 사용되지 않고, 단지 etc test를 위해 모든 trainig set으로 돌리기 위해 예외적으로 추가한다.

#painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup'] \
#+ msGroup['capsaicinGroup'] + msGroup['pslGroup']
#nonpainGroup = msGroup['salineGroup']
#etc = msGroup['lidocaineGroup'][0]
  
###############
# hyperparameters #############

# learning intensity
epochs = 50 # epoch 종료를 결정할 최소 단위.
lr = 2e-3 # learning rate

n_hidden = int(12 * 1) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(12 * 1) # fully conneted laye node 갯수 # 8

duplicatedNum = 1
mspainThr = 0.305
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regularization
l2_rate = 0.2 # regularization 상수
dropout_rate1 = 0.10 # dropout late
dropout_rate2 = 0.10 # 나중에 0.1보다 줄여서 test 해보자

testsw = False  # test 하지 않고 model만 저장함. # cloud 사용량을 줄이기 위한 전략.. 
trainingsw = True # training 하려면 True 
statelist = ['exp'] # ['exp', 'con']  # random shuffled control 사용 유무
validation_sw = True # 시각화목적으로만 test set을 validset으로 배치함.
testsw2 = False

acc_thr = 0.95 # 0.93 -> 0.94
batch_size = 500 # 5000
###############

# constant 
maxepoch = 5000
n_in =  1 # number of features
n_out = 2 # number of class # 20191104: 3 class로 시도
classratio = 1 # class under sampling ratio

project_list = []
 # proejct name, seed
project_list.append(['1111_2class', 1])
#project_list.append(['1015_binfix_2', 2])
#project_list.append(['1029_adding_essential_1', 1])
#project_list.append(['0903_seeding_4', 4])
#project_list.append(['0903_seeding_5', 5])

q = project_list[0]
for q in project_list:
    settingID = q[0]; seed = q[1]
    print('settingID', settingID, 'seed', seed)

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


    # preprocessing 시작
    # 각 class의 data 입력준비
    X_save = []; Y_save = []; Z_save = []
#    X_save2 = []; Y_save2 = []; Z_save2 = [];
    for classnum in range(n_out):
        X_save.append([]); Y_save.append([]); Z_save.append([])
        
#        X_save2.append([])
#        Y_save2.append([])
#        Z_save2.append([])


    # 각 class의 data 입력조건설정
    formalin_painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
    nonpainGroup = salineGroup + lidocaineGroup
    all_painGroup = formalin_painGroup + capsaicinGroup + pslGroup

#    for SE in range(N):
#        for se in range(5):     
#            # nonpain
#            c1 = SE in formalin_painGroup and se in [0,2,4] # baseline, interphase, recorver
#            c2 = SE in capsaicinGroup and se in [0,2]
#            c3 = SE in pslGroup and se in [0]            
#            if SE in nonpainGroup or c1 or c2 or c3:
#                msclass = 0 # nonpain
##                
##                for ROI in range(signalss[SE][se].shape[1]):
#                X, Y, Z = dataGeneration(SE, se, label = msclass) 
#                X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
#
#            if (SE in all_painGroup and se == 1) or (SE in pslGroup and se == 2): # 1, 26은 특별히 제외함. 
#                msclass = 1 # pain
#                
##                for ROI in range(signalss[SE][se].shape[1]):
#                X, Y, Z = dataGeneration(SE, se, label = msclass)
#                X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
    #            
    #        if (SE == 60 and se == 0) or (SE == 61 and se == 2): # capsacine 특이 케이스 
    #            msclass = 0 # nonpain
    #            X, Y, Z = dataGeneration(SE, se, label = msclass)
    #            X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
                
    # class 별로 sample수 맞추기, 최소 갯수 기준으로 넘치는 class는 random sampling 한다. 
    
    # GAN data import
    
    if False:
        GAN_loadpath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\GAN\\GAN_data\\'
        classlabel = ['notpain', 'pain']
        
        GAN_data = []
        f = open(GAN_loadpath + classlabel[msclass] + '.csv', 'r', encoding='utf-8')
        rdr = csv.reader(f)
        for line in rdr:
            GAN_data .append(line)
        f.close()
        GAN_data  = np.array(GAN_data )
    
        for msclass in range(n_out):
            for dataNum in range(GAN_data.shape[0]):
                X, Y, Z = dataGeneration(SE, se, label=msclass, GAN=True)
                X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
    
    # In[] down sampling

    def ms_sampling():
        datasetX = []; datasetY = []; datasetZ = []
#    X_save2 = []; Y_save2 = []; Z_save2 = [];
        for classnum in range(n_out):
            datasetX.append([]); datasetY.append([]); datasetZ.append([])
        
        # lable로 최소 아픔 thr 를 결정해보자 
        msclass = 1 # pain
        ixsave = []
        valsave = []
        
        for SE in range(N):
            for se in range(5):      
                # pain Group에 들어갈 수 있는 모든 경우의 수 
                if (SE in formalin_painGroup and se in [1,3]) or (SE in pslGroup and se in [1,2]) \
                or (SE in capsaicinGroup and se in [1]): 
                    
                    tmp = pointSave[SE][se]
                    for BIN in range(len(tmp)):
                        valsave.append(tmp[BIN])
                        ixsave.append([SE,se,BIN])
        
        axiss = []; [axiss.append([]) for u in range(4)]
        for painThr in np.arange(0, 1.01, 0.05):
            painIx = np.array(valsave) > painThr
            
            originalCnt = 0; seletedCnt = 0 # 전체 중에 몇 프로 인지
            originalCnt2 = 0; seletedCnt2 = 0 # 쥐가 다 들어가긴 했는지 - 1로 맞추길 권장함
            seletedCnt3 = [] # 한 쥐가 얼마나 중복되서 들어가는지
            
            for SE2 in pslGroup:
                originalCnt += np.sum(np.array(ixsave)[:,0] == SE2)
                seletedCnt += np.sum(np.array(ixsave)[painIx][:,0] == SE2)
                originalCnt2 += np.sum(np.array(ixsave)[:,0] == SE2) > 0
                seletedCnt2 += np.sum(np.array(ixsave)[painIx][:,0] == SE2) > 0
                seletedCnt3.append(np.sum(np.array(ixsave)[painIx][:,0] == SE2))
            
            axiss[0].append(painThr)
            axiss[1].append(seletedCnt/originalCnt)
            axiss[2].append(seletedCnt2/originalCnt2)
            axiss[3].append(np.mean(seletedCnt3))
            
#            print(painThr, seletedCnt2/originalCnt2)
            
        if False:
            plt.figure()
            plt.plot(axiss[0], axiss[1])
            plt.plot(axiss[0], axiss[2])
            plt.figure()
            plt.plot(axiss[0], axiss[3])
        
        painThr = mspainThr # 어림 짐작
        painIx = np.array(valsave) > painThr
        painIx2 = np.array(painIx)
        
        # painThr를 위한 시각화 체크
        if False:
            axiss = []; [axiss.append([]) for u in range(3)]
            for painThr in np.arange(0,1,0.05):
                mspain = 0
                msnonpain = 0
                for SE in pslGroup:
                    for se in range(3):
                        pain_assume = np.sum(pointSave[SE][se] > painThr)
                        if se == 0:
                            mspain += pain_assume
                        elif se in [1,2]:
                            msnonpain += pain_assume
                            plt.figure()
                            plt.title(str(SE) + '_' + str(se))
                            plt.plot(np.array(pointSave[SE][se]))
                            
                axiss[0].append(painThr)
                axiss[1].append(mspain)
                axiss[2].append(msnonpain)
            
            plt.figure()
            plt.plot(axiss[0], axiss[1])
            plt.plot(axiss[0], axiss[2])
                        
        
#        duplicatedNum = duplicatedNum # round(len(formalin_painGroup)/len(pslGroup))
        print('duplicatedNum', duplicatedNum)
        
        for SE2 in range(N):
            for se in range(5):
                selfix = ((np.array(ixsave)[:,0] == SE2) * (np.array(ixsave)[:,1] == se))
                selfout = selfix == False
                
                if np.mean(selfix) == 0.0:
                    continue
                
                for k in  np.argsort(np.array(valsave) * (selfix * painIx))[::-1][:duplicatedNum]:
                    selfout[k] = True
                
                painIx2 = painIx2 * selfout
    
        X_tmp = []; Y_tmp = []; Z_tmp = []
        painindex_class1 = np.array(ixsave)[painIx2]
        print('painindex_class1')
        print(painindex_class1)
        for i in painindex_class1:
            SE = i[0]; se = i[1]; BINS = i[2]
            
            startat = int(BINS*bins)
            mannual_signal = signalss[SE][se][startat:startat+497,:]
            X, Y, Z = dataGeneration(SE, se, label = msclass, Mannual=True, mannual_signal=mannual_signal)
            X_tmp += X; Y_tmp += Y; Z_tmp += Z
            
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
        
        # 그룹별 사용 현황 print
        fe = 0
        fl = 0
        p3 = 0
        p10 = 0
        c = 0
#        t = len(painindex_class1)
        for j in painindex_class1:
            SE = j[0]; se = j[1]
        
            if (SE in formalin_painGroup and se in [1]):
                fe += 1
            elif (SE in formalin_painGroup and se in [3]):
                fl += 1
            elif (SE in pslGroup and se in [1]):
                p3 += 1
            elif (SE in pslGroup and se in [2]):
                p10 += 1
            elif (SE in capsaicinGroup and se in [1]):
                c += 1
                
        print('fe', fe, '/', len(formalin_painGroup))
        print('fl', fl, '/', len(formalin_painGroup))
        print('p3', p3, '/', len(pslGroup))
        print('p10', p10, '/', len(pslGroup))
        print('c', c, '/', len(capsaicinGroup))
                
#                
#            (SE in pslGroup and se in [1,2]) \
#             (SE in capsaicinGroup and se in [1]): 
#            
#            for foramlinGroup
#            painindex_class1
        
        
#        datasetX[msclass] = np.concatenate((np.array(X_tmp),np.array(X_tmp)), axis=0)
#        datasetY[msclass] = np.concatenate((np.array(Y_tmp),np.array(Y_tmp)), axis=0)
#        datasetZ[msclass] = np.concatenate((np.array(Z_tmp),np.array(Z_tmp)), axis=0)
        
        sampleNum = round(len(datasetX[msclass]) * classratio); print('sampleNum', sampleNum)
        
        # nonpain        
        msclass = 0
        ixsave = []
        valsave = []
        
        for SE in range(N):
            for se in range(5):
                c1 = SE in formalin_painGroup and se in [0,2,4] # baseline, interphase, recorver
                c2 = SE in capsaicinGroup and se in [0,2]
                c3 = SE in pslGroup and se in [0]
                c4 = SE in shamGroup and se in [0,1,2]
                
                if SE in nonpainGroup or c1 or c2 or c3 or c4:# 1, 26은 특별히 제외함. 
                    tmp = pointSave[SE][se]
                    for BIN in range(len(tmp)):
                        valsave.append(tmp[BIN])
                        ixsave.append([SE,se,BIN])
                        
        for painThr in np.arange(0, 1.01, 0.0005):
#            print('painThr', painThr)
            painIx = np.array(valsave) > painThr
            painIx2 = np.array(painIx)
            
            for SE2 in range(N):
                for se in range(5):
                    selfix = ((np.array(ixsave)[:,0] == SE2) * (np.array(ixsave)[:,1] == se))
                    selfout = selfix == False
                    
                    if np.mean(selfix) == 0.0:
                        continue
                    
                    for k in  np.argsort(np.array(valsave) * (selfix * painIx))[::-1][:duplicatedNum]:
                        selfout[k] = True
                    
                    painIx2 = painIx2 * selfout
            
            nonpain_sampleNum = np.sum(painIx2)
#            print(painThr, nonpain_sampleNum)
#            print(painThr, 'nonpain_sampleNum * 42', nonpain_sampleNum * 42)

            if nonpain_sampleNum * 42 < sampleNum:
                print('nonpain thr at', painThr, '#', nonpain_sampleNum * 42)
                break
            
        X_tmp = []; Y_tmp = []; Z_tmp = []
        painindex_class0 = np.array(ixsave)[painIx2]
        print('painindex_class0')
        print(painindex_class0)
        for i in painindex_class0:
            SE = i[0]; se = i[1]; BINS = i[2]
            
            startat = int(BINS*bins)
            mannual_signal = signalss[SE][se][startat:startat+497,:]
            X, Y, Z = dataGeneration(SE, se, label = msclass, Mannual=True, mannual_signal=mannual_signal)
            X_tmp += X; Y_tmp += Y; Z_tmp += Z
            
        datasetX[msclass] = X_tmp; datasetY[msclass] = Y_tmp; datasetZ[msclass] = Z_tmp
#        painIx2_class0 = np.array(painIx2)
        
        # pain duplicate
        
        
        return datasetX, datasetY, datasetZ
    
    
 
    X_save2, Y_save2, Z_save2 = ms_sampling()
#    painindex_classs = np.concatenate((painindex_class0, painindex_class1), axis=0)
    #  datasetX = X_save; datasetY = Y_save; datasetZ = Z_save
    
    for i in range(n_out):
        print('class', str(i),'sampling 이후', np.array(X_save2[i]).shape[0])
        
    # In[]


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
        
    def keras_setup():
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        
        dt = datetime.now()
        idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

        #init = initializers.glorot_normal(seed=None)
        
        try:
            model.reset_states()
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

    # 학습 순서 무작위 배치, seed none으로 설정함.

    trainingset = list(grouped_total_list)
    etc = []
    for SE in trainingset:
        c1 = SE in lowGroup + restrictionGroup # 둘 빼 고 
        c2 = np.sum(indexer[:,0]==SE) == 0 # 옥으로 전혀 선택되지 않았다면 test set으로 빼지 않음
        if c1 or c2:
            trainingset.remove(SE)
        if c2:
            etc.append(SE)
    mouselist = list(trainingset)
    mouselist.sort()
    mouselist.append(etc[0])

    # 학습할 set 결정, 따로 조작하지 않을 땐 mouselist로 설정하면 됨.
    wanted = [70] # mouselist # mouselist #highGroup + midleGroup + [etc[0]] # 작동할것을 여기에 넣어 
    mannual = [] # 절대 아무것도 넣지마 

    print('wanted', wanted)
    for i in wanted:
        try:
            mannual.append(np.where(np.array(mouselist)==i)[0][0])
        except:
            print(i, 'is excluded, etc group을 확인하세요.')
            
    np.random.seed(seed+1)
    shuffleix = list(range(len(mannual)))
    np.random.shuffle(shuffleix)
    print('shuffleix', shuffleix)
    mannual = np.array(mannual)[shuffleix]
    #print('etc ix', np.where(np.array(mouselist)== etc)[0])
    # 구지 mannual을 두고 다시 indexing 하는 이유는, 인지하기 편하기 때문임. 딱히 안써도 됨
    
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
    
    
    savename4 = RESULT_SAVE_PATH + 'model/' + '00_model_save_hyper_parameters.csv'
    
    if not (os.path.isfile(savename4)):
        print(settingID, 'prameter를 저장합니다. prameter를 저장합니다. prameter를 저장합니다.')
        csvfile = open(savename4, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        for row in range(len(save_hyper_parameters)):
            csvwriter.writerow(save_hyper_parameters[row])
        csvfile.close()
        
    # validation 개선용
    valsave = []; ixsave= []
    for SE in range(N):
        if SE in grouped_total_list and SE not in (restrictionGroup + lowGroup):
            for se in range(3):
                tmp = pointSave[SE][se]
                for BIN in range(len(tmp)):
                    valsave.append(tmp[BIN])
                    ixsave.append([SE,se,BIN])
    
    
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
                    model, idcode = keras_setup() # 시작과 함께 weight reset 됩니다.

                    df2 = [idcode]
                    csvfile = open(loadname, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(df2)         
                    csvfile.close() 

#                     validation set을 사용할경우 준비합니다.
                    if validation_sw and state == 'exp': # control은 validation을 볼 필요가없다.
                        init = True
                        totalROI = signalss[mouselist[sett]][0].shape[1]#; painIndex = 1
                        X_all = []; [X_all.append([]) for i in range(msunit)]
                        for se in range(3):
                            label = 0
                            if mouselist[sett] in pslGroup and se in [1,2]:
                                label = 1

                            SEindex = np.array(ixsave)[:,0] == mouselist[sett]
                            seindex = np.array(ixsave)[:,1] == se
                            valsave2 = np.array(valsave)
                            valsave2[(SEindex * seindex) == False] = np.nan
                            msbins = [np.array(ixsave)[np.nanargmax(valsave2),2]]
          
                            for BINS in msbins:
                                for ROI in range(totalROI):
                                    startat = int(BINS*bins) # bins = 10
                                    mannual_signal = signalss[mouselist[sett]][se][startat:startat+497,:]
                                
                                    unknown_data, Y_val, Z = \
                                    dataGeneration(mouselist[sett], se, roiNum=ROI, label = label, Mannual=True, mannual_signal=mannual_signal)
                                    Z = np.array(Z); tmpROI = np.zeros((Z.shape[0],1)); tmpROI[:,0] = ROI
                                    Z = np.concatenate((Z, tmpROI), axis = 1) # Z에 SE, se + ROI 정보까지 저장
    
                                    unknown_data_toarray = array_recover(unknown_data)
    
                                    if init:
                                        for k in range(msunit):
                                            X_all[k] = np.array(unknown_data_toarray[k])    
                                        Z_all = np.array(Z); Y_all = np.array(Y_val)
                                        init = False
    
                                    elif not(init):
                                        for k in range(msunit):
                                            X_all[k] = np.concatenate((X_all[k],unknown_data_toarray[k]), axis=0); 
                                        Z_all = np.concatenate((Z_all,Z), axis=0); Y_all = np.concatenate((Y_all, np.array(Y_val)), axis=0)
                                            
                                        # Z는 안쓰는데,, 걍 복붙이라 남아있는듯? 
                        valid = tuple([X_all, Y_all])

                    # training set을 준비합니다. cross validation split 
                    
                    X_training = []; [X_training.append([]) for i in range(msunit)] # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
                    X_valid = []; [X_valid.append([]) for i in range(msunit)]
                    Y_training_list = []
                    Y_training_control_list = []
#                    Y_training = np.array(Y); Y_training_control = np.array(Y_control)# 여기서 뺸다
                    
                    delist = np.where(indexer[:,0]==mouselist[sett])[0] # index는 각 data의 [SE, se]를 저장하고 있음
                    for unit in range(msunit): # input은 msunit 만큼 병렬구조임. for loop으로 각자 계산함
                        X_training[unit] = np.delete(np.array(X[unit]), delist, 0)
#                        X_valid[unit] = np.array(X[unit])[delist]
                
                    Y_training_list = np.delete(np.array(Y), delist, 0)
                    Y_training_control_list = np.delete(np.array(Y_control), delist, 0)
                    Y_valid = np.array(Y)[delist]
                    
#                    valid = tuple([X_valid, Y_valid])
                    
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
                            model, idcode = keras_setup()
                            current_acc = -np.inf; cnt = -1
                            print('model reset 후 처음부터 다시 학습합니다.')

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
                        
                        if validation_sw and state == 'exp':
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = int(epochs/2)-1)
                            hist_save_loss.append(np.array(hist.history['loss'])); hist_save_acc.append(np.array(hist.history['accuracy']))
                            hist_save_val_loss.append(np.array(hist.history['val_loss'])); hist_save_val_acc.append(np.array(hist.history['val_accuracy'])) 
                            
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1, validation_data = valid)
                            hist_save_loss.append(np.array(hist.history['loss'])); hist_save_acc.append(np.array(hist.history['accuracy']))
                            
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = int(epochs/2)-1)
                            hist_save_loss.append(np.array(hist.history['loss'])); hist_save_acc.append(np.array(hist.history['accuracy']))
                            
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = 1, validation_data = valid)
                            hist_save_loss.append(np.array(hist.history['loss'])); hist_save_acc.append(np.array(hist.history['accuracy']))
                            
                        elif not(validation_sw) and state == 'exp': 
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs) #, validation_data = valid)
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
            if not(etc[0] == mouselist[sett]):
                testlist = [mouselist[sett]]
                print('test ssesion, mouse #', [mouselist[sett]], '입니다.')
            elif etc[0] == mouselist[sett]:
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
            
            # test version 1, 20191017 현재 version 2가 범용적이므로 v1 은 사용하지 않음.
#            if testsw:
#                for test_mouseNum in testlist:
#                    print('mouse #', test_mouseNum, '에 대한 기존 test 유무를 확인합니다.')
#                    #    test 되어있는지 확인.
#
#                    if state == 'exp':
#                        savename = RESULT_SAVE_PATH + 'exp_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
#                    elif state == 'con':
#                        savename = RESULT_SAVE_PATH + 'control_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
#
#                    tested = False
#                    print(savename)
#                    try:
#                        csvfile = open(savename, 'r', newline='')
#                        tested = True
#                        print('tested', tested)
#                    except:
#                        tested = False
#                        print('tested', tested)
#
#                    if not(tested) and trained_fortest: 
#                        print('mouse #', test_mouseNum, 'test 진행')
#                        totalROI = signalss[test_mouseNum][0].shape[1]; painIndex = 1
#                        X_all = []; [X_all.append([]) for i in range(msunit)]
#
#                        for se in range(5):
#                            for ROI in range(totalROI):
#                                unknown_data, Y_val, Z = dataGeneration(test_mouseNum, se, label=1, roiNum = ROI)
#                                Z = np.array(Z); tmpROI = np.zeros((Z.shape[0],1)); tmpROI[:,0] = ROI
#                                Z = np.concatenate((Z, tmpROI), axis = 1)    
#
#                                unknown_data_toarray = array_recover(unknown_data)
#
#                                if se == 0 and ROI == 0:
#                                    for k in range(msunit):
#                                        X_all[k] = np.array(unknown_data_toarray[k])    
#                                    Z_all = np.array(Z); Y_all = np.array(Y_val)
#
#                                elif not(se == 0 and ROI == 0):
#                                    for k in range(msunit):
#                                        X_all[k] = np.concatenate((X_all[k],unknown_data_toarray[k]), axis=0); 
#                                    Z_all = np.concatenate((Z_all,Z), axis=0); Y_all = np.concatenate((Y_all, np.array(Y_val)), axis=0)
#
#                        prediction = model.predict(X_all)
#
#                        df1 = np.concatenate((Z_all,prediction), axis=1)
#                        df2 = [['SE', 'se', 'nonpain', 'pain']]; se = 0 # 최종결과 (acc) 저장용
#
#                        # [SE, se, ROI, nonpain, pain]
#                        for se in range(5):
#                            predicted_pain = np.mean(df1[:,painIndex+3][np.where(df1[:,1]==se)[0]] > 0.5)
#                            mspredict = [1-predicted_pain, predicted_pain] # 전통을 중시...
#
#                            df2.append([[test_mouseNum], se] + mspredict)
#
#                        for d in range(len(df2)):
#                            print(df2[d])
#
#                        # 최종평가를 위한 저장 
#                        # acc_experiment 저장
#                        if state == 'exp':
#                            savename = RESULT_SAVE_PATH + 'exp/' + 'biRNN_acc_' + str(test_mouseNum)  + '.csv'
#                        elif state == 'con':
#                            savename = RESULT_SAVE_PATH + 'control/' + 'biRNN_acc_' + str(test_mouseNum)  + '.csv'
#
#                        csvfile = open(savename, 'w', newline='')
#                        csvwriter = csv.writer(csvfile)
#                        for row in range(len(df2)):
#                            csvwriter.writerow(df2[row])
#                        csvfile.close()
#
#                        # raw 저장
#                        if state == 'exp':
#                            savename = RESULT_SAVE_PATH + 'exp_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
#                        elif state == 'con':
#                            savename = RESULT_SAVE_PATH + 'control_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
#
#                        csvfile = open(savename, 'w', newline='')
#                        csvwriter = csv.writer(csvfile)
#                        for row in range(len(df1)):
#                            csvwriter.writerow(df1[row])
#                        csvfile.close()
#                        
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
                            
                            binning = list(range(0,(signalss[test_mouseNum][se].shape[0] - 497) +1, bins))
                            binNum = len(binning)
                            
                            # dataGeneration _ modify
                            
                            binlist = list(range(0, full_sequence-np.min(sequenceSize), bins))
                            minimum_binning = len(binlist)
                                    
                            if binNum > 0 and binNum < mslong +1 -42  : # for 2 mins
                                binNum2 = msshort-minimum_binning+1
                                print(SE, se, 'msshort', binNum2)
                            elif binNum >= mslong +1 -42: # for 4 mins
                                binNum2 = mslong-minimum_binning+1
                                print(SE, se, 'mslong', binNum2)
                            elif binNum == 0:
                                print(SE, se, '예상되지 않은 길이입니다. 체크')
                                import sys
                                sys.exit()
                            else: # for 2 mins
                                print(SE, se, '예상되지 않은 길이입니다. 체크')
                                
                            [PSL_result_save[test_mouseNum][se].append([]) for i in range(binNum2)]
                            
                            i = 54; ROI = 0
                            for i in range(binNum2):         
                                signalss_PSL_test = signalss[test_mouseNum][se][binning[i]:binning[i]+497]
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
                                    print(test_mouseNum, se, 'BINS', i ,'/', binNum2, 'ROI', ROI)
                                    prediction = model.predict(X_array)
                                    PSL_result_save[test_mouseNum][se][i][ROI] = prediction
                    
    #                    msdata = {'PSL_result_save' : PSL_result_save}
                        
                        with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
                            pickle.dump(PSL_result_save, f, pickle.HIGHEST_PROTOCOL)
                            print(picklesavename, '저장되었습니다.')


 

















