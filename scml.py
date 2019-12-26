# msbak, 2019. 09. 02.
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import os
import random
from scipy import stats

def msGrouping_pslOnly(psl): # psl만 처리
    psldata = np.array(psl)
    
    df3 = pd.DataFrame(psldata[shamGroup,0:3]) 
    df3 = pd.concat([df3, pd.DataFrame(psldata[pslGroup,0:3]), \
                     pd.DataFrame(psldata[adenosineGroup,0:3])], ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
    except:
        savepath = ''; # os.chdir(savepath);
print('savepath', savepath)
#

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
with open('mspickle_msdict.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdict = pickle.load(f)
    msdict = msdict['msdict']
    
#with open('PSL_result_save.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
#    PSL_result_save = pickle.load(f)
#    PSL_result_save = PSL_result_save['PSL_result_save']
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']
behavss2 = msdata_load['behavss2']
#baseindex = msdata_load['baseindex']
#basess = msdata_load['basess']
movement = msdata_load['movement']
msGroup = msdata_load['msGroup']
msdir = msdata_load['msdir']
signalss = msdata_load['signalss']
    
highGroup = msGroup['highGroup']
midleGroup = msGroup['midleGroup']
lowGroup = msGroup['lowGroup']
salineGroup = msGroup['salineGroup']
restrictionGroup = msGroup['restrictionGroup']
ketoGroup = msGroup['ketoGroup']
lidocaineGroup = msGroup['lidocaineGroup']
capsaicinGroup = msGroup['capsaicinGroup']
yohimbineGroup = msGroup['yohimbineGroup']
pslGroup = msGroup['pslGroup']
shamGroup = msGroup['shamGroup']
adenosineGroup = msGroup['adenosineGroup']
highGroup2 = msGroup['highGroup2']

msset = msGroup['msset']
del msGroup['msset']
skiplist = restrictionGroup + lowGroup + lidocaineGroup

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup
pslset = pslGroup + shamGroup + adenosineGroup

painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup']
nonpainGroup = msGroup['salineGroup'] 

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

# 최종 평가 함수 
def accuracy_cal(pain, non_pain, fsw=False):
    pos_label = 1; roc_auc = -np.inf; fig = None
    
    while roc_auc < 0.5:
        pain = np.array(pain); non_pain = np.array(non_pain)
        pain = pain[np.isnan(pain)==0]; non_pain = non_pain[np.isnan(non_pain)==0]
        
        anstable = list(np.ones(pain.shape[0])) + list(np.zeros(non_pain.shape[0]))
        predictValue = np.array(list(pain)+list(non_pain)); predictAns = np.array(anstable)
        #            
        fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
        
        maxix = np.argmax((1-fpr) * tpr)
        specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
        accuracy = ((pain.shape[0] * sensitivity) + (non_pain.shape[0]  * specificity)) / (pain.shape[0] + non_pain.shape[0])
        
        roc_auc = metrics.auc(fpr,tpr)
        
        if roc_auc < 0.5:
            pos_label = 0
    
    if fsw:
        print('total samples', pain.shape[0]+non_pain.shape[0])
        print('specificity', round(specificity,3), 'sensitivity', round(sensitivity,3), 'accuracy', round(accuracy,3), 'roc_auc', round(roc_auc,3)) 
    
    if fsw:
        sz = 1
        fig = plt.figure(1, figsize=(7*sz, 5*sz))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
#        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
        
    return accuracy, roc_auc, fig

#def mstest2_psl():
#    # test 2: timewindow
#    mssave = []; forlist = list(range(1, 300)) # 앞뒤가 nan이 찍히는 모든 범위로 설정 할 것 
#    print('test2: timewindow 최적화를 시작합니다.')
#    for mssec in forlist:
##        print(mssec)
#        biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
#        skipsw = False
#        msduration = int(round((((mssec*FPS)-82)/10)+1))
##        print(msduration)
#        for SE in range(N):
#            for se in range(3):
#                c1 = SE in pslset and [SE, se] in longlist
#                
#                if c1:
##                    print(SE,se)
#                    min_mean_mean = np.array(min_mean_save[SE][se])
#                    
#                    if min_mean_mean.shape[0] - msduration <= 0 or msduration < 1:
#                        skipsw = True
#                        break
#                    
#                    meansave = []
#                    for msbin in range(min_mean_mean.shape[0] - msduration):
#                        meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
#                        
#                    maxix = np.argmax(meansave)
#                    biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
#        
#        if not(skipsw):
#            msacc = msloss2_psl(biRNN_2)
#            mssave.append(msacc)
#        elif skipsw:
#            mssave.append(np.nan)
##    plt.plot(mssave)
#                          
#    mssec = forlist[np.nanargmax(mssave)]
##    mssec = 60 # mannual (sec)
#    msduration = int(round((((mssec*FPS)-82)/10)+1))
#    print('optimized time window, mssec', mssec)
#    biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
#    for SE in range(N):
#        for se in range(3):
#            c1 = SE in pslset and [SE, se] in longlist
#            if c1:
#                min_mean_mean = np.array(min_mean_save[SE][se])
#                
#                if min_mean_mean.shape[0] - msduration <= 0 or msduration < 1:
#                    skipsw = True
#                    break
#                
#                meansave = []
#                for msbin in range(min_mean_mean.shape[0] - msduration):
#                    meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
#                    
#                maxix = np.argmax(meansave)
#                biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
#                msacc = msloss2_psl(biRNN_2)
#                
#    return biRNN_2, msacc

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))

# load 할 model 경로(들) 입력
# index, project
project_list = []
project_list.append(['1128_binfix5_1', 100, None])

model_name = project_list 

bins = 10
shortlist = []; longlist = []
for SE in range(N):
    if SE in mouseGroup:
        if not SE in skiplist:
            sessionNum = 5
            if SE in se3set:
                sessionNum = 3
            
            for se in range(sessionNum):
                length = np.array(signalss[SE][se]).shape[0]
                if length > 180*FPS:
                    longlist.append([SE,se])
                elif length < 180*FPS:
                    shortlist.append([SE,se])
                else:
                    print('error')


# In
################

msunit = 4; fn = 1; #seed = 100
div = msunit
rationList = list(np.arange(1/div, 1+1/div, 1/div))
                    
X = []; [X.append([]) for u in range(len(rationList))]
Y = []; Z = []
for q, roiRatio in enumerate(rationList): # 0.5    # In min_mean_save에 모든 data 저장
    min_mean_save = []
    [min_mean_save.append([]) for k in range(N)]
     
    ## pointSvae - 2차 학습 label 판단에 사용하기 위해 예측 평균값 저장
    #pointSave = []
    #[pointSave.append([]) for k in range(N)]
    for SE in range(N):
        if not SE in grouped_total_list or SE in skiplist: # ETC 추가후 lidocine skip 삭제할것 (여러개)
    #        print(SE, 'skip')
            continue
    
        sessionNum = 5
        if SE in se3set:
            sessionNum = 3
        
        [min_mean_save[SE].append([]) for k in range(sessionNum)]
    #    [pointSave[SE].append([]) for k in range(sessionNum)]
        msreport = True
        for se in range(sessionNum):
            current_value = []
            for i in range(len(model_name)): # repeat model 만큼 반복 후 평균냄
                ssw = False
                
                loadpath5 = savepath + 'result\\' + model_name[i][0] + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
              
                if os.path.isfile(loadpath5):
                    ssw = True
                else: # 구형과 호환을 위해 2주소 설정
                    loadpath5 = savepath + 'result\\' + model_name[i][0] + '\\exp_raw\\' + \
                    model_name[i][0] + '_PSL_result_' + str(SE) + '.pickle'
                    if os.path.isfile(loadpath5):
                        ssw = True
                    else:
                        if msreport:
                            msreport = False
#                            print(SE, 'skip')
                
                if ssw:
                    with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                        PSL_result_save = pickle.load(f)
                
                # ##################################
                    PSL_result_save2 = PSL_result_save[SE][se] # [BINS][ROI][bins] # BINS , full length 넘어갈때, # bins는 full length 안에서
                    current_BINS = []
                    BINnum = len(PSL_result_save2)
                    if BINnum != 0:
                        for BINS in range(len(PSL_result_save2)):
                            current_ROI = []
                            for ROI in range(len(PSL_result_save2[BINS])):
                                
                                current_ROI.append(np.argmax(PSL_result_save2[BINS][ROI], axis=1) == 1)
    #                            current_ROI.append(PSL_result_save2[BINS][ROI][:,1])
                                
                            roiRank = np.mean(np.array(current_ROI), axis=1) #[ROI, bins]
                            current_ROI_rank = np.array(current_ROI)[np.argsort(roiRank)[::-1][:int(round(roiRank.shape[0]*roiRatio))], :]
                            current_BINS.append(np.mean(np.array(current_ROI_rank ), axis=0)) # ROI 평균
                        current_value.append(current_BINS)
            
            if len(current_value) > 0:
                current_value = np.mean(np.array(current_value), axis=0) # 모든 반복 project에 대해서 평균처리함
      
                binNum = current_value.shape[0] # [BINS]
                mslength = current_value.shape[1]
                
                # 시계열 형태로 표현용 
                empty_board = np.zeros((binNum, mslength + (binNum-1)))
                empty_board[:,:] = np.nan
        #       
                label = []       
                set1 = highGroup + midleGroup + lowGroup + yohimbineGroup + ketoGroup + lidocaineGroup + restrictionGroup     
                c1 = SE in set1 and se in [0,2]
                c2 = SE in capsaicinGroup and se in [0,2]
                c3 = SE in pslGroup + adenosineGroup and se in [0]
                c4 = SE in shamGroup and se in [0,1,2]
                c5 = SE in salineGroup and se in [0,1,2,3,4]
                c6 = SE in highGroup2 and se in [2]
                                
                if c1 or c2 or c3 or c4 or c5 or c6:
                    exceptbaseline = (SE in np.array(msset)[:,1:].flatten()) and se == 0
                    if not exceptbaseline: # baseline을 공유하므로, 사용하지 않는다. 
                            label = [1,0] # class 0 : nonpain
                            
                set2 = highGroup + midleGroup + yohimbineGroup + ketoGroup + capsaicinGroup + highGroup2
                c7 = SE in set2 and se in [1]
                if c7: # 
                    label = [0,1] # class 1 : pain
            
                if len(label) == 0:
                    continue

                for BIN in range(binNum):
                    plotsave = current_value[BIN,:]
                    
                    X[q].append(plotsave)
                    if q == 0:
                        Y.append(label)
                        Z.append([SE, se])
                  
X = np.array(X); Y = np.array(Y); Z = np.array(Z)
print('div', X.shape[0], ',, dataset size', len(X[0]), Y.shape[0], Z.shape[0])

# In[]####################        

from datetime import datetime
from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam         

n_in =  1 # number of features
n_out = 2 # number of class # 20191104: 3 class로 시도

# learning intensity
epochs = 5 # epoch 종료를 결정할 최소 단위.
lr = 1e-3 # learning rate
fn = 1

n_hidden = int(8 * 3) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 3) # fully conneted laye node 갯수 # 8

l2_rate = 0.25 # regularization 상수
dropout_rate1 = 0.20 # dropout late; before
dropout_rate2 = 0.10 # dropout late; after

batch_size = 500

def keras_setup(seed):
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  
    dt = datetime.now()
    idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)
    
    inputsize = np.zeros(msunit *fn, dtype=int) 
    for unit in range(msunit *fn):
        inputsize[unit] = X[unit].shape[1]
        
    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
    
    input1 = []; [input1.append([]) for i in range(msunit *fn)] # 최초 input layer
    input2 = []; [input2.append([]) for i in range(msunit *fn)] # input1을 받아서 끝까지 이어지는 변수
    
    for unit in range(msunit *fn):
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

##
runlist = []
runlist.append(['secondML_1', 100])    

for i1, i2 in enumerate(runlist):
    seed = i2[1]; settingID = i2[0]
    model, idcode = keras_setup(seed)  
    
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

      
    initial_weightsave = RESULT_SAVE_PATH + 'model//' + 'initial_weight.h5'
    model.save_weights(initial_weightsave)     
    
    ###
    # oversampling

    nonpain_sampleNum = np.sum(Y[:,0] == 1)
    pain_sampleNum = np.sum(Y[:,1] == 1)
    
    print('nonpain sample #... ', nonpain_sampleNum , 'pain sample #... ', pain_sampleNum )
    
    if pain_sampleNum * nonpain_sampleNum == 0:
        import sys
        sys.exit()
    
    duplicateNum = nonpain_sampleNum // pain_sampleNum
    remain = nonpain_sampleNum - pain_sampleNum * duplicateNum
    
    pix = np.where(Y[:,1] == 1)[0]
    
    X2 = np.array(X); Y2 = np.array(Y); Z2 = np.array(Z)
    
    for u in range(duplicateNum):
        X2 = np.concatenate((X2, X[:,pix,:]), axis=1)
        Y2 = np.concatenate((Y2, Y[pix,:]), axis=0)
        Z2 = np.concatenate((Z2, Z[pix,:]), axis=0)

    random.seed(seed)
    rix = random.sample(range(Y.shape[0]), remain)
    X2 = np.concatenate((X2, X[:,rix,:]), axis=1)
    Y2 = np.concatenate((Y2, Y[rix,:]), axis=0)
    Z2 = np.concatenate((Z2, Z[rix,:]), axis=0)
          
    del X; del Y; del Z
    X = X2; Y = Y2; Z = Z2
    
    nonpain_sampleNum = np.sum(Y[:,0] == 1)
    pain_sampleNum = np.sum(Y[:,1] == 1)
    print('nonpain sample #... ', nonpain_sampleNum , 'pain sample #... ', pain_sampleNum )
    
    # training
    trainingset = list(grouped_total_list); etc = []
    forlist = list(trainingset)
    for SE in forlist:
        c1 = np.sum(Z[:,0]==SE) == 0 # 옥으로 전혀 선택되지 않았다면 test set으로 빼지 않음
        if c1 and SE in trainingset:
            trainingset.remove(SE)
            print('removed', SE)
            
            if not SE in np.array(msset).flatten():
                etc.append(SE)
#                print('etc append', SE)
            
        c2 = np.array(msset)[:,0]
        if SE in c2:
            for u in np.array(msset)[np.where(np.array(msset)[:,0] == SE)[0][0],:][1:]:
                trainingset.remove(u)
                print('removed, msset', u)

    mouselist = trainingset
    mouselist.sort()
    if not(len(etc) == 0):
        mouselist.append(etc[0])
    
    wanted = pslset
    print('mouselist', mouselist)
    print('etc', etc)
    forlist = list(wanted); wanted = list(wanted)
    for i in forlist:
        if not i in mouselist:
            wanted.remove(i)
            print('remove in wanted list', i)
            
    for msindex, mousenum in enumerate(wanted):
        final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mousenum) + '_my_model_weights_final.h5'
        
        if not(os.path.exists(final_weightsave)):
            print('mouse #', [mousenum], '학습된 model 없음. 새로시작합니다.')
            model.load_weights(initial_weightsave)
            
            X_training = []; [X_training.append([]) for i in range(msunit *fn)] # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
            X_valid = []; [X_valid.append([]) for i in range(msunit *fn)]
            Y_training_list = []
#            Y_training_control_list = []
            
            delist = np.where(Z[:,0]==mousenum)[0]
            
            if mousenum in np.array(msset)[:,0]:
                for u in np.array(msset)[np.where(np.array(msset)[:,0] == mousenum)[0][0],:][1:]:
                    delist = np.concatenate((delist, np.where(mousenum==u)[0]), axis=0)
            
            print('mouse #', [mousenum],'delist #', len(delist))
            
            for unit in range(msunit *fn): # input은 msunit 만큼 병렬구조임. for loop으로 각자 계산함
                X_training[unit] = np.delete(np.array(X[unit]), delist, 0)
        
            Y_training_list = np.delete(np.array(Y), delist, 0)
        
    
            # bias 방지를 위해 동일하게 shuffle 
            np.random.seed(seed)
            shuffleix = list(range(X_training[0].shape[0]))
            np.random.shuffle(shuffleix) 
#                    print(shuffleix)
   
            tr_y_shuffle = Y_training_list[shuffleix]
            tr_x = []
            for unit in range(msunit *fn):
                tr_x.append(X_training[unit][shuffleix])
                
            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs)
            # missing data 처리, 크기 불일치 문제
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
