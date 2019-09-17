# msbak, 2019. 09. 02.
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""

import pandas as pd
#import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def msMinMaxScaler(matrix1):
    matrix1 = np.array(matrix1)
    msmin = np.min(matrix1)
    msmax = np.max(matrix1)
    
    return (matrix1 - msmin) / (msmax-msmin)

def mslinear_regression(x,y):
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m

    return m, b # bx+a

import pickle
import os
try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\msbak\\Documents\\tensor\\'; os.chdir(savepath);
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
lidocainGroup = msGroup['lidocaineGroup']
capsaicinGroup = msGroup['capsaicinGroup']
yohimbineGroup = msGroup['yohimbineGroup']

# mouselist는 training에 사용됩니다.
mouselist = []
mouselist += msGroup['highGroup']
mouselist += msGroup['ketoGroup']
mouselist += msGroup['midleGroup']
mouselist += msGroup['salineGroup']
mouselist += msGroup['yohimbineGroup']
mouselist += [msGroup['lidocaineGroup'][0]]
etc = msGroup['lidocaineGroup'][0]

mouselist.sort()

# 최종 평가 함수 
def accuracy_cal(pain, non_pain, fsw):
    pos_label = 1; roc_auc = -np.inf
    
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
        plt.figure()
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
        
    return accuracy, roc_auc 
    
def pain_nonpain_sepreate(target, painGroup, nonpainGroup):
    valuetable = np.array(target)
    painset = []; nonpainset =[]

    for i in mouseGroup:
        if i in painGroup:
            painset.append(i)
        elif i in nonpainGroup:
            nonpainset.append(i)

    pain = (valuetable[painset,:][:,1]).flatten()
    
    sal = valuetable[nonpainset,:].flatten()
    base = valuetable[painset,:][:,0].flatten()
    inter = valuetable[painset,:][:,2].flatten()
#    non_pain = np.concatenate((sal, base, inter))
    
    # within, between
    
    nonpain_within = np.concatenate((base, inter), axis=0)
    nonpain_between = sal
    
    return pain, nonpain_within, nonpain_between

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))

# load 할 model 경로(들) 입력
model_name = []
#model_name.append(['0819_test_1/', 1])
#model_name.append(['0829_downsize_7_1/', 1])
model_name.append(['0903_seeding_1/', 1])
model_name.append(['0903_seeding_2/', 2])
model_name.append(['0903_seeding_3/', 3]) 
model_name.append(['0903_seeding_4/', 4]) 
model_name.append(['0903_seeding_5/', 5]) 

accsave = np.zeros(N); accsave[:] = np.nan
repeat = 0
for repeat in range(len(model_name)):
    msloadpath = savepath + 'result\\' + model_name[repeat][0] + '\\model';
    os.chdir(msloadpath)
    for SE in mouselist:
        try:
            df1 = np.array(pd.read_csv(str(SE) + '_exp_validationSet_result.csv', header=None))
            accsave[SE] = df1[0,-1]
            # print(SE)
        except:
            # print(SE, 'none')
            pass

    print(model_name[repeat] , '>>', np.nanmean(accsave))            

# test data import
rawsave = []; [rawsave.append([]) for i in range(N)]
repeat = 0
for repeat in range(len(model_name)):
    msloadpath = savepath + 'result\\' + model_name[repeat][0] + '\\exp_raw'; 

    try:
        os.chdir(msloadpath)
        print(model_name[repeat] , 'is loaded'); printsw=False
    except:
        continue
        
    for SE in range(N):
        try:
            df1 = pd.read_csv('biRNN_raw_' + str(SE) + '.csv', header=None)
        except:
            if SE in mouseGroup and SE not in (restrictionGroup + lowGroup):
                print(repeat, 'repeat에', SE, 'data 없음')
            continue
            
        df2 = np.array(df1)
        df2[:,4] = df2[:,4] > 0.5 # binarization
        rawsave[SE].append(df2)
        
for SE in range(N):
    rawsave[SE] = np.nanmean(np.array(rawsave[SE]), axis=0) # nanmean으로 공백 매꿈     
         

# In[]
def ms_thresholding(optimalThr = 0):
    target = np.zeros((N,5)); target[:] = np.nan
    for SE in range(N):
        if rawsave[SE].shape: # 예외 data는 실행하지 않음
            ROInum = signalss[SE][0].shape[1]
            for se in range(5):
                valuesave = np.zeros(ROInum); valuesave[:] = np.nan 
                for ROI in range(ROInum):
                    rowindex = np.where((np.array(rawsave[SE][:,1]==se) * np.array( rawsave[SE][:,2]==ROI)) == True)[0]
                    valuesave[ROI] = np.mean(rawsave[SE][rowindex,4]) # 4 for pain probability 
                    # >0.5 를 사용하면 loss -> acc로 변환
                    
    #            valuesave = valuesave > optimalThr # binary
                valuesave[valuesave < optimalThr] = 0 # thr 이하의 값은 noise로 취급, 0으로 처리
                    
                target[SE,se] = np.mean(valuesave) # 데이터들 수정후, mean으로 교체
    return target
            
# In[]

optimized_thr = 0
print('optimized_thr', optimized_thr)


target = ms_thresholding(optimized_thr)

# 최종 평가
print('______________ biRNN ______________')

painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

print('highGroup', np.nanmean(target[highGroup,1]))
print('midleGroup', np.nanmean(target[midleGroup,1]))
print('lowGroup', np.nanmean(target[lowGroup,1]))
print('restrictionGroup', np.nanmean(target[restrictionGroup,1]))
print('ketoGroup', np.nanmean(target[ketoGroup,1]))
print('salineGroup', np.nanmean(target[salineGroup,1]))
print('lidocainGroup', np.nanmean(target[lidocainGroup,1]))
print('capsaicinGroup ', np.nanmean(target[capsaicinGroup ,1]))
print('yohimbineGroup_early ', np.nanmean(target[yohimbineGroup ,1]))
print('yohimbineGroup_late ', np.nanmean(target[yohimbineGroup ,3]))

# control, 그룹내, 그룹외 추가 요망

# 개별평가
# formalin - within
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('formalin - within')
accuracy_cal(pain, nonpain_within, True)

# formalin - between
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('formalin - between')
accuracy_cal(pain, nonpain_between, True)

# capsaicin - within
painGroup = capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('capsaicin - within')
accuracy_cal(pain, nonpain_within, True)

# capsaicin - between
painGroup = capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('capsaicin - between')
accuracy_cal(pain, nonpain_between, True)

# In[] test용 임시구문

# 개별평가
if False:
    tartget_tmp = t10
    
    # formalin - within
    painGroup = highGroup + midleGroup + ketoGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('formalin - within')
    accuracy_cal(pain, nonpain_within, True)
    
    # formalin - between
    painGroup = highGroup + midleGroup + ketoGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('formalin - between')
    accuracy_cal(pain, nonpain_between, True)
    
    # capsaicin - within
    painGroup = capsaicinGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('capsaicin - within')
    accuracy_cal(pain, nonpain_within, True)
    
    # capsaicin - between
    painGroup = capsaicinGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('capsaicin - between')
    accuracy_cal(pain, nonpain_between, True)

# In[]
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
print('movment--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
print('total--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(msdict['total'], painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

# In[] 취약점 분석
# predict_matrix_total
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

targetMatrix = np.array(target)
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(targetMatrix, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)

pain1 = np.array(pain); non_pain1 = np.array(nonpain)
pain1 = pain1[np.isnan(pain1)==0]; non_pain1 = non_pain1[np.isnan(non_pain1)==0]
maxvalue = np.max([np.concatenate((pain1, non_pain1))])
print('maxvalue', maxvalue)

xaxis = []; yaxis = []
for thr in np.arange(0,maxvalue,maxvalue/1000):
    TP = np.sum(pain1 >= thr)
    FN = np.sum(pain1 < thr)
    FP = np.sum(non_pain1 >= thr)
    TN = np.sum(non_pain1 < thr)
    msaccuracy = (TP + TN) / (TP + TN + FN + FP)
    
    xaxis.append(thr)
    yaxis.append(msaccuracy)
    
#    plt.plot(xaxis, yaxis)
    
optimized_thr = xaxis[np.argmax(yaxis)]
print('optimized_thr', optimized_thr)

Z = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        Z[SE,se] = SE*100 + se

painGroup = highGroup + midleGroup + ketoGroup; painGroup = np.array(painGroup)
nonpainGroup = salineGroup + lidocainGroup

FNlist = Z[painGroup,1][np.where(targetMatrix[painGroup,1] < optimized_thr)[0]]

FPlist_ix = np.where(np.concatenate((targetMatrix[painGroup,0], targetMatrix[painGroup,2], targetMatrix[nonpainGroup,:].flatten())) > optimized_thr)[0]
FPlist = np.concatenate((Z[painGroup,0], Z[painGroup,2], Z[nonpainGroup,:].flatten()))[FPlist_ix]

for i in FNlist:
    SE = int(i // 100)
    se = int(i % 100)
    print(SE, se, 'False Negative', targetMatrix[SE,se])

print('____________________________________________')

for i in FPlist:
    SE = int(i // 100)
    se = int(i % 100)
    print(SE, se, 'False Positive', targetMatrix[SE,se])
    
# In[] 움직임 noise 문제 해결하기 - new

# 랜덤한 갯수로 nonpain을 뽑아서 평균 movement와, 그 그룹만 썻을때 accuracy를 2d plot
# 서로 상관관계가 있는지.
    
import random

painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

pain_movement, nonpain_within_movement, nonpain_between_movement = pain_nonpain_sepreate(movement, painGroup, nonpainGroup)
nonpain_movement = np.concatenate((nonpain_within_movement, nonpain_between_movement), axis=0)

axiss = []; [axiss.append([]) for i in range(2)]
totalNum = nonpain_movement.shape[0]
epochs = 50000
for epoch in range(epochs):
    if epoch % int(epochs/10) == 1:
        print(epoch, '/', epochs)
    random_N = random.randrange(totalNum)
    ixlist = random.sample(list(range(totalNum)), random_N)
    
    accuracy, _ = accuracy_cal(pain_movement, nonpain_movement[ixlist], False)
    axiss[0].append(np.mean(nonpain_movement[ixlist], axis=0))
    axiss[1].append(accuracy)

for d in range(2):
    axiss[d] = np.array(axiss[d])
    axiss[d] = axiss[d][np.isnan(axiss[d])==0];
    
#maxvalue = np.max([np.concatenate((pain1, non_pain1))])
    
m, b = mslinear_regression(axiss[0], axiss[1])
print(m, b)
plt.figure()
plt.scatter(axiss[0], axiss[1], s = 0.4)
plt.xlabel('movement ratio mean')
plt.ylabel('accuracy by movement as feature')

xaxis = np.arange(0,0.5,0.5/100)
plt.plot(xaxis, xaxis*m + b, c = 'orange')
plt.xlim([0,0.5]), plt.ylim([0,1])

# 대조군

pain_target, nonpain_within_target, nonpain_between_target = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain_target = np.concatenate((nonpain_within_target, nonpain_between_target), axis=0)


axiss = []; [axiss.append([]) for i in range(2)]
totalNum = nonpain_movement.shape[0]

#epochs = 50000
for epoch in range(epochs):
    if epoch % int(epochs/10) == 1:
        print(epoch, '/', epochs)
        
    random_N = random.randrange(totalNum)
    ixlist = random.sample(list(range(totalNum)), random_N)
    
    accuracy, _ = accuracy_cal(pain_target, nonpain_target[ixlist], False)
    axiss[0].append(np.mean(nonpain_movement[ixlist], axis=0))
    axiss[1].append(accuracy)

for d in range(2):
    axiss[d] = np.array(axiss[d])
    axiss[d] = axiss[d][np.isnan(axiss[d])==0];
    
#maxvalue = np.max([np.concatenate((pain1, non_pain1))])
    
m, b = mslinear_regression(axiss[0], axiss[1])
print(m, b)
plt.figure()
plt.scatter(axiss[0], axiss[1], s = 0.4)
plt.xlabel('movement ratio mean')
plt.ylabel('accuracy by RNN')

xaxis = np.arange(0,0.5,0.5/100)
plt.plot(xaxis, xaxis*m + b, c = 'orange')
plt.xlim([0,0.5]), plt.ylim([0,1])

# In[] to Prism

def msGrouping_nonexclude(msmatrix): # base 예외처리 없음, goruping된 sample만 뽑힘
    target = np.array(msmatrix)
    
    df3 = pd.DataFrame(target[highGroup]) 
    df3 = pd.concat([df3, pd.DataFrame(target[midleGroup]), \
                     pd.DataFrame(target[salineGroup]), \
                     pd.DataFrame(target[ketoGroup]), pd.DataFrame(target[lidocainGroup]), \
                     pd.DataFrame(target[yohimbineGroup]), pd.DataFrame(target[capsaicinGroup][:,0:3])], \
                        ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

Aprism = msGrouping_nonexclude(target)

# In[] fig1. movement
target1 = np.array(movement)
Aprism_movement_supple = msGrouping_nonexclude(target1)
Aprism_movement_main = np.concatenate((Aprism_movement_supple[:,0:5], \
                                       Aprism_movement_supple[:,5:10], \
                                       Aprism_movement_supple[:,10:15], \
                                       Aprism_movement_supple[:,25:30]), axis = 1)


# In[] 움직임 정보를 이용하여  regression 시도

target = ms_thresholding()

target[highGroup,1]


# In[] Pain cell?
    
painROIsave = []; [painROIsave.append([]) for i in range(N)]

SE = 0; se = 1; ROI = 0
for SE in range(N):
    ROInum = signalss[SE][0].shape[1]
    [painROIsave[SE].append([]) for i in range(5)]
    for se in range(5):
        valuesave = []; pain_ROI = np.zeros(ROInum); pain_time = []
        [painROIsave[SE][se].append([]) for i in range(ROInum)]
        for ROI in range(ROInum):       
            if not rawsave[SE].shape:
                predict_series = [0]
                
            elif rawsave[SE].shape:
                rowindex = np.where((np.array(rawsave[SE][:,1]==se) * np.array( rawsave[SE][:,2]==ROI)) == True)[0]
                predict_series = np.array(rawsave[SE][rowindex,4])
            
            painROIsave[SE][se][ROI] = predict_series
            
plt.imshow(painROIsave[SE][se])

# 여기까지 pain 확률 matrix 생성

# In[] session 1 pain cell의 total activity로 pain state 추정
# painROIsave [SE][se][ROI][bins] = pain % 

painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

##
axiss = []; [axiss.append([]) for i in range(2)]
for thr in np.arange(0,1+0.1,0.1):
    print('thr', thr)

    t4_sum = np.zeros((N,5)); t4_min = np.zeros((N,5))
    SE = 0; se = 1; ROI = 0
    for SE in range(N):
        ROInum = signalss[SE][0].shape[1]
        for se in range(5):
            signal = np.array(signalss[SE][se])[:497,:]
            meansignal = np.mean(signal, axis=1)
            
            valsave = []
            for ROI in range(ROInum):
                paincell = (np.mean(painROIsave[SE][1][ROI]) > thr) # paincell
                valsave.append(np.mean(signal[:,ROI]) * paincell) 
            
            t4_sum[SE,se] = np.mean(valsave)
            
    target = t4_sum     
    
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
    nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
    ms1 = accuracy_cal(pain, nonpain, True)[1]
    
    axiss[0].append(thr); axiss[1].append(ms1)
plt.plot(axiss[0], axiss[1])
# 결론 -> 아무짓도 안하는게 좋다.. ?

####
# In[] session 1 pain cell 추정후 min 값으로 total activity 추정
# min: x ensemble의 activity를 esembel의 cell min으로 추정함
painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

##
axiss = []; [axiss.append([]) for i in range(2)]
for thr in np.arange(0,1+0.1,0.1):
    print('thr', thr)

    t4_sum = np.zeros((N,5)); t4_min = np.zeros((N,5))
    SE = 0; se = 1; ROI = 0
    for SE in range(N):
        ROInum = signalss[SE][0].shape[1]
        for se in range(5):
            signal = np.array(signalss[SE][se])[:497,:]
            meansignal = np.mean(signal, axis=1)
            
            valsave = []
            for ROI in range(ROInum):
                paincell = (np.mean(painROIsave[SE][1][ROI]) > thr) # paincell
                valsave.append(np.mean(signal[:,ROI]) * paincell)
                
            # 임의 최적화
            valsave = np.array(valsave)
            if np.sum(valsave>0) > 0:
                t4_sum[SE,se] = np.min(valsave[valsave>0])
            elif np.sum(valsave>0) == 0:
                t4_sum[SE,se] = 0
                
    target = t4_sum     
    
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
    nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
    ms1 = accuracy_cal(pain, nonpain, True)[1]
    
    axiss[0].append(thr); axiss[1].append(ms1)
plt.plot(axiss[0], axiss[1])

chanceLevel = np.max([pain.shape[0], nonpain.shape[0]]) / (pain.shape[0] + nonpain.shape[0])
print('chanceLevel', chanceLevel)


    
# In[]
# binning load
# 여기 parameter는 model에 의존적임. model 변경시 수정필수.
bins = 10; binningSave = []; thr = 0.5
for frame in range(0, 497-82+1, bins):
    binningSave.append([frame, frame+82])


painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
axiss = []; [axiss.append([]) for i in range(2)]
for thr in np.arange(0,1+0.1,0.1):
    print('thr', thr)

    SE = 0; se = 1; ROI = 0; bins = 0

    painDegree_save = []; [painDegree_save.append([]) for i in range(N)]
    for SE in range(N):
        [painDegree_save[SE].append([]) for i in range(5)]
        ROInum = signalss[SE][0].shape[1]
        for se in range(5):
            [painDegree_save[SE][se].append([]) for i in range(ROInum)]
            signal = np.array(signalss[SE][se])[:497,:]
            meansignal = np.mean(signal, axis=1)
            valsave = []
            for ROI in range(ROInum):
                painMatrix = np.array(painROIsave[SE][1][ROI])
                for bins in range(painMatrix.shape[0]):
                    paincell = painMatrix[bins] > thr
                    totalActivity = np.sum(signal[:,ROI][binningSave[bins][0]:binningSave[bins][1]])
                    
                    painDegree_save[SE][se][ROI].append(paincell * totalActivity)
                
    # save 후 정량 계산
                    
    t4_sum = np.zeros((N,5))                
    for SE in range(N):          
        for se in range(5):
            t4_sum[SE,se] = np.mean(np.array(painDegree_save[SE][se]))
            
    target = np.array(t4_sum)
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
    nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
    ms1 = accuracy_cal(pain, nonpain, True)[1]
    
    axiss[0].append(thr); axiss[1].append(ms1)
plt.plot(axiss[0], axiss[1])

# 응 실패 

###

# 원본 reload
optimized_thr = 0
print('optimized_thr', optimized_thr)
target = ms_thresholding(optimized_thr)
Aprism3 = msGrouping_nonexclude(target)
# optimizsed thr 계산
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

targetMatrix = np.array(target)
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(targetMatrix, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)

pain1 = np.array(pain); non_pain1 = np.array(nonpain)
pain1 = pain1[np.isnan(pain1)==0]; non_pain1 = non_pain1[np.isnan(non_pain1)==0]
maxvalue = np.max([np.concatenate((pain1, non_pain1))])
print('maxvalue', maxvalue)

xaxis = []; yaxis = []
for thr in np.arange(0,maxvalue,maxvalue/1000):
    TP = np.sum(pain1 >= thr)
    FN = np.sum(pain1 < thr)
    FP = np.sum(non_pain1 >= thr)
    TN = np.sum(non_pain1 < thr)
    msaccuracy = (TP + TN) / (TP + TN + FN + FP)
    
    xaxis.append(thr)
    yaxis.append(msaccuracy)
    
#    plt.plot(xaxis, yaxis)
    
optimized_thr = xaxis[np.argmax(yaxis)]
print('optimized_thr', optimized_thr)

msix = target>optimized_thr 
    
# t4 :497 계산
t4_497 = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        meansignal = np.mean(np.array(signalss[SE][se]), axis=1)
        
        t4_497[SE,se] = np.mean(meansignal[:497])
        
painDegree = np.array(t4_497 * msix)
painDegree[painDegree==0] = np.nan
        
target = np.array(t4_497 * msix)
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
ms1 = accuracy_cal(pain, nonpain, True)[1]

Aprism = msGrouping_nonexclude(painDegree)
Aprism2 = msGrouping_nonexclude(t4_497)

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














