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
import scipy

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

shortlist = []; longlist = []
def msGrouping_nonexclude(msdata): 
    df3 = pd.concat([pd.DataFrame(msdata[highGroup + highGroup2]) , pd.DataFrame(msdata[midleGroup]), \
                     pd.DataFrame(msdata[salineGroup])] \
                     , ignore_index=True, axis=1)
        
    df3 = np.array(df3)
    
    return df3

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
#movement = msdata_load['movement']
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
pslGroup = msGroup['pslGroup']
shamGroup = msGroup['shamGroup']
adenosineGroup = msGroup['adenosineGroup']
highGroup2 = msGroup['highGroup2']
CFAgroup = msGroup['CFAgroup']

msset = msGroup['msset']
del msGroup['msset']
skiplist = restrictionGroup + lowGroup + lidocainGroup

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup
pslset = pslGroup + shamGroup + adenosineGroup

#painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup']
#nonpainGroup = msGroup['salineGroup'] 

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
    
def pain_nonpain_sepreate(target, painGroup, nonpainGroup):
    valuetable = np.array(target)
    painset = []; nonpainset =[]

    for i in grouped_total_list:
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

def ms_batch_ind(target):
    target = np.array(target)
    out = np.zeros((N,5)); out[:] = np.nan
    
    # 일단 다 넣고
    for SE in mouseGroup:
        out[SE,:] = target[SE,:]
    
    # 첫번째가 아니면 baseline을 nan으로 교체 
    for SE in mouseGroup:
        if SE in np.array(msset)[:,1:].flatten():
            out[SE,0] = np.nan
            
    return out

def msloss2_psl(biRNN_2):
#    biRNN_22 = ms_batchmean(biRNN_2) # 중복 데이터 처리.. 평균을 내든 뺴든 ...
    biRNN_22 = ms_batch_ind(biRNN_2) # 단일쥐에서 평균내지 않을 때

    a1 = biRNN_22[shamGroup,0].flatten()
    a1 = a1[np.isnan(a1) == False]
    a2 = biRNN_22[shamGroup,1].flatten()
    a2 = a2[np.isnan(a2) == False]
    a3 = biRNN_22[shamGroup,2].flatten()
    a3 = a3[np.isnan(a3) == False]
    
    b2 = biRNN_22[pslGroup,1].flatten()
    b2 = b2[np.isnan(b2) == False]
    b3 = biRNN_22[pslGroup,2].flatten()
    b3 = b3[np.isnan(b3) == False]
#    pain = np.concatenate((b1,b2))
    
#    accuracy, roc_auc, fig = accuracy_cal(pain, nonpain, fsw=False)
    pvalue = stats.ttest_ind(a2, b2)[1] + stats.ttest_ind(a3, b3)[1]
    
    return -pvalue

def mstest2_psl():
    # test 2: timewindow
    mssave = []; forlist = list(range(1, 300)) # 앞뒤가 nan이 찍히는 모든 범위로 설정 할 것 
    print('test2: timewindow 최적화를 시작합니다.')
    for mssec in forlist:
#        print(mssec)
        biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
        skipsw = False
        msduration = int(round((((mssec*FPS)-82)/10)+1))
#        print(msduration)
        for SE in range(N):
            for se in range(3):
                c1 = SE in pslset and [SE, se] in longlist
                
                if c1:
#                    print(SE,se)
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    
                    if min_mean_mean.shape[0] - msduration <= 0 or msduration < 1:
                        skipsw = True
                        break
                    
                    meansave = []
                    for msbin in range(min_mean_mean.shape[0] - msduration):
                        meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
                        
                    maxix = np.argmax(meansave)
                    biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
        
        if not(skipsw):
            msacc = msloss2_psl(biRNN_2)
            mssave.append(msacc)
        elif skipsw:
            mssave.append(np.nan)
#    plt.plot(mssave)
                          
    mssec = forlist[np.nanargmax(mssave)]
    mssec = 60 # mannual (sec)
    msduration = int(round((((mssec*FPS)-82)/10)+1))
    print('optimized time window, mssec', mssec)
    biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
    for SE in range(N):
        for se in range(3):
            c1 = SE in pslset and [SE, se] in longlist
            if c1:
                min_mean_mean = np.array(min_mean_save[SE][se])
                
                if min_mean_mean.shape[0] - msduration <= 0 or msduration < 1:
                    skipsw = True
                    break
                
                meansave = []
                for msbin in range(min_mean_mean.shape[0] - msduration):
                    meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
                    
                maxix = np.argmax(meansave)
                biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
                msacc = msloss2_psl(biRNN_2)
                
    return biRNN_2, msacc

def msacc2(pain, nonpain):
    pos_label = 1; roc_auc = -np.inf
    
    pain = np.array(pain); nonpain = np.array(nonpain)
    pain = pain[np.isnan(pain)==0]; nonpain = nonpain[np.isnan(nonpain)==0]
    
    anstable = list(np.ones(pain.shape[0])) + list(np.zeros(nonpain.shape[0]))
    predictValue = np.array(list(pain)+list(nonpain)); predictAns = np.array(anstable)
    #            
    fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
    maxix = np.argmax((1-fpr) * tpr)
    specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
    accuracy = ((pain.shape[0] * sensitivity) + (nonpain.shape[0]  * specificity)) / (pain.shape[0] + nonpain.shape[0])
    roc_auc = metrics.auc(fpr,tpr)
    
    return accuracy

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))

# load 할 model 경로(들) 입력
# index, project
project_list = []

#project_list.append(['0114_double_merge', 100, None])

project_list.append(['0116_CFA_l2_1', 200, None])
project_list.append(['0116_CFA_l2_2', 300, None])
#project_list.append(['0116_CFA_l2_3', 400, None])

model_name = project_list 
             
t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
        # 개별 thr로 relu 적용되어있음. frame은 signal과 syn가 다름

# In[]
min_mean_save = []; [min_mean_save.append([]) for k in range(N)]
roiRatio = 1
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist: # ETC 추가후 lidocine skip 삭제할것 (여러개)
#        print(SE, 'skip')
        continue

    sessionNum = 5
    if SE in se3set:
        sessionNum = 3
    
    [min_mean_save[SE].append([]) for k in range(sessionNum)]
    for se in range(sessionNum):
        current_value = []
        for i in range(len(model_name)): # repeat model 만큼 반복 후 평균냄
            ssw = False
            
            loadpath5 = savepath + 'result\\' + model_name[i][0] + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
                
            if os.path.isfile(loadpath5):
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
                            sw = 'binarization'
                            if sw == 'binarization':
                                current_ROI.append(np.argmax(PSL_result_save2[BINS][ROI], axis=1) == 1)
                            elif sw == 'probability':
                                current_ROI.append(PSL_result_save2[BINS][ROI][:,1])
                            
                        roiRank = np.mean(np.array(current_ROI), axis=1) #[ROI, bins]
                        
                        # 상위 x % ROI 만 filtering 
                        current_ROI_rank = np.array(current_ROI)[np.argsort(roiRank)[::-1][:int(round(roiRank.shape[0]*roiRatio))], :]
                        current_BINS.append(np.mean(np.array(current_ROI_rank ), axis=0)) # ROI 평균
                    current_value.append(current_BINS)
                    
        if len(current_value) > 0:
            current_value = np.mean(np.array(current_value), axis=0) # mean by project
            mean_bins = np.mean(np.array(current_value), axis=1) # mean by bins
            mean_BINS = np.mean(mean_bins) # mean by BINS
            min_mean_save[SE][se] = mean_BINS

        elif len(current_value) == 0:
            min_mean_save[SE][se] = np.nan

biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan;
## 모든 set 평균값으로만 계산
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist:
#            print(SE, 'skip')
        continue
    sessionNum = 5
    if SE in se3set:
        sessionNum = 3
        
    for se in range(sessionNum):
        biRNN_2[SE,se]  = min_mean_save[SE][se]

## subset 평균처리 후 main set으로 통합, subset은 nan으로 제외
biRNN_22 = np.zeros((N,5)); biRNN_22[:] = np.nan
for SE in range(N):
    if SE in np.array(msset)[:,0]:
        biRNN_22[SE,:] = np.nanmean(biRNN_2[np.array(msset)[np.where(np.array(msset)[:,0]==SE)[0][0],:],:],axis=0)
    elif SE not in np.array(msset).flatten(): 
        biRNN_22[SE,:] = biRNN_2[SE,:]

## PRISM 정리 및 통계처리
Aprism_biRNN2_formalin = msGrouping_nonexclude(biRNN_22)
Aprism_biRNN2_capsaicin = biRNN_22[capsaicinGroup,0:3]
Aprism_biRNN2_CFA = biRNN_22[CFAgroup,0:3]
Aprism_biRNN2_psl = msGrouping_pslOnly(biRNN_22)





# In[]
import os
os.sys.exit()

















