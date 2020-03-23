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
    df3 = pd.concat([pd.DataFrame(msdata[salineGroup,0:4]) \
                                  ,pd.DataFrame(msdata[highGroup + highGroup2,0:4]) \
                                  ,pd.DataFrame(msdata[midleGroup,0:4]) \
                                  ,pd.DataFrame(msdata[ketoGroup,0:4]) \
                                  ,pd.DataFrame(msdata[lidocainGroup,0:4])] \
                                  ,ignore_index=True, axis=1)
    
    df3 = np.array(df3)
    return df3

def msGrouping_pslOnly(psl): # psl만 처리
    psldata = np.array(psl)
    
    df3 = pd.DataFrame(psldata[shamGroup,0:3]) 
    df3 = pd.concat([df3, pd.DataFrame(psldata[pslGroup,0:3]), \
                     pd.DataFrame(psldata[adenosineGroup,0:3])], ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

def msGrouping_yohimbine(msdata):
    msdata = np.array(msdata)
    df3 = np.array(pd.concat([pd.DataFrame(msdata[salineGroup,1:4]) \
                                  ,pd.DataFrame(msdata[yohimbineGroup,1:4])] \
                                  ,ignore_index=True, axis=1))
    df3 = np.array(df3)
    return df3


try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
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
chloroquineGroup = msGroup['chloroquineGroup']

msset = msGroup['msset']
msset2 = msGroup['msset2']
del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

skiplist = restrictionGroup + lowGroup

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup
pslset = pslGroup + shamGroup + adenosineGroup

#painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup']
#nonpainGroup = msGroup['salineGroup'] 

def msGrouping_pain_vs_itch(msdata): # psl만 처리
    msdata = np.array(msdata)
    
    df3 = pd.DataFrame(msdata[highGroup,1]) 
    df3 = pd.concat([df3, pd.DataFrame(msdata[chloroquineGroup,1:3])], ignore_index=True, axis = 1)
    df3 = np.array(df3)
    
    return df3

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

def mslinear_regression(x,y):
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m

    return m, b # bx+a
    
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

def nanex(array1):
    array1 = np.array(array1)
    array1 = array1[np.isnan(array1)==0]
    return array1

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))
      
# long, short separate
#msshort = 42; mslong = 97; 
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

# load 할 model 경로(들) 입력
# index, project
project_list = []

#project_list.append(['0116_CFA_l2_1', 100, None])
#project_list.append(['0116_CFA_l2_2', 100, None])
#project_list.append(['0116_CFA_l2_3', 100, None])

#project_list.append(['control2_roiroi', 200, None])
#project_list.append(['control2_roiroi', 100, None])
##
project_list.append(['control_test_segment_adenosine_set1', 100, None])
project_list.append(['control_test_segment_adenosine_set2', 200, None])
project_list.append(['control_test_segment_adenosine_set3', 300, None])
project_list.append(['control_test_segment_adenosine_set4', 400, None])
project_list.append(['control_test_segment_adenosine_set5', 500, None])
project_list.append(['control_test_segment_adenosine_set6', 600, None])
#project_list.append(['control_test_segment_adenosine_set7', 700, None])
#project_list.append(['control_test_segment_adenosine_set8', 800, None])

#project_list.append(['202012_withCFA_1', 100, None])
#project_list.append(['202012_withCFA_2', 200, None])
#project_list.append(['202012_withCFA_3', 300, None]) # chroloquine 추가후 첫 test임.
#project_list.append(['202012_withCFA_4', 400, None]) # chroloquine 추가후 첫 test임.

#project_list.append(['200224_half_nonseg_1', 100, None])
#project_list.append(['200224_half_nonseg_2', 200, None])

#project_list.append(['200224_half_seg_1', 100, None])
#project_list.append(['200224_half_seg_2', 200, None])

#project_list.append(['200226_0.75_segv2_1', 100, None])
#project_list.append(['200226_0.75_segv2_2', 200, None])

#project_list.append(['20200302_basevsitch_1', 100, None])
#project_list.append(['20200302_basevsitch_2', 200, None])

#project_list.append(['20200302_painitch_1', 100, None])
#project_list.append(['20200302_painitch_2', 200, None])
#project_list.append(['20200302_painitch_3', 300, None]) # acc_thr 증가
#project_list.append(['20200302_painitch_4', 400, None])

#project_list.append(['20200304_basic_1', 100, None])
#project_list.append(['20200304_basic_2', 200, None]) 
#project_list.append(['20200304_basic_3', 300, None]) 
#project_list.append(['20200304_basic_4', 400, None]) 

#project_list.append(['20200305_badmove_1', 111, None])

#project_list.append(['20200308_itch_vs_before', 333, None])
#project_list.append(['20200308_itch_vs_before2', 333, None])
#project_list.append(['20200308_itch_vs_before3', 333, None])
#

model_name = project_list 
             
t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
        # 개별 thr로 relu 적용되어있음. frame은 signal과 syn가 다름

# In[] testsw3
        
def msGrouping_base_vs_itch(msdata): # psl만 처리
    msdata = np.array(msdata)
    
    df3 = pd.DataFrame(msdata[salineGroup,:]) 
    df3 = pd.concat([df3, pd.DataFrame(msdata[chloroquineGroup,0:3])], ignore_index=True, axis = 1)
    df3 = np.array(df3)
    return df3
        
testsw3_mean = np.zeros((N,5,len(model_name)))
for ix, p in enumerate(model_name):
    for SE in range(N):
        loadpath5 = savepath + 'result\\' + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
        if os.path.isfile(loadpath5):
            with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                testsw3 = pickle.load(f)
            testsw3_mean[SE,:,ix] = testsw3[SE,:]
            
testsw3_mean = np.mean(testsw3_mean, axis=2)

biRNN_short = testsw3_mean

Aprism_biRNN2_pain_vs_itch = msGrouping_base_vs_itch(testsw3_mean)

# In[]
min_mean_save = []; [min_mean_save.append([]) for k in range(N)]
ROImean_save = []; [ROImean_save.append([]) for k in range(N)]
roiRatio = 1
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist: # ETC 추가후 lidocine skip 삭제할것 (여러개)
#        print(SE, 'skip')
        continue

    sessionNum = 5
    if SE in se3set:
        sessionNum = 3
    
    [min_mean_save[SE].append([]) for k in range(sessionNum)]
    [ROImean_save[SE].append([]) for k in range(sessionNum)]
    for se in range(sessionNum):
        current_value = []; result_mean_projects = []
        for i in range(len(model_name)): # repeat model 만큼 반복 후 평균냄
            ssw = False
            
            loadpath5 = savepath + 'result\\' + model_name[i][0] + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
            loadpath_mean = savepath + 'result\\' + model_name[i][0] + '\\exp_raw\\' + 'PSL_result_mean_' + str(SE) + '.pickle'
                
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    PSL_result_save = pickle.load(f)
                PSL_result_save2 = PSL_result_save[SE][se] # [BINS][ROI][bins] # BINS , full length 넘어갈때, # bins는 full length 안에서
                
                # ROI 평균처리에 대하여 및 반복처리
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
                    # ROI 평균처리에 대하여 - 끝
                    
            if os.path.isfile(loadpath_mean):
                with open(loadpath_mean, 'rb') as f:  # Python 3: open(..., 'rb')
                    result_mean = pickle.load(f)
                result_mean_projects.append(np.array(result_mean[SE][se])) # [BINS][bins][nonpain,pain]
 
        if len(current_value) > 0:
            current_value = np.mean(np.array(current_value), axis=0) # mean by project
            if sw == 'binarization':
                min_mean_save[SE][se] = current_value # [BINS][bins]
            elif sw == 'probability':
                min_mean_save[SE][se] = current_value > 0.5 # [BINS][bins]
        elif len(current_value) == 0:
            min_mean_save[SE][se] = np.nan
            
        if np.array(result_mean_projects).shape[0] > 0:
            result_mean_projects2 = np.mean(np.array(result_mean_projects), axis=0) # mean by project
            ROImean_save[SE][se] = result_mean_projects2[:,0,1]
        elif np.array(result_mean_projects).shape[0] == 0:
            ROImean_save[SE][se] = np.nan
               
# In[]
# mean, for shortlist (formalin, capsaicin) - short + long = all 로 사용
       
#       ROImean_save        # average test 
#       min_mean_save       # individual test
            
#calc_target = np.array(ROImean_save)
calc_target = np.array(min_mean_save)

            
biRNN_short = np.zeros((N,5)); biRNN_short[:] = np.nan;
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist:
#            print(SE, 'skip')
        continue
    sessionNum = 5
    if SE in se3set:
        sessionNum = 3
        
    for se in range(sessionNum):
#        if [SE, se] in shortlist:
        biRNN_short[SE,se]  = np.mean(calc_target[SE][se]) # [BINS][bins]
        # In[]
biRNN_long_subset = np.zeros((N,5)); biRNN_long_subset[:] = np.nan
for SE in range(N):
    if SE in np.array(msset_total)[:,0]:
        settmp = np.array(msset_total)[np.where(np.array(msset_total)[:,0]==SE)[0][0],:]
        biRNN_long_subset[SE,:] = np.nanmean(biRNN_short[settmp,:],axis=0)
        print('set averaging', settmp)
    elif SE not in np.array(msset_total).flatten(): 
        biRNN_long_subset[SE,:] = biRNN_short[SE,:]

# In ## PRISM 정리 및 통계처리
Aprism_biRNN2_formalin = msGrouping_nonexclude(biRNN_long_subset)
Aprism_biRNN2_capsaicin = biRNN_long_subset[capsaicinGroup,0:3]
Aprism_biRNN2_CFA = biRNN_long_subset[CFAgroup,0:3]
Aprism_biRNN2_psl = msGrouping_pslOnly(biRNN_long_subset)
Aprism_biRNN2_pain_vs_itch = msGrouping_pain_vs_itch(biRNN_long_subset)

# In
# 통계처리 출력용
# PSL
ms_statistics = pd.DataFrame([])

sham1 = biRNN_long_subset[shamGroup,1]; sham1 = sham1[np.isnan(sham1)==0]
sham2 = biRNN_long_subset[shamGroup,2]; sham2 = sham2[np.isnan(sham2)==0]     
       
psl0 = biRNN_long_subset[pslGroup,0]; psl0 = psl0[np.isnan(psl0)==0]
psl1 = biRNN_long_subset[pslGroup,1]; psl1 = psl1[np.isnan(psl1)==0]
psl2 = biRNN_long_subset[pslGroup,2]; psl2 = psl2[np.isnan(psl2)==0]
      
unpaired1 = stats.ttest_ind(sham1, psl1)[1]
unpaired2 = stats.ttest_ind(sham2, psl2)[1]
paired1 = stats.ttest_ind(psl0, psl1)[1]
paired2 = stats.ttest_ind(psl0, psl2)[1]

# Capsaicin
msname = pd.DataFrame(['sham_d3 vs PSL_d3', 'sham_d10 vs PSL_d10', 'PSL_base vs PSL_d3', 'PSL_base vs PSL_d10'])
stat = pd.DataFrame([unpaired1, unpaired2, paired1, paired2])

ms_statistics = pd.concat([ms_statistics, msname, stat]
           ,ignore_index=True, axis = 1)

cap0 = nanex(biRNN_long_subset[capsaicinGroup,0])
cap1 = nanex(biRNN_long_subset[capsaicinGroup,1])
cap2 = nanex(biRNN_long_subset[capsaicinGroup,2])
paired_cap0 = stats.ttest_rel(cap0, cap1)[1]
paired_cap2 = stats.ttest_rel(cap2, cap1)[1]

ms_statistics = pd.concat([ms_statistics, pd.DataFrame(['cap_before vs cap', paired_cap0]).T \
                           , pd.DataFrame(['cap_after vs cap', paired_cap2]).T] \
,ignore_index=True, axis = 0)


# CFA
CFA0 = nanex(biRNN_long_subset[CFAgroup,0])
CFA1 = nanex(biRNN_long_subset[CFAgroup,1])
CFA2 = nanex(biRNN_long_subset[CFAgroup,2])
paired_CFA1 = stats.ttest_rel(CFA0, CFA1)[1]
paired_CFA2 = stats.ttest_rel(CFA0, CFA2)[1]

ms_statistics = pd.concat([ms_statistics, pd.DataFrame(['CFA base vs CFA d1', paired_CFA1]).T \
                           , pd.DataFrame(['CFA base vs CFA d3', paired_CFA2]).T] \
,ignore_index=True, axis = 0)


print(ms_statistics)

# In[] movement 정리

movement_subset = np.zeros((N,5)); movement_subset[:] = np.nan
for SE in range(N):
    if SE in np.array(msset_total)[:,0]:
        settmp = np.array(msset_total)[np.where(np.array(msset_total)[:,0]==SE)[0][0],:]
        movement_subset[SE,:] = np.nanmean(movement[settmp,:],axis=0)
        print('set averaging', 'movement', settmp)
    elif SE not in np.array(msset_total).flatten(): 
        movement_subset[SE,:] = movement[SE,:]
        
Aprism_mov_biRNN2_formalin = msGrouping_nonexclude(movement_subset)
Aprism_mov_biRNN2_capsaicin = movement_subset[capsaicinGroup,0:3]
Aprism_mov_biRNN2_CFA = movement_subset[CFAgroup,0:3]
Aprism_mov_biRNN2_psl = msGrouping_pslOnly(movement_subset)
Aprism_mov_biRNN2_yohimbine = msGrouping_yohimbine(movement_subset)

# total activity 정리

t4_subset = np.zeros((N,5)); t4_subset[:] = np.nan
for SE in range(N):
    if SE in np.array(msset_total)[:,0]:
        settmp = np.array(msset_total)[np.where(np.array(msset_total)[:,0]==SE)[0][0],:]
        t4_subset[SE,:] = np.nanmean(t4[settmp,:],axis=0)
        print('set averaging', 'movement', settmp)
    elif SE not in np.array(msset_total).flatten(): 
        t4_subset[SE,:] = t4[SE,:]
        
Aprism_t4_biRNN2_formalin = msGrouping_nonexclude(t4_subset)
Aprism_t4_biRNN2_capsaicin = t4_subset[capsaicinGroup,0:3]
Aprism_t4_biRNN2_CFA = t4_subset[CFAgroup,0:3]
Aprism_t4_biRNN2_psl = msGrouping_pslOnly(t4_subset)
Aprism_t4_biRNN2_yohimbine = msGrouping_yohimbine(t4_subset)


# In[] 시간에 따른 통증확률 시각화 (작업중)
minsize=[]; painindex=[]; resultsave=[]                
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist:
        continue
    
    if not(SE in pslset): # psl만 시각화
        continue
    
    sessionNum = 5
    if SE in se3set:
        sessionNum = 3
     
    for se in range(sessionNum):
        result_BINS = np.mean(calc_target[SE][se], axis=1) # [BINS][bins]
        
        if result_BINS.shape[0] < 5:
            continue
        
        resultsave.append(result_BINS)
        minsize.append(result_BINS.shape[0])
        pix = 0
        if SE in pslGroup and se in [1,2]:
            pix = 1
        painindex.append(pix)
print('최소 BINS', np.min(minsize))        
        
for i in range(len(resultsave)):
    resultsave[i] = resultsave[i][:np.min(minsize)]
resultsave = np.array(resultsave)
painindex = np.array(painindex)

ix0 = np.where(painindex==0)[0] # nonpain
ix1 = np.where(painindex==1)[0] # pain
ix2 = np.concatenate((ix0[-32:], ix1), axis=0)

plt.figure(1, figsize=(9.7*1, 6*1))
plt.imshow(resultsave[ix2], cmap='hot')
plt.colorbar()
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + str(SE) + '_heatmap.png', dpi=1000)

# In[]
import os
os.sys.exit()


# In[] ROC plot (동시)
def msacc(class0, class1, mslabel='None', figsw=False):
    pos_label = 1; roc_auc = -np.inf; fig = None
    while roc_auc < 0.5:
        class0 = np.array(class0); class1 = np.array(class1)
        class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
        
        anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
        predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)
        #            
        fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
        
        maxix = np.argmax((1-fpr) * tpr)
        specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
        accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
        roc_auc = metrics.auc(fpr,tpr)
        
        if roc_auc < 0.5:
            pos_label = 0
            
    if figsw:
        sz = 0.9
        fig = plt.figure(1, figsize=(7*sz, 5*sz))
        lw = 2
        plt.plot(fpr, tpr, lw=lw, label = (mslabel + ' ' + str(round(roc_auc,2))))
        plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate', fontsize=20)
#        plt.ylabel('True Positive Rate', fontsize=20)
#        plt.title('ROC')
        plt.legend(loc="lower right", prop={'size': 15})
#        plt.show()
            
    return roc_auc, accuracy, fig

# ROC1
# (t4, event amp)
# formalin only 
target = np.array(Aprism_t4_biRNN2_formalin)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:4].flatten()) + list(target[:,4]) + list(target[:,6]) \
+ list(target[:,8]) + list(target[:,10])
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Mean activity, AUC:', figsw=True)   

with open('formalin_event_detection.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    formalin_event = pickle.load(f)
    formalin_event_detection = formalin_event['Aprism_amplitude_formalin']
    formalin_frequency_detection = formalin_event['Aprism_frequency_formalin']

target = np.array(formalin_event_detection)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:4].flatten()) + list(target[:,4]) + list(target[:,6]) \
+ list(target[:,8]) + list(target[:,10]) 
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Event amplitude, AUC:', figsw=True)

target = np.array(formalin_frequency_detection)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:4].flatten()) + list(target[:,4]) + list(target[:,6]) \
+ list(target[:,8]) + list(target[:,10]) 
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Event frequency, AUC:', figsw=True)  

plt.savefig(savepath2 + 'ROC_fig1.png', dpi=1000)

# ROC2
# # (bRNN), AA, AI, IA, II
# formalin only
 
# ROC3
# bRNN
# Capsaicin, CFA, PSL

# In[] Movement, t4 corr, Fig1 H

target = np.array(movement)
paindata = target[highGroup+midleGroup,1]
nonpaindata = list(target[highGroup+midleGroup,0]) + list(target[highGroup+midleGroup,2]) + \
list(target[salineGroup,:].flatten())

target2 = np.array(t4)
paindata2 = target2[highGroup+midleGroup,1]
nonpaindata2 = list(target2[highGroup+midleGroup,0]) + list(target2[highGroup+midleGroup,2]) + \
list(target2[salineGroup,:].flatten())

plt.figure(0, figsize=(1.2*4, 1*4))
plt.scatter(paindata, paindata2, s=3) # x:mov, y:t4
m, b = mslinear_regression(paindata, paindata2)
xaxis = np.arange(0,1.1,0.1)
plt.plot(xaxis, xaxis*m+b, label='pain')

plt.scatter(nonpaindata, nonpaindata2, s=3) # x:mov, y:t4
m, b = mslinear_regression(nonpaindata, nonpaindata2)
plt.plot(xaxis, xaxis*m+b, label='nonpain')

#plt.xlabel('Movement ratio', fontsize=20)
#plt.ylabel('Mean activity (df/f0)', fontsize=20)
plt.axis([0, 0.8, 0, 1.2])
plt.legend(prop={'size': 15})
plt.savefig(savepath2 + 't4_mov_corr', dpi=1000)

from scipy import stats
print('t4, movement corr', stats.pearsonr(nonpaindata, nonpaindata2))


# event amplitude, frequency를 movement와 corr 비교

target = np.array(formalin_event_detection)
corr_t = np.concatenate((target[:-1,4], target[:-1,6], target[:8,8], target[:8,10]), axis=0).flatten()
tmp1 = np.array(movement)[highGroup + highGroup2 + midleGroup, 0]
tmp2 = np.array(movement)[highGroup + highGroup2 + midleGroup, 2]
corr_m = np.concatenate((tmp1, tmp2), axis=0)
print('amp, movement corr', stats.pearsonr(corr_t, corr_m))
plt.figure()
plt.scatter(corr_t, corr_m)

target = np.array(formalin_frequency_detection)
corr_t = np.concatenate((target[:-1,4], target[:-1,6], target[:8,8], target[:8,10]), axis=0).flatten()
tmp1 = np.array(movement)[highGroup + highGroup2 + midleGroup, 0]
tmp2 = np.array(movement)[highGroup + highGroup2 + midleGroup, 2]
corr_m = np.concatenate((tmp1, tmp2), axis=0)
print('amp, movement corr', stats.pearsonr(corr_t, corr_m))
plt.figure()
plt.scatter(corr_t, corr_m)
# '안나오네..'


























