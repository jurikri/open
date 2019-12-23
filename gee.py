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

def msGrouping_nonexclude(formalin, psl): # base 예외처리 없음, goruping된 sample만 뽑힘
    short = np.array(formalin); long = np.array(psl)
    
    control = []; [control.append([]) for u in range(2)]
    for u in longlist:
        SE = u[0]; se = u[1]
        c1 = SE in highGroup + midleGroup + yohimbineGroup + ketoGroup + lidocainGroup and se in [0,2,4]
        c2 = SE in capsaicinGroup and se in [0,2]
        c3 = SE in pslGroup and se in [0]
        c4 = SE in shamGroup and se in [0] # sham session은 제외되어야함. 
        c5 = SE in salineGroup and se in [0,1,2,3,4]
        
        if c1 or c2 or c5:
            control[0].append(long[SE,se]) # not psl
            
        if c3 or c4:
            control[1].append(long[SE,se]) # psl

    df3 = pd.DataFrame(short[highGroup]) 
    df3 = pd.concat([df3, pd.DataFrame(short[midleGroup]), \
                     pd.DataFrame(short[salineGroup]), \
                     pd.DataFrame(short[ketoGroup]), pd.DataFrame(short[lidocainGroup]), \
                     pd.DataFrame(short[yohimbineGroup]), pd.DataFrame(short[capsaicinGroup][:,0:3]), \
                     pd.DataFrame(control[0]), pd.DataFrame(control[1]), \
                     pd.DataFrame(long[pslGroup][:,1:3]), pd.DataFrame(long[shamGroup][:,1:3].flatten())], ignore_index=True, axis = 1)
        
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
pslGroup = msGroup['pslGroup']
shamGroup = msGroup['shamGroup']
adenosineGroup = msGroup['adenosineGroup']
highGroup2 = msGroup['highGroup2']

msset = msGroup['msset']
del msGroup['msset']
skiplist = restrictionGroup + lowGroup + lidocainGroup

painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup']
nonpainGroup = msGroup['salineGroup'] 

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

# excluded
#highGroup.remove(1)

# mouselist는 training에 사용됩니다.
#mouselist = []
#mouselist += msGroup['highGroup']
#mouselist += msGroup['ketoGroup']
#mouselist += msGroup['midleGroup']
#mouselist += msGroup['salineGroup']
#mouselist += msGroup['yohimbineGroup']
#mouselist += [msGroup['lidocaineGroup'][0]]
#etc = msGroup['lidocaineGroup'][0]
#mouselist.sort()

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

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))

# load 할 model 경로(들) 입력
# index, project
project_list = []
# index, proejct
#model_name.append(['1128_binfix5_1', 5])
#model_name.append(['1128_binfix5_2', 5])
#model_name.append(['1126_binfix2_saline', 3])
#model_name.append(['1205_duplicated_add_1', 100])
#model_name.append(['1205_duplicated_add_2', 200])
#model_name.append(['1207_recovery_except_1', 100])
#model_name.append(['1207_recovery_except_2', 200])
#model_name.append(['1207_recovery_except_3', 300])

#model_name.append(['1217_adenosine_1', 100, None])

project_list.append(['1217_adenosine_1', 100, None])
project_list.append(['1217_adenosine_2', 200, None])
project_list.append(['1217_adenosine_3', 500, None])
project_list.append(['1217_adenosine_4', 600, None])

model_name = project_list 
# In short, long test 1차 by signalss
#msshort = 42; mslong = 97; 
bins = 10
shortlist = []; longlist = []
for SE in range(N):
    if SE in mouseGroup:
        if not SE in skiplist:
            sessionNum = 5
            if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
                sessionNum = 3
            
            for se in range(sessionNum):
                length = np.array(signalss[SE][se]).shape[0]
                if length > 180*FPS:
                    longlist.append([SE,se])
                elif length < 180*FPS:
                    shortlist.append([SE,se])
                else:
                    print('error')

# In min_mean_save에 모든 data 저장
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
    if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
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
                        print(SE, 'skip')
            
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
        
                        current_BINS.append(np.mean(np.array(current_ROI), axis=0)) # ROI 평균
                    current_value.append(current_BINS)
                    
#        # add-hoc, 길이통일, 추후 삭제되어야 함
#        lensave = []
#        for u in range(len(current_value)):
#            tmp = len(current_value)
#            for k in range(len(current_value[u])): 
#                if type(current_value[u][k]) != np.ndarray:
#                    if np.isnan(current_value[u][k]):
#                        tmp = k
#                        break
#            lensave.append(tmp)
#                    
#        current_value2 = []
#        for u in range(len(current_value)):
#            current_value2.append(np.array(current_value[u][:np.min(lensave)])[0,:])
#            # # 2진화 
        if len(current_value) > 0:
            current_value = np.mean(np.array(current_value), axis=0) # 모든 반복 project에 대해서 평균처리함
  
            binNum = current_value.shape[0] # [BINS]
            mslength = current_value.shape[1]
            
            # 시계열 형태로 표현용 
            empty_board = np.zeros((binNum, mslength + (binNum-1)))
            empty_board[:,:] = np.nan
    #       
#            [pointSave[SE][se].append([]) for u in range(binNum)]
            for BIN in range(binNum):
                plotsave = current_value[BIN,:]
                empty_board[BIN, BIN:BIN+mslength] = plotsave
                
#                print(BIN, np.mean(plotsave))
                
                
                
#                pointSave[SE][se][BIN] = np.mean(plotsave)

            empty_board = np.nanmean(empty_board, axis=0)
            min_mean_save[SE][se] = empty_board
            
#            if SE == 72:
#                plt.figure()
#                plt.plot(empty_board)
#                plt.ylim(0,1)
            
        elif len(current_value) == 0:
            min_mean_save[SE][se] = np.nan

if False:
    savename2 = '1122_driect_cut'
    with open(savename2 + '_pointSave.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(pointSave, f, pickle.HIGHEST_PROTOCOL)
        print(savename2 + '_pointSave.pickle 저장되었습니다.')
        
min_mean_save = min_mean_save
#    exceptlist = [[70, 0], [71, 0], [72, 0]] # 2분 촬영 psl은 제외 한다.
#    minimal_size = 497
          
biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan

## 우선 평균값으로 채워넣고,
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist:
#            print(SE, 'skip')
        continue
    
    sessionNum = 5
    if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
        sessionNum = 3
        
    for se in range(sessionNum):
        min_mean_mean = np.array(min_mean_save[SE][se])
        if not np.isnan(np.nanmean(min_mean_mean)):
            biRNN_2[SE,se] = np.mean(min_mean_mean, axis=0)
            
#                startat = 10*maxix
            movement_497_2[SE,se] = np.mean(bahavss[SE][se])
            
            meansignal = np.mean(np.array(signalss[SE][se]),axis=1)
            t4_497_2[SE,se] = np.mean(meansignal,axis=0)
            
Aprism_biRNN2_pslOnly = msGrouping_pslOnly(biRNN_2)
             
        
# In[]
        
# maxix 적용전
def msfilter(target, msfilter):
    msfilter = msfilter
    target = np.array(target)
    target2 = np.zeros((N,5)); target2[:] = np.nan
    for SE in range(N):
        for se in range(5):
            if [SE, se] in msfilter:
                target2[SE,se] = target[SE, se]
                
    return target2
        
def report(biRNN, PLStest=False):
    # 평가용 함수, formalin pain과 PSL within- between 평가함
    
    target = biRNN
    
    if not(PLStest):
        print('Formalin')
        painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
        nonpainGroup = salineGroup + lidocainGroup
        pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(msfilter(target, shortlist), painGroup, nonpainGroup)
        nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
        print('pain #', pain.shape[0], 'nonpain #', nonpain.shape[0])
        _, _, fig = accuracy_cal(pain, nonpain, True)
        fig.savefig('ROC_formalin.png', dpi=1000)
        
        print('Capsaicin')
        painGroup = capsaicinGroup
        nonpainGroup = salineGroup + lidocainGroup
        pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(msfilter(target, shortlist), painGroup, nonpainGroup)
        nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
        pain = target[capsaicinGroup,1].flatten()
        print('pain #', pain.shape[0], 'nonpain #', nonpain.shape[0])
        _, _, fig = accuracy_cal(pain, nonpain, True)
        fig.savefig('ROC_capsaicin.png', dpi=1000)
    
#    if PLStest:
#        print('PSL전용으로 계산됩니다.')
    #  우선은 , 4mins sample이 없으니깐, nonpain의 모든 그룹을 다 사용하자
    painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup + ketoGroup
    nonpainGroup = salineGroup + yohimbineGroup
    _, nonpain_within, nonpain_between = pain_nonpain_sepreate(msfilter(target, longlist), painGroup, nonpainGroup)
    nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
    nonpain = nonpain[np.isnan(nonpain)==0]
    
    # nonpain에 psl base 추가 
    nonpain2 = msfilter(target, longlist)[pslGroup][:,0]
    nonpain2 = nonpain2[np.isnan(nonpain2)==0]
    nonpain = np.concatenate((nonpain, nonpain2), axis=0)
    
    # nonpain에 psl sham 추가 
    nonpain3 = msfilter(target, longlist)[shamGroup].flatten()
    nonpain3 = nonpain3[np.isnan(nonpain3)==0]
    nonpain = np.concatenate((nonpain, nonpain3), axis=0)
    
    pain = target[pslGroup,1:3].flatten()
    
    
    if not(PLStest):
        print('PSL')
        print('pain #', pain.shape[0], 'nonpain #', nonpain.shape[0])
        acc, _, fig = accuracy_cal(pain, nonpain, True)
        fig.savefig('ROC_PSL.png', dpi=1000)
    
    if PLStest:
        acc, _, fig = accuracy_cal(pain, nonpain, False)
    
    return acc

def ms_batchmean(target):
    target = np.array(target)
    out = np.zeros((N,5)); out[:] = np.nan

    for SE in mouseGroup:
        if not SE in np.array(msset).flatten():
            out[SE,:] = target[SE,:]
    
    for u in range(len(msset)):
        mstmp = []
        for k in msset[u]:
            mstmp.append(target[k,:])
        out[msset[u][0],:] = np.nanmean(mstmp, axis=0)
           
    return out

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

def relu_optimize(min_mean_save, relu=True):
    if relu:
        print('relu optimize를 실행합니다.')
    elif not(relu):
        print('binarization optimize를 실행합니다.')
        
    mssave = []
    for optimizedthr in np.arange(0,0.5,0.001):
        biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
        biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan
        for SE in range(N):
            sessionNum = 5
            if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
                sessionNum = 3
            for se in range(sessionNum):       
                if [SE, se] in longlist:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                if not np.isnan(np.nanmean(min_mean_mean)):
                    min_mean_mean = np.array(min_mean_mean)
                    
                    if relu:
                        min_mean_mean[min_mean_mean < optimizedthr] = 0  
                    elif not(relu):
                        min_mean_mean = min_mean_mean > optimizedthr
                        
                    biRNN_2[SE,se] = np.mean(min_mean_mean)
                    
        msloss = msloss2(biRNN_2)
        mssave.append([optimizedthr, msloss])
#        print(optimizedthr, msloss)
#    plt.plot(np.array(mssave)[:,0], np.array(mssave)[:,1])
    return np.array(mssave)[:,0][np.nanargmax(np.array(mssave)[:,1])]
        

################################ 수정요망, 개발용과 최종 처리용을 따로..  
def msloss2(biRNN_2):
#    biRNN_22 = ms_batchmean(biRNN_2) # 중복 데이터 처리.. 평균을 내든 뺴든 ...
    biRNN_22 = ms_batch_ind(biRNN_2) # 단일쥐에서 평균내지 않을 때
    PSL_acc = report(biRNN = biRNN_22, PLStest=True)
            
    pslbase = biRNN_22[pslGroup,:][:,0]; 
    nanixbase = np.isnan(pslbase) == 0

    pslday3 = biRNN_22[pslGroup,:][:,1]
    nanixday3 = np.isnan(pslday3) == 0
    
    pslday10 = biRNN_22[pslGroup,:][:,2]
    nanixday10 = np.isnan(pslday10) == 0
    
    cix = nanixbase * nanixday3 * nanixday10
    
    sham = biRNN_22[shamGroup,:][:,0:3].flatten()
    sham = sham[np.isnan(sham) == 0]
                
    base_3 = stats.ttest_rel(pslbase[cix], pslday3[cix])[1]
    base_10 = stats.ttest_rel(pslbase[cix], pslday10[cix])[1]
    sham_3 = stats.ttest_ind(sham, pslday3[cix])[1]
    sham_10 = stats.ttest_ind(sham, pslday10[cix])[1]
    
    msindex1 = (PSL_acc * 10) - (base_3 + base_10 + sham_3 + sham_10)
    
    c1 = np.mean(pslday3[cix]) > np.mean(pslbase[cix]) and np.mean(pslday3[cix]) > np.mean(sham)
    c2 = np.mean(pslday10[cix]) > np.mean(pslbase[cix]) and np.mean(pslday10[cix]) > np.mean(sham)
    if not(c1 and c2):
        msindex1 = 0
    
    return msindex1
    
# 모든 optimal, method 다 고려해서 추천
def otimal_msduration(min_mean_save):    
    # min_mean_save로 부터 각 value를 계산함
#    PSL_acc = None
    min_mean_save = min_mean_save
#    exceptlist = [[70, 0], [71, 0], [72, 0]] # 2분 촬영 psl은 제외 한다.
#    minimal_size = 497
          
    biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
    biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan
    
    ## 우선 평균값으로 채워넣고,
    for SE in range(N):
        if not SE in grouped_total_list or SE in skiplist:
#            print(SE, 'skip')
            continue
        
        sessionNum = 5
        if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
            sessionNum = 3
            
        for se in range(sessionNum):
            min_mean_mean = np.array(min_mean_save[SE][se])
            if not np.isnan(np.nanmean(min_mean_mean)):
                biRNN_2[SE,se] = np.mean(min_mean_mean, axis=0)
                
#                startat = 10*maxix
                movement_497_2[SE,se] = np.mean(bahavss[SE][se])
                
                meansignal = np.mean(np.array(signalss[SE][se]),axis=1)
                t4_497_2[SE,se] = np.mean(meansignal,axis=0)
             
    ## 그다음 PSL을 특수 처리한다.
    
    method_num = 7
    msindex = []; [msindex.append([]) for u in range(method_num)]
    
    def mstest0():
        # test 0: maxpeak
        for SE in range(N):
            for se in range(3):
    #            c1 = SE in pslGroup + shamGroup
                c2 = se in [0,1,2]
                c3 = [SE, se] in longlist
                
                if c3 and c2:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    if not(len(np.array(min_mean_mean).shape) == 0):
                        maxix = np.argmax(min_mean_mean)
                        biRNN_2[SE,se] = min_mean_mean[maxix]
                        
        return msloss2(biRNN_2), biRNN_2
        
    msindex[0], _ = mstest0() 
    
    
    def mstest1():
        # test 1: mean
        for SE in range(N):
            for se in range(3):
                c1 = [SE, se] in longlist
                
                if c1:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    biRNN_2[SE,se] = np.mean(min_mean_mean, axis=0)

                    
        return msloss2(biRNN_2), biRNN_2
                    
    msindex[1], _ = mstest1()
    
    def mstest2():
        # test 2: timewindow
        mssave = []; forlist = list(range(1, 300)) # 앞뒤가 nan이 찍히는 모든 범위로 설정 할 것 
        print('test2: timewindow 최적화를 시작합니다.')
        for mssec in forlist:
    #        print(mssec)
            skipsw = False
            msduration = int(round((((mssec*FPS)-82)/10)+1))
            for SE in range(N):
                for se in range(3):
                    c1 = [SE, se] in longlist
                    
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
        
            msacc = msloss2(biRNN_2)
    #        print(msacc)
            if not(skipsw):
                mssave.append(msacc)
            elif skipsw:
                mssave.append(np.nan)
                              
        mssec = forlist[np.nanargmax(mssave)]
        msduration = int(round((((mssec*FPS)-82)/10)+1))
        print('optimized time window, mssec', mssec)
        for SE in range(N):
            for se in range(3):
                c2 = se in [0,1,2]
                c3 = [SE, se] in longlist
                
                if c3 and c2:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    
                    meansave = []
                    for msbin in range(min_mean_mean.shape[0] - msduration):
                        meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
                        
                    maxix = np.argmax(meansave)
                    biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
                    
        return msloss2(biRNN_2), biRNN_2, msduration
    
    msindex[2], _, msduration = mstest2()

    # test 3: relu
    def mstest3():
        biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
        othr = relu_optimize(min_mean_save)
        for SE in range(N):
            for se in range(5):       
                if [SE, se] in longlist:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    min_mean_mean[min_mean_mean < othr] = 0
                    biRNN_2[SE,se] = np.mean(np.array(min_mean_mean))
                    
        return msloss2(biRNN_2), biRNN_2
                
    msindex[3], biRNN_2 = mstest3()
    print('mstest3', msindex[3])
    
    # test 4: binarization
#    def mstest4():
#        biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
#        othr = relu_optimize(min_mean_save, relu=False)
#        for SE in range(N):
#            for se in range(5):       
#                if [SE, se] in longlist:
#                    min_mean_mean = np.array(min_mean_save[SE][se])[:98]
#                    min_mean_mean = min_mean_mean > othr
#                    biRNN_2[SE,se] = np.mean(np.array(min_mean_mean))
#                    
#        return msloss2(biRNN_2), biRNN_2
    
#    msindex[4], _ = mstest4()
#    print('mstest4', msindex[4])
    
    # window x relu, window x binarization
    def mstest5(relu=True):
        # test 2: timewindow 
        forlist = list(range(16, 239)) # 앞뒤가 nan이 찍히는 모든 범위로 설정 할 것 
        forlist2 = list(np.arange(0,0.38,0.01))
        
        mssave = np.zeros((len(forlist),len(forlist2))); mssave[:] = np.nan # shape (windows, thr)
        
        if relu:
            print('test5: timewindow x relu 최적화를 시작합니다.')
        elif not relu:
            print('test6: timewindow x binarization 최적화를 시작합니다.')
        
        for i1, opThr in enumerate(forlist2): # thr
            print('i1', i1)
            for i2, mssec in enumerate(forlist): # timewindow
                
                biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
        #        print(mssec)
                msduration = int(round((((mssec*FPS)-82)/10)+1))
                if msduration <= 0:
                    continue
                
                skipsw = False
                for SE in range(N):
                    for se in range(5): 
                        if [SE, se] in longlist:
                            min_mean_mean = np.array(min_mean_save[SE][se])
                            
                            if min_mean_mean.shape[0] - msduration < 0:
                                skipsw = True
                                break
                            
                            meansave = []
                            forlist_t1 = range(min_mean_mean.shape[0] - msduration)
                            if min_mean_mean.shape[0] - msduration == 0:
                                forlist_t1 = [0]
                                
                            for msbin in forlist_t1:
                                tmp = min_mean_mean[msbin: msbin+msduration]
                                if relu:
                                    tmp[tmp < opThr] = 0
                                elif not relu:
                                    tmp = tmp > opThr
                                
                                meansave.append(np.mean(tmp))
                                
                            maxix = np.argmax(meansave)
                            tmp = min_mean_mean[maxix: maxix+msduration]
                            if relu:
                                tmp[tmp < opThr] = 0
                            elif not relu:
                                tmp = tmp > opThr
                            
                            biRNN_2[SE,se] = np.mean(tmp)
        
                if not(skipsw):
                    msacc = msloss2(biRNN_2)
                    mssave[i2,i1] = msacc
                elif skipsw:
                    mssave[i2,i1] = np.nan
                    
        opThr = forlist2[np.unravel_index(np.nanargmax(mssave, axis=None), mssave.shape)[1]]
        mssec = forlist[np.unravel_index(np.nanargmax(mssave, axis=None), mssave.shape)[0]]
        biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
        msduration = int(round((((mssec*FPS)-82)/10)+1))
        
        for SE in range(N):
            for se in range(5):       
                if [SE, se] in longlist:
                    min_mean_mean = np.array(min_mean_save[SE][se])
                    
                    if min_mean_mean.shape[0] - msduration < 0:
                        skipsw = True
                        break
                    
                    meansave = []
                    forlist_t1 = range(min_mean_mean.shape[0] - msduration)
                    if min_mean_mean.shape[0] - msduration == 0:
                        forlist_t1 = [0]
                        
                    for msbin in forlist_t1:
                        tmp = min_mean_mean[msbin: msbin+msduration]
                        if relu:
                            tmp[tmp < opThr] = 0
                        elif not relu:
                            tmp = tmp > opThr
                        
                        meansave.append(np.mean(tmp))
                        
                    maxix = np.argmax(meansave)
                    tmp = min_mean_mean[maxix: maxix+msduration]
                    if relu:
                        tmp[tmp < opThr] = 0
                    elif not relu:
                        tmp = tmp > opThr
                    
                    biRNN_2[SE,se] = np.mean(tmp)

        return msloss2(biRNN_2), biRNN_2
              
    msindex[5], _ = mstest5(relu=True)
    msindex[6], _ = mstest5(relu=False)   
    
    bestix = np.argmax(msindex)
    print(msindex)
    
    if bestix == 0:
        optimalmethod = 'maxpeak'
    elif bestix == 1:
        optimalmethod = 'all mean'
    elif bestix == 2:
        optimalmethod = 'window mean'
    elif bestix == 3:
        optimalmethod = 'relu mean'
    elif bestix == 4:
        optimalmethod = 'binarization'
    elif bestix == 5:
        optimalmethod = 'timewindow x relu'
    elif bestix == 6:
        optimalmethod = 'timewindow x binarization'
        
    print('optimalmethod >>', optimalmethod)

    if bestix == 0:
        _, biRNN_2 = mstest0()
    elif bestix  == 1:
        _, biRNN_2 = mstest1()
    elif bestix == 2:
        _, biRNN_2, _ = mstest2()
    elif bestix == 5:
        _, biRNN_2 = mstest5(relu=True)
    elif bestix == 6:
        _, biRNN_2 = mstest5(relu=False)  
        
#        for SE in range(N):
#            for se in range(5):
#                c2 = se in [0,1,2]
#                c3 = [SE, se] in longlist
#                
#                if c3 and c2:
#                    min_mean_mean = np.array(min_mean_save[SE][se])
#                    
#                    meansave = []
#                    for msbin in range(min_mean_mean.shape[0] - msduration):
#                        meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
#                        
#                    maxix = np.argmax(meansave)
#                    startat = 10*maxix
#                    meansignal = np.mean(np.array(signalss[SE][se]),axis=1)
#                    t4_497_2[SE,se] = np.mean(meansignal[startat:startat+(msduration*bins+82)])
                    
#    elif bestix == 3:
#        _, biRNN_2 = mstest3()
 
    return biRNN_2, t4_497_2, movement_497_2

# In[]

# 현재 사용중인 test 분석방식은 peak time window
# 즉 특정 x time window 동안의 값을 평균내 모든 값들 중 최대값을 그 session의 value로 사용하여 평가함.
# 계산은 control group도 동일하게 적용된느 것은 맞지만, x를 최적화 할때 평가결과를 가지고 결정하는 문제가 있다.
# 이 계산이 bias가 아님을 증명하기 위해, time window x가 특별한 값이 아니고, 대충 아무거나 써도 마찬가지의 결과를 냄을
# x에 따른 acc를 보여줌으로써 어필해야 한다. -> x는 일반화 가능성이 높다. = 매우 특정한 값이 아니다.   
    

#biRNN_2, t4_497_2, movement_497_2 = otimal_msduration(min_mean_save) # 2 mins, or 4 mins

# 예외규정: 70, 72는 동일 생쥐의 반복 측정이기 때문에 평균처리한다.
#biRNN_22 = ms_batchmean(biRNN_2)
#t4_497_22 = ms_batchmean(t4_497_2)
#movement_497_22 = ms_batchmean(movement_497_2)

# test 3: relu
def mstest3(): # 시간 상관없이 전체 평균을 relu로 처리함. long list 전용임.
    biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
    othr = relu_optimize(min_mean_save)
    print('optimized threshold', othr)
    for SE in range(N):
        for se in range(5):       
            if [SE, se] in longlist:
                min_mean_mean = np.array(min_mean_save[SE][se])
                min_mean_mean[min_mean_mean < othr] = 0
                biRNN_2[SE,se] = np.mean(np.array(min_mean_mean))
                
    return othr, biRNN_2
            
optiThr, biRNN_2_psl = mstest3(); biRNN_2_psl = ms_batch_ind(biRNN_2_psl)

def mstest4(): # shortlist 전용, 단순 평균
    biRNN_2 = np.zeros((N,5)); biRNN_2[:] = np.nan
    short_length = min_mean_save[shortlist[0][0]][shortlist[0][1]].shape[0]
    for SE in range(N):
        for se in range(5):       
            if [SE, se] in shortlist:
                min_mean_mean = np.array(min_mean_save[SE][se])
                biRNN_2[SE,se] = np.mean(np.array(min_mean_mean))
                
            for u in longlist:
                SE1 = u[0]; se1 = u[1]
                min_mean_mean = np.array(min_mean_save[SE1][se1])
                biRNN_2[SE1,se1] = np.mean(min_mean_mean[:short_length])
                
    return biRNN_2
            
biRNN_2_formalin = mstest4(); biRNN_2_formalin = ms_batch_ind(biRNN_2_formalin)

Aprism_biRNN = msGrouping_nonexclude(biRNN_2_formalin, biRNN_2_psl)

#
#t4_497_22 = ms_batch_ind(t4_497_2)
#movement_497_22 = ms_batch_ind(movement_497_2)
#
#
#
##Aprism_total = msGrouping_nonexclude(t4_497_22)
##Aprism_movement = msGrouping_nonexclude(movement)           
#
#report(biRNN = biRNN_22)
# In[] 시계열 시각화, 저장

RESULT_SAVE_PATH = savepath + '\\psl_visualization\\'
if not os.path.exists(RESULT_SAVE_PATH):
    os.mkdir(RESULT_SAVE_PATH)
    
min_mean_save = np.array(min_mean_save)
   
biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan

## 우선 평균값으로 채워넣고,
for SE in range(N):
    if not SE in grouped_total_list or SE in skiplist:
#            print(SE, 'skip')
        continue
    
    # PSL만 시각화
    if SE in pslGroup + shamGroup:
        sessionNum = 5
        if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
            sessionNum = 3
            
        for se in range(sessionNum):
            min_mean_mean = np.array(min_mean_save[SE][se])
            
            plt.figure()
            plt.plot(min_mean_mean, label = 'bRNN') 
            thrline = np.zeros(min_mean_mean.shape[0]); thrline[:] = optiThr
            plt.plot(thrline, label = 'relu_threshold')
            plt.ylim(0,1)
            
            plt.ylabel('Normalized index')
            plt.xlabel('Segmented frame')
            
            
            plt.legend()
            
            mouseID = str(SE) + '_'+ str(se)
            plt.title('mouseID  ' + mouseID)
            
            plt.savefig(RESULT_SAVE_PATH + mouseID + '_psl.png', dpi=1000)
            plt.close()
        
        
        # time window 맞춰서 t4, movement 추가. optimal window 시각화


############################################################# 
# 아래 부터는 임시,, 통계적 차이가 가장큰 방법으로 분석중
import sys
sys.exit()
# In[] thr, separation
bRNN_result = np.array(biRNN_2)

baselin_control = []
for u in longlist:
    SE = u[0]; se = u[1]
    
    c1 = SE in highGroup + midleGroup + yohimbineGroup + ketoGroup + lidocainGroup and se in [0,2,4]
    c2 = SE in capsaicinGroup and se in [0,2]
    c3 = SE in pslGroup and se in [0]
    c4 = SE in shamGroup and se in [0] # sham session [1, 2]은 제외되어야함. 
    c5 = SE in salineGroup and se in [0,1,2,3,4]
    
    if c1 or c2 or c3 or c4 or c5:
        baselin_control.append(bRNN_result[SE,se])
baselin_control = np.array(baselin_control)
psl = np.array(bRNN_result[pslGroup][:,1:3].flatten())
sham = np.array(bRNN_result[shamGroup][:,1:3].flatten())


# In[] 빈도로 표현 

axiss = []; [axiss.append([]) for u in range(5)]
sw = True
for thr in np.arange(0,0.3,0.001):
    ##

    
    axiss[0].append(thr)
    axiss[1].append(baselin_control[baselin_control>thr].shape[0]/baselin_control.shape[0])
    axiss[2].append(psl[psl>thr].shape[0]/psl.shape[0])
    axiss[3].append(sham[sham>thr].shape[0]/sham.shape[0])
    
    stats.ttest_ind((baselin_control>thr), (psl>thr))[1]
    axiss[4].append(stats.ttest_ind((sham>thr), (psl>thr))[1])
    
    if thr >= 0.12 and sw:
        sw = False
        print(thr, baselin_control[baselin_control>thr].shape[0]/baselin_control.shape[0], \
              psl[psl>thr].shape[0]/psl.shape[0], \
              sham[sham>thr].shape[0]/sham.shape[0])
    
plt.figure(0)
plt.plot(axiss[0], axiss[1], label='base')
plt.plot(axiss[0], axiss[2], label='PSL')
plt.plot(axiss[0], axiss[3], label='sham')
plt.legend()
plt.savefig('frequency', dpi=1000)


# optimized thr로 binarization 하여 표현
thr = axiss[0][np.nanargmin(axiss[4])]
print('optimized thr', thr)
Aprism_biRNN_highrank_thr = pd.concat([pd.DataFrame(baselin_control>thr, dtype=int), \
                                   pd.DataFrame(psl>thr, dtype=int), \
                                   pd.DataFrame(sham>thr, dtype=int)], ignore_index=True, axis=1)

plt.figure(1)
plt.plot(axiss[0], axiss[4], label='sham vs PSL')

# In[]
# relu
axiss = []; [axiss.append([]) for u in range(5)]
for thr in np.arange(0,0.3,0.001):
    
    msbase = np.array(baselin_control)
    msPSL = np.array(psl)
    mssham = np.array(sham)
    
    msbase[msbase<thr] = 0
    msPSL[msPSL<thr] = 0
    mssham[mssham<thr] = 0

    axiss[0].append(thr)
    axiss[1].append(stats.ttest_ind(msPSL, mssham)[1])
    
plt.figure(1)
plt.plot(axiss[0], axiss[1], label='sham vs PSL, relu')
thr = axiss[0][np.nanargmin(axiss[1])]
print('relu', 'optimized thr', thr, 'p value', np.nanmin(axiss[1]))


# In[] 상위 x %의 data만 사용
 
msdata = []; [msdata.append([]) for u in range(3)]
msdata[0] = baselin_control
msdata[1] = psl
msdata[2] = sham

axiss = []; [axiss.append([]) for u in range(4)]
for ratio in np.arange(1,0,-0.1):
    Aprism_biRNN_highrank = pd.DataFrame([])
    for i in range(3):
        Aprism_biRNN_highrank = pd.concat([Aprism_biRNN_highrank, pd.DataFrame(np.sort(msdata[i])[::-1][:int(round(msdata[i].shape[0] * ratio))])], \
                   ignore_index=True, axis=1)
    
    mstmp = np.array(Aprism_biRNN_highrank)
    
    base_vs_psl = stats.ttest_ind(mstmp[:,1][np.isnan(mstmp[:,1])==0], mstmp[:,0][np.isnan(mstmp[:,0])==0])[1]
    sham_vs_psl = stats.ttest_ind(mstmp[:,1][np.isnan(mstmp[:,1])==0], mstmp[:,2][np.isnan(mstmp[:,2])==0])[1]
    
    print(ratio, )

    axiss[0].append(ratio)
    axiss[1].append(base_vs_psl)
    axiss[2].append(sham_vs_psl)
    
plt.figure(1)
plt.plot(axiss[0], axiss[1], label='base vs PSL')
plt.plot(axiss[0], axiss[2], label='sham vs PSL')
plt.legend()
plt.savefig('highRank', dpi=1000)


bins = 40
plt.figure(2)

values, base = np.histogram(msdata[0], bins=bins)
cumulative = np.cumsum(values/len(msdata[0]))
plt.plot(base[:-1], cumulative)

values, base = np.histogram(msdata[1], bins=bins)
cumulative = np.cumsum(values/len(msdata[1]))
plt.plot(base[:-1], cumulative)

values, base = np.histogram(msdata[2], bins=bins)
cumulative = np.cumsum(values/len(msdata[2]))
plt.plot(base[:-1], cumulative)






# In[]
mslength = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        try:
            mslength[SE,se] = min_mean_save[SE][se].shape[0]
        except:
            pass












