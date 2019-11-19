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

def msGrouping_nonexclude(msmatrix): # base 예외처리 없음, goruping된 sample만 뽑힘
    target = np.array(msmatrix)
    
    df3 = pd.DataFrame(target[highGroup]) 
    df3 = pd.concat([df3, pd.DataFrame(target[midleGroup]), \
                     pd.DataFrame(target[salineGroup]), \
                     pd.DataFrame(target[ketoGroup]), pd.DataFrame(target[lidocainGroup]), \
                     pd.DataFrame(target[yohimbineGroup]), pd.DataFrame(target[capsaicinGroup][:,0:3]), \
                     pd.DataFrame(target[pslGroup][:,0:3]), pd.DataFrame(target[shamGroup][:,0:3])], ignore_index=True, axis = 1)
        
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

painGroup = msGroup['highGroup'] + msGroup['ketoGroup'] + msGroup['midleGroup'] + msGroup['yohimbineGroup']
nonpainGroup = msGroup['salineGroup'] 

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

# excluded
#highGroup.remove(1)

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
model_name = []

model_name.append(['1118_direct_2/', 2])
#model_name.append(['1107_2class/', 1])
##model_name.append(['1111_2class/', 1])
#model_name.append(['1015_binfix_1/', 1])
#model_name.append(['1015_binfix_2/', 2])
#model_name.append(['1117_2class/', 2])
#model_name.append(['0903_seeding_2/', 4])
#model_name.append(['0903_seeding_2/', 5])

# In short, long test 1차 by signalss
msshort = 42; mslong = 97; bins = 10
shortlist = []; longlist = []
SE=79;se=2;i=0
for SE in range(N):
    for se in range(5):
        binNum = len(list(range(0,(np.array(signalss[SE][se]).shape[0] - 497) +1, bins)))
        minimum_binning = 42
        
        if binNum >= msshort-minimum_binning+1 and binNum < mslong-minimum_binning+1: # for 2 mins
            shortlist.append([SE,se])
#            print(SE, se, 'msshort')
        elif binNum >= mslong-minimum_binning+1: # for 4 mins
            longlist.append([SE,se])
#            print(SE, se, 'mslong')
        else:
            print('e')

# In min_mean_save에 모든 data 저장

min_mean_save = []
[min_mean_save.append([]) for k in range(N)]

# pointSvae - 2차 학습 label 판단에 사용하기 위해 예측 평균값 저장
pointSave = []
[pointSave.append([]) for k in range(N)]
for SE in range(N):
    if not SE in grouped_total_list or SE in restrictionGroup or SE in lowGroup:
#        print(SE, 'skip')
        continue

    sessionNum = 5
    if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
        sessionNum = 3
    
    [min_mean_save[SE].append([]) for k in range(sessionNum)]
    [pointSave[SE].append([]) for k in range(sessionNum)]
    msreport = True
    for se in range(sessionNum):
        # In
        # 문제아들, 67,74
        current_value = []
        for i in range(len(model_name)): # repeat model 만큼 반복 후 평균냄
            ssw = False
            
            loadpath5 = savepath + 'result\\' + model_name[i][0] + 'exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
          
            if os.path.isfile(loadpath5):
                ssw = True
            else:
                loadpath5 = savepath + 'result\\' + model_name[i][0] + 'exp_raw\\' + \
                model_name[i][0][:-1] + '_PSL_result_' + str(SE) + '.pickle'
                if os.path.isfile(loadpath5):
                    ssw = True
                else:
                    if msreport:
                        msreport = False
                        print(SE, 'skip')
            
            if ssw:
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    PSL_result_save = pickle.load(f)
            
                if [SE, se] in shortlist:
                    PSL_result_save2 = PSL_result_save[SE][se][:msshort-minimum_binning+1]
                elif [SE, se] in longlist:
                    PSL_result_save2 = PSL_result_save[SE][se][:mslong-minimum_binning+1]
                else:
                    print('e')
                    
                current_value.append(np.array(PSL_result_save2) > 0.5 ) # PSL_result_save2[BINS][ROI][bins][nonpain,pain]
                
            # # 2진화 
        if len(current_value) > 0:
            current_value = np.mean(np.array(current_value), axis=0) # 모든 반복 project에 대해서 평균처리함
            binNum = current_value.shape[0] # [BINS]
            
            # 시계열 형태로 표현용 
            empty_board = np.zeros((binNum, minimum_binning + binNum-1))
            empty_board[:,:] = np.nan
    #            print(empty_board.shape)
            
            # 2min mean 형태로 표현용  
            # 20191017, 2 mins 형태로는 사용하지 않고, 42 bins에서 겹치는 time windows를 평균내어 사용하고 있음.
            # 즉 msplot은 안쓴단 말임
    #        msplot = []
            
            BIN = 0;
            [pointSave[SE][se].append([]) for u in range(binNum)]
            for BIN in range(binNum):
                plotsave = np.mean(current_value[BIN,:,:,1], axis=0) 
                empty_board[BIN, BIN: BIN+minimum_binning] = plotsave
                
                pointSave[SE][se][BIN] = np.mean(plotsave)
                
    #            figNum = int(str(SE)+str(se))
    #            msplot.append(np.mean(plotsave))
                
            empty_board = np.nanmean(empty_board, axis=0)
            
#            if False:
#                plt.plot(empty_board) # 2 mins window의 시계열을 그대로 남겨, 모든 bins를 평균내어 연결
#                plt.ylim(0,1)
#                
#                plt.plot(maxsave)
#                
#                binning = list(range(0,(signalss[SE][se].shape[0] - 497) +1, bins))
#                
#                for top in range(5):
#                    startat = binning[np.argsort(np.array(maxsave))[::-1][top]]
#                    print('startat', startat)
#                    print('top', top, maxsave[np.argsort(np.array(maxsave))[::-1][top]])
#                
#    #            plt.figure() 
#    #            plt.plot(msplot) # 2 mins window를 평균내서 각각 플롯 
#    #            plt.ylim(0,1)
                
            # In
                
            # plotsave와 empty_board는 별개로 생각할 수 있음. 
            # 당장은 empty_board는 사용하지 않음 
            
            min_mean_save[SE][se] = empty_board
        elif len(current_value) == 0:
            min_mean_save[SE][se] = np.nan

if False:
    with open('pointSave.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(pointSave, f, pickle.HIGHEST_PROTOCOL)
        print('pointSave.pickle 저장되었습니다.')
        
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
    out = np.zeros((N,5))
    
    for SE in range(N):
        if SE == 70:
            out[SE,:] = (target[70,:] + target[72,:])/2
        elif SE == 72:
            out[SE,:] = np.nan
        else:
            out[SE,:] = target[SE,:]
            
    return out

def relu_optimize(min_mean_save):
    print('relu optimize를 실행합니다.')
    mssave = []
    for optimizedthr in np.arange(0.1,0.5,0.001):
        biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
        biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan
        for SE in range(N):
            if not SE in grouped_total_list or SE in restrictionGroup or SE in lowGroup:
    #            print(SE, 'skip')
                continue
            
            sessionNum = 5
            if SE in capsaicinGroup or SE in pslGroup or SE in shamGroup:
                sessionNum = 3
        
            for se in range(sessionNum):
                min_mean_mean = np.array(min_mean_save[SE][se])
                if not np.isnan(np.nanmean(min_mean_mean)):
    
                    biRNN_2[SE,se] = np.mean(np.array(min_mean_mean) > optimizedthr)
                    
        msloss = msloss2(biRNN_2)
        mssave.append([optimizedthr, msloss])
#        print(optimizedthr, msloss)
#    plt.plot(np.array(mssave)[:,0], np.array(mssave)[:,1])
    return np.array(mssave)[:,0][np.nanargmax(np.array(mssave)[:,1])]
        
def msloss2(biRNN_2):
    biRNN_22 = ms_batchmean(biRNN_2) # 중복 데이터 처리.. 평균을 내든 뺴든 ...        
    PSL_acc = report(biRNN = biRNN_22, PLStest=True)
            
    pslbase = biRNN_22[pslGroup,:][:,0]; pslbase = pslbase[np.isnan(pslbase)==0]
    pslday3 = biRNN_22[pslGroup,:][:,1]; pslday3 = pslday3[np.isnan(pslday3)==0]
    pslday10 = biRNN_22[pslGroup,:][:,2]; pslday10 = pslday10[np.isnan(pslday10)==0]
    sham = biRNN_22[shamGroup,:][:,0:3].flatten()
                
    base_3 = stats.ttest_rel(pslbase, pslday3)[1]
    base_10 = stats.ttest_rel(pslbase, pslday10)[1]
    sham_3 = stats.ttest_ind(sham, pslday3)[1]
    sham_10 = stats.ttest_ind(sham, pslday10)[1]
    
    msindex1 = (PSL_acc * 10) - (base_3 + base_10 + sham_3 + sham_10)
    
    return msindex1
    
# 모든 optimal, method 다 고려해서 추천
def otimal_msduration(msduration, min_mean_save, optimalSW=False):    
    # min_mean_save로 부터 각 value를 계산함
#    PSL_acc = None
    min_mean_save = min_mean_save
    exceptlist = [[70, 0], [71, 0], [72, 0]] # 2분 촬영 psl은 제외 한다.
    minimal_size = 497
          
    biRNN_2 = np.zeros((N,5)); movement_497_2 = np.zeros((N,5)); t4_497_2 = np.zeros((N,5))
    biRNN_2[:] = np.nan; movement_497_2[:] = np.nan; t4_497_2[:] = np.nan
    
    ## 우선 평균값으로 채워넣고,
    for SE in range(N):
        if not SE in grouped_total_list or SE in restrictionGroup or SE in lowGroup:
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
    
    method_num = 4
    msindex = []; [msindex.append([]) for u in range(method_num)]
    
    # test 0: maxpeak
    for SE in range(N):
        for se in range(3):
#            c1 = SE in pslGroup + shamGroup
            c2 = se in [0,1,2]
            c3 = [SE, se] in longlist
            
            if c3 and c2:
                min_mean_mean = np.array(min_mean_save[SE][se])
                # method1
                maxix = np.argmax(min_mean_mean)
                biRNN_2[SE,se] = min_mean_mean[maxix]
    
    msindex[0] = msloss2(biRNN_2)

    # test 1: mean
    for SE in range(N):
        for se in range(3):
    #            c1 = SE in pslGroup + shamGroup
            c2 = se in [0,1,2]
            c3 = [SE, se] in longlist
            
            if c3 and c2:
                min_mean_mean = np.array(min_mean_save[SE][se])
                biRNN_2[SE,se] = np.mean(min_mean_mean, axis=0)
                
    msindex[1] = msloss2(biRNN_2)
    
    # test 2: timewindow
    mssave = []; forlist = list(range(1, 300)) # 앞뒤가 nan이 찍히는 모든 범위로 설정 할 것 
    print('test2: timewindow 최적화를 시작합니다.')
    for mssec in forlist:
#        print(mssec)
        skipsw = False
        msduration = int(round((((mssec*FPS)-82)/10)+1))
        for SE in range(N):
            for se in range(3):
        #            c1 = SE in pslGroup + shamGroup
                c2 = se in [0,1,2]
                c3 = [SE, se] in longlist
                
                if c3 and c2:
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
    print('optimized time window', msduration)
    for SE in range(N):
        for se in range(3):
    #            c1 = SE in pslGroup + shamGroup
            c2 = se in [0,1,2]
            c3 = [SE, se] in longlist
            
            if c3 and c2:
                min_mean_mean = np.array(min_mean_save[SE][se])
                
                if min_mean_mean.shape[0] - msduration <= 0 or msduration < 1:
                    skipsw = True
                    break
                
                meansave = []
                for msbin in range(min_mean_mean.shape[0] - msduration):
                    meansave.append(np.mean(min_mean_mean[msbin: msbin+msduration]))
                    
                maxix = np.argmax(meansave)
                biRNN_2[SE,se] = np.mean(min_mean_mean[maxix: maxix+msduration], axis=0)
            
                    
    msindex[2] = msloss2(biRNN_2)

                    
    # test 3: relu 
    othr = relu_optimize(min_mean_save)
    for SE in range(N):
        for se in range(3):
    #            c1 = SE in pslGroup + shamGroup
            c2 = se in [0,1,2]
            c3 = [SE, se] in longlist
            
            if c3 and c2:
                min_mean_mean = np.array(min_mean_save[SE][se])
                biRNN_2[SE,se] = np.mean(np.array(min_mean_mean) > optimizedthr)
                
    msindex[3] = msloss2(biRNN_2)
    
    print(msindex)






                    
                    movement_497_2[SE,se] = np.mean(bahavss[SE][se])
                    
                    meansignal = np.mean(np.array(signalss[SE][se]),axis=1)
                    t4_497_2[SE,se] = np.mean(meansignal,axis=0)
             
            elif np.isnan(np.nanmean(min_mean_mean)):
                biRNN_2[SE,se] = np.nan
                movement_497_2[SE,se] = np.nan
                t4_497_2[SE,se] = np.nan
                
            
    # Aprism 변수 update
    accuracy = None
    if optimalSW:
        biRNN_22 = ms_batchmean(biRNN_2)            
        target = biRNN_22
#        painGroup = painGroup
#        nonpainGroup = nonpainGroup
#        pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
#        pain = target[pslGroup,1:3].flatten()
    #    nonpain_within = target[pslGroup,0]
#        accuracy, roc_auc, fig = accuracy_cal(pain, nonpain_between)
        tmp = target[pslGroup,1:3]; tmp = tmp[np.isnan(tmp)==0]
        diff = np.mean(target[painGroup,1]) - np.mean(tmp)
        
        PSL_acc = report(biRNN = biRNN_22, PLStest=True)
        
        pslbase = biRNN_22[pslGroup,:][:,0]; pslbase = pslbase[np.isnan(pslbase)==0]
        pslday3 = biRNN_22[pslGroup,:][:,1]; pslday3 = pslday3[np.isnan(pslday3)==0]
        pslday10 = biRNN_22[pslGroup,:][:,2]; pslday10 = pslday10[np.isnan(pslday10)==0]
        
        day3p = stats.ttest_rel(pslbase, pslday3)[1]
        day10p = stats.ttest_rel(pslbase, pslday10)[1]
                
        print(msduration, 'Formalin_acc', accuracy, 'PSL_acc', PSL_acc, 'diff', diff)
        print(msduration, 'day3p', day3p, 'day10p', day10p)
    
        
    return accuracy, biRNN_2, t4_497_2, movement_497_2, PSL_acc

# In[]

# 현재 사용중인 test 분석방식은 peak time window
# 즉 특정 x time window 동안의 값을 평균내 모든 값들 중 최대값을 그 session의 value로 사용하여 평가함.
# 계산은 control group도 동일하게 적용된느 것은 맞지만, x를 최적화 할때 평가결과를 가지고 결정하는 문제가 있다.
# 이 계산이 bias가 아님을 증명하기 위해, time window x가 특별한 값이 아니고, 대충 아무거나 써도 마찬가지의 결과를 냄을
# x에 따른 acc를 보여줌으로써 어필해야 한다. -> x는 일반화 가능성이 높다. = 매우 특정한 값이 아니다.   
    
#axiss = []; [axiss.append([]) for u in range(2)]
#for msduration in range(2,20):
#    accuracy, _, _, _, PSL_acc= otimal_msduration(msduration, min_mean_save, optimalSW=True) # 전체 값 모두 사용 
#    axiss[0].append(msduration)
#    axiss[1].append(PSL_acc)
#plt.plot(axiss[0], axiss[1])


#

# 특정한 time window x를 잡고 계산 시작
#  time window x를 초로 환산하면 (approximately) mssec과 같다
mssec = 25
# (82 + ((msduration-1) * 10)) / FPS = sec
msduration = int(round((((mssec*FPS)-82)/10)+1))

if False:
    for mssec in range(1, 120):
        msduration = int(round((((mssec*FPS)-82)/10)+1))
        print(mssec, msduration)
    
print('msduration', msduration)

#msduration = 25 # 의미없음 
_, biRNN_2, t4_497_2, movement_497_2, _ = otimal_msduration(msduration, min_mean_save) # 2 mins, or 4 mins

# 예외규정: 70, 72는 동일 생쥐의 반복 측정이기 때문에 평균처리한다.
biRNN_22 = ms_batchmean(biRNN_2)
t4_497_22 = ms_batchmean(t4_497_2)
movement_497_22 = ms_batchmean(movement_497_2)

Aprism_biRNN = msGrouping_nonexclude(biRNN_22)
Aprism_total = msGrouping_nonexclude(t4_497_22)
Aprism_movement = msGrouping_nonexclude(movement)           

report(biRNN = biRNN_22)

        
# In[] optimal thr

painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

targetMatrix = np.array(biRNN_2)
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


# In[] 20191008 원천보고용 figure 작성 

# biRNN
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(biRNN_22, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy, roc_auc, fig = accuracy_cal(pain, nonpain, True) 
fig.savefig('birnn_roc.png', dpi=1000) 

# t4

t4_497 = np.zeros((N,5)); movement_497 = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4_497[SE,se] = np.mean(signalss[SE][se][:497,:])
        movement_497[SE,se] = np.mean(bahavss[SE][se][:int(round(497/4.3*64)),0])

painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(t4_497, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy, roc_auc, fig = accuracy_cal(pain, nonpain, True)
fig.savefig('test.png', dpi=1000) 


# movement
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement_497_22, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)   


# high vs saline

# biRNN
painGroup = highGroup
nonpainGroup = salineGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(biRNN_2, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)    

# t4
painGroup = highGroup
nonpainGroup = salineGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(t4_497_2, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)   


# movement
painGroup = highGroup
nonpainGroup = salineGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement_497_2, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)   


# 각 session 별 움직임과 t4 corr
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement_497, painGroup, nonpainGroup)
pain_movement = np.array(pain)
nonpain_movement = np.concatenate((nonpain_within, nonpain_between))

pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(t4_497, painGroup, nonpainGroup)
pain_t4 = np.array(pain)
nonpain_t4 = np.concatenate((nonpain_within, nonpain_between))

axiss = []; [axiss.append([]) for u in range(2)]

m, b = mslinear_regression(nonpain_movement, nonpain_t4)
print(m, b)
sc = 0.7
plt.figure(1, figsize=(9.7*sc, 6*sc))
plt.scatter(nonpain_movement, nonpain_t4, s = 2)
plt.xlabel('movement ratio mean')
plt.ylabel('mean of calcium intensity')

xaxis = np.arange(0,0.5,0.5/100)
plt.plot(xaxis, xaxis*m + b, c = 'orange')
plt.xlim([0,0.5]), plt.ylim([0,1])

import scipy
r, p = scipy.stats.pearsonr(nonpain_movement, nonpain_t4)
print('r square', r**2, 'p value', p)
plt.savefig('tmp.png', dpi=1000)

 # In[] movement와 total ROC
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
print('movment--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement_497_22, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup
print('total--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(t4_497_22, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

# In[]
target = biRNN_2
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
print('pslGroup_early ', np.nanmean(target[pslGroup ,1]))
print('pslGroup_late ', np.nanmean(target[pslGroup ,2]))

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

# PSL - within
painGroup = pslGroup
nonpainGroup = salineGroup + lidocainGroup
# PSL은 inter가 late pain이므로 특수처리해야함.
print('PSL - within')
pain = target[pslGroup,1:3].flatten()
nonpain_within = target[pslGroup,0]
accuracy_cal(pain, nonpain_within, True)

# PSL - between
painGroup = pslGroup
nonpainGroup = salineGroup + lidocainGroup
_, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('PSL - between')
pain = target[pslGroup,1:3].flatten()
accuracy_cal(pain, nonpain_between, True)

# In[] 움직임은 speicificity에 문제가있다 - new

# 2 class 상황, 적은 숫자인 pain class의 갯수 x 만큼 not pain class에서 랜덤하게 sampling 함
# 각 class sample 등의 평균 movement와, accuracy를 x, y plot
# OLS fitting 함

def movement_specificity():
    target = np.array(biRNN_22)
    movement = np.array(movement_497_22)
    
    painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup + yohimbineGroup
    nonpainGroup = salineGroup + lidocainGroup
    
    pain_movement, nonpain_within_movement, nonpain_between_movement = pain_nonpain_sepreate(movement, painGroup, nonpainGroup)
    nonpain_movement = np.concatenate((nonpain_within_movement, nonpain_between_movement), axis=0)
    
    axiss = []; [axiss.append([]) for i in range(2)]
    totalNum = nonpain_movement.shape[0]
    sampleNum = pain_movement.shape[0]
    epochs = 50000
    for epoch in range(epochs):
        if epoch % int(epochs/10) == 1:
            print(epoch, '/', epochs)
    #    random_N = random.randrange(2, sampleNum)
        ixlist = random.sample(list(range(totalNum)), sampleNum)
        
        accuracy, _, _ = accuracy_cal(pain_movement, nonpain_movement[ixlist], False)
        axiss[0].append(np.mean(nonpain_movement[ixlist], axis=0))
        axiss[1].append(accuracy)
        
        t1 = np.mean(nonpain_movement[ixlist], axis=0)
        t2 = accuracy
        
        if np.isnan(t1) or np.isnan(t2):
            print('breaking')
            break
            
    for d in range(2):
        axiss[d] = np.array(axiss[d])
    #    axiss[d] = axiss[d][np.isnan(axiss[d])==0];
        
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
    plt.savefig('tmp1.png', dpi=1000)
    
    # 대조군
    
    pain_target, nonpain_within_target, nonpain_between_target = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
    nonpain_target = np.concatenate((nonpain_within_target, nonpain_between_target), axis=0)
    
    
    axiss = []; [axiss.append([]) for i in range(2)]
    totalNum = nonpain_movement.shape[0]
    sampleNum = pain_movement.shape[0]
    
    #epochs = 50000
    for epoch in range(epochs):
        if epoch % int(epochs/10) == 1:
            print(epoch, '/', epochs)
            
    #    random_N = random.randrange(2, sampleNum)
        ixlist = random.sample(list(range(totalNum)), sampleNum)
        
        accuracy, _ , _= accuracy_cal(pain_target, nonpain_target[ixlist], False)
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
    plt.savefig('tmp2.png', dpi=1000)

if False:
    for sec in range(19, 60, 5):
        msduration = int(round((((mssec*FPS)-82)/10)+1))
        print('msduration', msduration, 'sec', sec)
        _, biRNN_2, t4_497_2, movement_497_2 = otimal_msduration(msduration, min_mean_save) # 2 mins, or 4 mins
        biRNN_22 = ms_batchmean(biRNN_2)
        t4_497_22 = ms_batchmean(t4_497_2)
        movement_497_22 = ms_batchmean(movement_497_2)
        movement_specificity()
        
movement_specificity()        
# In[] 취약점 분석
# dev에서 쓸모있어서 그냥 두지만, 사용 전에 체크할것 
# predict_matrix_total
painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup
nonpainGroup = salineGroup + lidocainGroup

targetMatrix = np.array(biRNN_2)
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
        
    
    
    
    
    
    
    














