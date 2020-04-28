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
    gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
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
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']
behavss2 = msdata_load['behavss2']
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
itSalineGroup = msGroup['itSalineGroup']
itClonidineGroup = msGroup['itClonidineGroup']
ipsaline_pslGroup = msGroup['ipsaline_pslGroup']

msset = msGroup['msset']
msset2 = msGroup['msset2']
del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

skiplist = restrictionGroup + lowGroup

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup
pslset = pslGroup + shamGroup + adenosineGroup

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
        plt.legend(loc="lower right", prop={'size': 15})
            
    return roc_auc, accuracy, fig


def nanex(array1):
    array1 = np.array(array1)
    array1 = array1[np.isnan(array1)==0]
    return array1

def msGrouping_base_vs_itch(msdata): # psl만 처리
    msdata = np.array(msdata)
    
    df3 = pd.DataFrame(msdata[salineGroup,:]) 
    df3 = pd.concat([df3, pd.DataFrame(msdata[chloroquineGroup,0:3])], ignore_index=True, axis = 1)
    df3 = np.array(df3)
    return df3

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))
      
# long, short separate
#msshort = 42; mslong = 97; 
bins = 10

t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
        # 개별 thr로 relu 적용되어있음. frame은 signal과 syn가 다름
                    
# In[] Formalin CV
 
if True:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model2\\'               
    project_list = []

    project_list.append(['foramlin_only_1', 100, None])
    project_list.append(['foramlin_only_2', 200, None]) 
    project_list.append(['foramlin_only_3', 300, None]) 
    project_list.append(['foramlin_only_4', 400, None])
    project_list.append(['foramlin_only_5', 500, None]) 
  
    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model2 = np.nanmean(testsw3_mean, axis=2)
                
    # In[] PSL용 load
if True:
    t = 10
    testsw3_mean = np.zeros((N,5,t)); testsw3_mean[:] = np.nan         
    for i in range(t):
        path1 = 'D:\\mscore\\syncbackup\\google_syn\\model3\\'
        path2 = 'fset + baseonly + CFAgroup + capsaicinGroup_0.69_0415_t' + str(i) + '.h5'
        path3 = path1+ path2
        
        if os.path.isfile(path3):
            with open(path3, 'rb') as f:  # Python 3: open(..., 'rb')
                testsw3 = pickle.load(f)
                testsw3_mean[:,:,i] = testsw3
    model3 = np.nanmean(testsw3_mean, axis=2)
    
        # In[] model4 load
if True:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model4\\'               
    project_list = []

    project_list.append(['model4_roiroi_formalin_cap_cfa_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model4 = np.nanmean(testsw3_mean, axis=2)
    

# In[] label 재정렬 movement 
target = np.array(movement)
for SE in range(N):
    if SE in [141,142,143]:
        target[SE,1:3] = target[SE,3:5] 
        target[SE,1:3] = np.nan
        
    if SE in [146,149]:
        target[SE,3:] = np.nan
movement_filter = np.array(target)
        
# In[]

def dict_gen(target):
    target = np.array(target)
    for SE in range(N):
        if SE in [141,142,143]:
            target[SE,1:3] = target[SE,3:5] 
            target[SE,1:3] = np.nan
            
        if SE in [146,149]:
            target[SE,3:] = np.nan
      
     
    # subset 평균처리        
    subset_mean = np.zeros((N,5)); subset_mean[:] = np.nan
    for SE in range(N):
        if SE in np.array(msset_total)[:,0]:
            settmp = np.array(msset_total)[np.where(np.array(msset_total)[:,0]==SE)[0][0],:]
            subset_mean[SE,:] = np.nanmean(target[settmp,:],axis=0)
    #        print('set averaging', settmp)
        elif SE not in np.array(msset_total).flatten(): 
            subset_mean[SE,:] = target[SE,:]
    
    # grouping
    
    cap0 = nanex(subset_mean[capsaicinGroup,0])
    cap1 = nanex(subset_mean[capsaicinGroup,1])
    
    CFA0 = nanex(subset_mean[CFAgroup,0])
    CFA1 = nanex(subset_mean[CFAgroup,1])
    CFA2 = nanex(subset_mean[CFAgroup,2])
    
    sham0 = nanex(subset_mean[shamGroup,0])
    sham1 = nanex(subset_mean[shamGroup,1])
    sham2 = nanex(subset_mean[shamGroup,2])
           
    psl0 = nanex(subset_mean[pslGroup,0])
    psl1 = nanex(subset_mean[pslGroup,1])
    psl2 = nanex(subset_mean[pslGroup,2])
    
    ipsaline0 = nanex(subset_mean[ipsaline_pslGroup,0])
    ipsaline1 = nanex(subset_mean[ipsaline_pslGroup,1])
    ipsaline2 = nanex(subset_mean[ipsaline_pslGroup,2])
    ipsaline3 = nanex(subset_mean[ipsaline_pslGroup,3])
    ipsaline4 = nanex(subset_mean[ipsaline_pslGroup,4])
    
    model2_dict = {'cap0': cap0, 'cap1': cap1, \
                   'CFA0': CFA0, 'CFA1': CFA1, 'CFA2': CFA2, \
                   'sham0': sham0, 'sham1': sham1, 'sham2': sham2, \
                   'psl0': psl0, 'psl1': psl1, 'psl2': psl2, \
                   'ipsaline0': ipsaline0, 'ipsaline1': ipsaline1, 'ipsaline2': ipsaline2, 'ipsaline3': ipsaline3, 'ipsaline4': ipsaline4, \
                   }
    
    return model2_dict

model2_dict = dict_gen(model2)
model3_dict = dict_gen(model3)
model4_dict = dict_gen(model4)


# cap, cfa prism, ROC
def capcfa_roc_gen(target, name):
    tdict = dict(target)
    pain = np.concatenate((tdict['cap1'], tdict['CFA1'], tdict['CFA2']), axis=0)
    nonpain = np.concatenate((tdict['cap0'], tdict['CFA0']), axis=0)
    roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=True)
    
def Aprim_capcfa_gen(taget):
    tdict = dict(target)
    base_merge = np.concatenate((tdict['cap0'], tdict['CFA0']), axis=0)
    Aprism = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(tdict['cap1']) \
                                  , pd.DataFrame(tdict['CFA1']) , pd.DataFrame(tdict['CFA2'])], ignore_index=True, axis=1)
    return Aprism
    
capcfa_roc_gen(model2_dict, 'Model2')
capcfa_roc_gen(model4_dict, 'Model4')
Aprism_biRNN_capcfa = Aprim_capcfa_gen(model2_dict)


# psl prism, ROC
def psl_roc_gen(target, name):
    tdict = dict(target)
    pain = np.concatenate((tdict['psl1'], tdict['psl2']), axis=0)
    nonpain = np.concatenate((tdict['psl0'], tdict['sham0'], tdict['sham1'], tdict['sham2']), axis=0)
    roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=True)
    
psl_roc_gen(model2_dict, 'Model2')
psl_roc_gen(model3_dict, 'Model3')

savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'psl_roc', dpi=1000)

# prsim 용으로 사용
#    base_merge = np.concatenate((tdict['psl0'], tdict['sham0']), axis=0)
#    Aprism_biRNN_psl= pd.concat([pd.DataFrame(base_merge), pd.DataFrame(tdict['sham1']) \
#                                  , pd.DataFrame(tdict['sham2']) , pd.DataFrame(tdict['psl1']), \
#                                  pd.DataFrame(tdict['psl2'])], ignore_index=True, axis=1)


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


# In[] 시간에 따른 통증확률 시각화
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


# ROC1, bRNN, PSL (sham (d3, d10) + before (sham, psl)) vs psl (d3, d10)


pain = np.concatenate((psl1, psl2), axis=0)
nonpain = np.concatenate((psl0, sham0, sham1, sham2), axis=0)
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Mean activity, AUC:', figsw=True)


# ROC1
# (t4, event amp)
# formalin only 
target = np.array(Aprism_t4_biRNN2_formalin)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:4].flatten()) + list(target[:,4]) + list(target[:,6]) \
+ list(target[:,8]) + list(target[:,10])
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Mean activity, AUC:', figsw=True)   


# formalin event amplitude and frequecny ROC curve

with open('formalin_event_detection.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    formalin_event = pickle.load(f)
    formalin_event_detection = formalin_event['Aprism_amplitude_formalin']
    formalin_frequency_detection = formalin_event['Aprism_frequency_formalin']

target = np.array(formalin_event_detection)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:2].flatten()) + list(target[:,4]) + list(target[:,8]) # interphase 제거됨
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Event amplitude, AUC:', figsw=True)

target = np.array(formalin_frequency_detection)
pain = list(target[:,5]) + list(target[:,9])
nonpain = list(target[:,0:2].flatten()) + list(target[:,4]) + list(target[:,8]) # interphase 제거됨
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

# In[]

























