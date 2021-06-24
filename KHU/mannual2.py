# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:24:00 2021

@author: MSBak
"""

import sys; 
msdir = 'C:\\Users\\skklab\\Documents\\mscode'; sys.path.append(msdir)
sys.path.append('D:\\mscore\\code_lab\\')
import msFunction
import os  
try: import pickle5 as pickle
except: import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import random
import time

gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)


FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열
signalss_raw = msdata_load['signalss_raw']

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
CFAgroup = msGroup['CFAgroup']
chloroquineGroup = msGroup['chloroquineGroup']
itSalineGroup = msGroup['itSalineGroup']
itClonidineGroup = msGroup['itClonidineGroup']
ipsaline_pslGroup = msGroup['ipsaline_pslGroup']
ipclonidineGroup = msGroup['ipclonidineGroup']
gabapentinGroup = msGroup['gabapentinGroup']

beevenomGroup =  msGroup['beevenomGroup']
oxaliGroup =  msGroup['oxaliGroup']
glucoseGroup =  msGroup['glucoseGroup']
PSLscsaline =  msGroup['PSLscsaline']
highGroup3 =  msGroup['highGroup3']
PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']

roiNum = np.zeros((N)) * np.nan
for SE in range(len(signalss)):
#    for se in range(len(signalss[SE])):
    roiNum[SE] = signalss[SE][0].shape[1]

print('np.nanmin(roiNum)', np.nanmin(roiNum))


#%% grouping

group_pain_training = []
group_nonpain_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    for se in range(10):
        painc, nonpainc, test_only = [], [], []
        
        # khu formalin
        painc.append(SE in list(range(230, 239)) and se in [1])
        painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
        painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])
        
        nonpainc.append(SE in list(range(230, 239)) and se in [0])
        nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
        nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
        nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
        
        # snu psl pain
        painc.append(SE in pslGroup and se in [1,2])
        
        # khu psl
        nonpainc.append(SE in PSLgroup_khu and se in [0])
        painc.append(SE in PSLgroup_khu and se in [1,2])
        
        # test only
#        test_only.append(SE in PSLgroup_khu and se in [1,2])
        
        if np.sum(np.array(painc)) > 0:
            group_pain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_pain_training.append([SE, se])
            
        if np.sum(np.array(nonpainc)) > 0:
            group_nonpain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_nonpain_training.append([SE, se])

def msROC(class0, class1):
    import numpy as np
    from sklearn import metrics
    
    pos_label = 1; roc_auc = -np.inf; fig = None

    class0 = np.array(class0); class1 = np.array(class1)
    class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
    
    anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
    predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)       
    fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
    maxix = np.argmax((1-fpr) * tpr)
    specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
    accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
    roc_auc = metrics.auc(fpr,tpr)
    
    return accuracy, roc_auc


def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

def ms_syn(target_signal=None, target_size=None):
    downratio = target_signal.shape[0] / target_size
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
    return allo

def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout

from tqdm import tqdm
MAXSE = 20
movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        movement_syn[SE][se] = downsampling(bahavss[SE][se], signalss[SE][se].shape[0])

# plt.plot(np.mean(signalss[257][5], axis=1))
#%%
from scipy import stats

if False:
    SE = 2; se = 0; thr = 0.4; ws = int((FPS * 3)/2)
    active_R_mean = msFunction.msarray([N, MAXSE])
    
    
for SE in range(N):
    if SE in [243, 244, 245, 246]:
        for se in range(len(signalss[SE])):
            if len(active_R_mean[SE][se]) == 0:
                mssignal = signalss[SE][se]
                meansignal = np.mean(mssignal, axis=1)
                ROInum = mssignal.shape[1]
                vix = np.where(movement_syn[SE][se] > 0)[0]
                timeline_R = np.zeros(mssignal.shape[0]) * np.nan
                
                print(); print(SE,se)
                for t in range(mssignal.shape[0]):
                    if len(mssignal[t-ws:t+ws]) == ws*2:
                        R_inROI = []
                        mean_data = meansignal[t-ws:t+ws]
                        for ROI in range(ROInum):
                            msdata = mssignal[t-ws:t+ws,ROI]
                            R = stats.pearsonr(msdata, mean_data)[0]
                            R_inROI.append(R)
                        timeline_R[t] = np.mean(R_inROI)
                                                
                plt.figure()
                plt.title(str(SE) + '_' + str(se))
                plt.plot(timeline_R)
                plt.plot(meansignal)
                
                vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*0.3)]
                
                
                print(se, np.nanmean(timeline_R[vix]))
                
# plt.plot(np.mean(signalss_raw[230][0], axis=1))
# plt.plot(np.mean(signalss_raw[230][1], axis=1))

#%%
path = 'C:\\mass_save\\roimannual\\'

mssignal = signalss[230][1]
for ROI in range(50, 80):
    plt.figure()
    plt.title(str(ROI))
    plt.plot(mssignal[:,ROI])




#%%
mssignal = np.array(signalss_raw[230][1])

def dynamic_base(mssignal, ws=20, bins=5, verbose=0):
    msbins = range(0, mssignal.shape[0]-ws, bins)
   
    baseratios = [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.5]
    losssave = []
    for baseratio in baseratios:
        msout = np.zeros(mssignal.shape) * np.nan
        for ROI in range(mssignal.shape[1]):
            roisignal = mssignal[:,ROI]
            mssave = []
            for b in msbins:
                mssave.append(np.std(roisignal[b:b+ws]))
            msstd = np.median(mssave)
            base = np.mean(np.sort(roisignal)[0:int(round(len(roisignal)*baseratio))])
            msout[:, ROI] = (roisignal - base)/msstd
        losssave.append(np.abs(np.min(np.mean(msout, axis=1))))
    losssave = np.array(losssave)
    baseratio = baseratios[np.argmin(losssave)]
    
    msout = np.zeros(mssignal.shape) * np.nan
    for ROI in range(mssignal.shape[1]):
        roisignal = mssignal[:,ROI]
        for b in msbins:
            mssave.append(np.std(roisignal[b:b+ws]))
        msstd = np.median(mssave)
        base = np.mean(np.sort(roisignal)[0:int(round(len(roisignal)*baseratio))])
        msout[:, ROI] = (roisignal - base)/msstd
    
    if verbose > 0:
        print('loss', np.min(losssave), 'at', baseratio)
        plt.plot(np.mean(msout, axis=1))
    return msout


SE = 247
dynamic_base(signalss_raw[SE][0], ws=20, bins=5, verbose=1)
dynamic_base(signalss_raw[SE][1], ws=20, bins=5, verbose=1)

plt.plot(np.mean(signalss[SE][0], axis=1))
plt.plot(np.mean(signalss[SE][2], axis=1))
plt.plot(np.mean(signalss[SE][4], axis=1))
plt.plot(np.mean(signalss[SE][5], axis=1))



            #%%
target = np.zeros((N,MAXSE))
for SE in range(N):
    if SE in [247, 248, 250, 251]:
        for se in range(len(signalss[SE])):
            meansignal = np.mean(signalss[SE][se], axis=1)
            msout = ms_smooth(mssignal=meansignal, ws=10)
            # plt.plot(meansignal[:-1] / np.mean(meansignal[:-1]))
            # plt.plot(msout[:-1] / np.mean(msout[:-1]))
            
            diff = np.abs(msout[1:] - msout[:-1])
            # plt.plot(diff / np.mean(diff))
            
            # plt.plot(active_R_mean[SE][se] / np.nanmean(active_R_mean[SE][se]))
            # target = np.zeros((N,MAXSE))
            
            vix = np.isnan(active_R_mean[SE][se][:-1])==0
            a = active_R_mean[SE][se][:-1][vix]
            b = diff[vix]
            
            target[SE,se] = stats.pearsonr(a, b)[0]
            
            plt.figure(); plt.plot(a/np.mean(a)); plt.plot(b/np.mean(b))
            plt.title(str(SE) + '_' + str(se))
            
            b = b/np.mean(b)
            
            vix = np.where(b > 1)[0]
            print(SE, se, np.mean(a[vix]))
#%%

SE, se = 8, 1

mssignal = signalss[SE][se]
meansignal = np.mean(mssignal, axis=1)
ROInum = mssignal.shape[1]
vix = np.where(movement_syn[SE][se] > 0)[0]
timeline_R = np.zeros(mssignal.shape[0]) * np.nan

print(); print(SE,se)
for t in range(mssignal.shape[0]):
    if len(mssignal[t-ws:t+ws]) == ws*2:
        corr_matrix = []
        for ROI in range(ROInum):
            msdata = mssignal[t-ws:t+ws,ROI]
            corr_matrix.append(msdata)
        
        R_matrix = np.zeros((ROInum,ROInum)) * np.nan
        for i1 in range(ROInum):
            for i2 in range(i1+1, ROInum):
                R = stats.pearsonr(corr_matrix[i1], corr_matrix[i2])[0]
                R_matrix[i1, i2] = R
        timeline_R[t] = np.nanmean(R_matrix)
active_R_mean[SE][se] = timeline_R
plt.figure()
plt.title(str(SE) + '_' + str(se))
plt.plot(timeline_R)

plt.plot(np.mean(signalss[241][0], axis=1))
plt.plot(np.mean(signalss[241][1], axis=1))
plt.plot(np.mean(signalss[241][2], axis=1))
#%%
for SE in [257]:
    for se in range(len(signalss[SE])):
        plt.figure()
        plt.title(str(SE) + '_' + str(se))
        plt.plot(active_R_mean[SE][se])
                           
       #%% prism 복붙용 변수생성

pain_time = msFunction.msarray([MAXSE])
nonpain_time = msFunction.msarray([MAXSE])

# target = np.array(mssave)
for row in range(len(target)):
    target[row,:] = target[row,:] - target[row,0]
    print(target[row,:4])

nonpain1, nonpain2, pain = [], [], []
for SE in range(N):
    if SE in highGroup3: # filter
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test:
                nonpain_time[se].append(target[SE,se])
            if [SE, se] in group_pain_test:
                pain_time[se].append(target[SE,se])

def to_prism(target):
    Aprism = pd.DataFrame([])
    for row in range(len(target)):
        Aprism = pd.concat((Aprism, pd.DataFrame(target[row])), ignore_index=True, axis=1)
    return Aprism

Aprism_nonpain = to_prism(nonpain_time)
Aprism_pain = to_prism(pain_time)

# ROC 판정용 - 직렬화
def to_linear(target):
    linear = []
    for row in range(len(target)):
        linear += target[row]
    return linear

nonpain = to_linear(nonpain_time)
pain = to_linear(pain_time)

print(np.nanmean(nonpain), np.nanmean(pain))
accuracy, roc_auc = msROC(nonpain, pain)
print(accuracy, roc_auc)                 
                        







#%%

pain = np.nanmean(mssave[:-3,:], axis=0)
nonpain = np.nanmean(mssave[-3:,:], axis=0)

plt.plot(pain); plt.plot(nonpain)

accuracy, roc_auc = msROC(mssave[-3:,1:].flatten(), mssave[:-3,1:].flatten())
print(accuracy, roc_auc)

for c in range(2, 10):
    accuracy, roc_auc = msROC(mssave[-3:,c:].flatten(), mssave[:-3,c:].flatten())
    print(c, accuracy, roc_auc)


#% prism 복붙용 변수생성

pain_time = msFunction.msarray([MAXSE])
nonpain_time = msFunction.msarray([MAXSE])

target = np.array(mssave)
for row in range(len(target)):
    target[row,:] = target[row,:]

nonpain1, nonpain2, pain = [], [], []
for SE in range(N):
    if SE in targetGroup: # filter
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test:
                nonpain_time[se].append(target[SE,se])
            if [SE, se] in group_pain_test:
                pain_time[se].append(target[SE,se])

def to_prism(target):
    Aprism = pd.DataFrame([])
    for row in range(len(target)):
        Aprism = pd.concat((Aprism, pd.DataFrame(target[row])), ignore_index=True, axis=1)
    return Aprism

Aprism_nonpain = to_prism(nonpain_time)
Aprism_pain = to_prism(pain_time)

# ROC 판정용 - 직렬화
def to_linear(target):
    linear = []
    for row in range(len(target)):
        linear += target[row]
    return linear

nonpain = to_linear(nonpain_time)
pain = to_linear(pain_time)

print(np.mean(nonpain), np.mean(pain))
accuracy, roc_auc = msROC(nonpain, pain)
print(accuracy, roc_auc)

#%%
SE , se, ROI = 240, 0, 0
signalss_v2 = msFunction.msarray([N,MAXSE])
for SE in PSLgroup_khu:
    for se in range(len(signalss_raw[SE])):
        mssignal = np.array(signalss_raw[SE][se])
        allo = np.zeros(mssignal.shape) * np.nan
        for ROI in range(mssignal.shape[1]):
            std = np.std(mssignal[:,ROI])
            mean = np.mean(mssignal[:,ROI])
            msout = ms_smooth(mssignal=mssignal[:,ROI], ws=40)
            base_element = np.sort(msout)[0:int(round(mssignal.shape[1]*0.10))]
            base = np.mean(base_element)
            
            base_minus = msout-base
                
            allo[:, ROI] = ((base_minus)/base) / (std/mean)
            
            # plt.plot(msout)
            # plt.plot(base_minus)
            # plt.figure(); plt.plot(allo[:, ROI]); plt.title(str(ROI))
            
        signalss_v2[SE][se] = allo

#%%


nonpain_set = [[[239, 0], [240, 0]], \
               [[241, 0], [242, 0]], \
               [[243, 0], [244, 0]], \
               [[245, 0], [246, 0]]]
    
pain_set = [[[239, 1], [240, 1], [239, 2], [240, 2]], \
            [[241, 1], [242, 1], [241, 2], [242, 2]], \
            [[243, 1], [244, 1], [243, 2], [244, 2]], \
            [[245, 1], [246, 1], [245, 2], [246, 2]]]

    
for SE in PSLgroup_khu:
    for se in range(3):
        plt.figure()
        plt.plot(np.mean(signalss_v2[SE][se], axis=1))
        plt.title(str(SE)+'_'+str(se))
    
    #%%

testslit = [247, 248, 250, 251, 257, 258, 259, 262]

formalin = [252, 253, 254, 256, 260, 261, 265, 266, 267, 269, 272]

saline = list(set(list(range(247, 273))) - set(formalin))

mssave = []
for SE in formalin:
    starndard = np.mean(signalss_raw[SE][0], axis=0)
    starndard = starndard / np.mean(starndard)
    
    for se in range(3):
        target = np.mean(signalss_raw[SE][se], axis=0)
        target = target / np.mean(target)
        
        r = np.mean(np.abs(starndard-target))
        print(SE, se, r)
        if se == 2: mssave.append(r)
print(np.mean(mssave))

mssave = []
for SE in saline:
    starndard = np.mean(signalss_raw[SE][0], axis=0)
    starndard = starndard / np.mean(starndard)
    
    for se in range(3):
        target = np.mean(signalss_raw[SE][se], axis=0)
        target = target / np.mean(target)
        
        r = np.mean(np.abs(starndard-target))
        print(SE, se, r)
        if se == 2: mssave.append(r)
print(np.mean(mssave))
        
    
    #%%
    ROInum = signalss_raw[SE][0].shape[1]
    THR = 0.0;
    painROI = np.zeros(ROInum) * np.nan
    for ROI in range(ROInum):
        nonpain, pain = [] ,[]
        for SE, se in nonpain_set[i]:
            nonpain.append(np.mean(signalss_v2[SE][se][:,ROI]))  
        for SE, se in pain_set[i]:
            pain.append(np.mean(signalss_v2[SE][se][:,ROI]))
            
        painROI[ROI] = np.min(pain) - np.min(nonpain) > THR
    print(i, np.mean(painROI))
    
    for se in range(3):
        plt.figure()
        plt.plot(np.mean(signalss_v2[SE][se][:,np.where(painROI==1)[0]], axis=1))
        plt.plot(np.mean(signalss_v2[SE][se][:,np.where(painROI==0)[0]], axis=1))
        plt.title(str(i) +'_' + str(se))

        
for SE in [239]:
    for ROI in [42]:
        a = signalss_v2[SE][0][:,ROI]
        b = signalss_v2[SE][1][:,ROI]
        c = signalss_v2[SE][2][:,ROI]
        plt.plot(np.concatenate((a,b,c), axis=0))
        
        plt.plot(c)
        
        
        
        
        
    nonpain, pain = [] ,[]
    for SE, se in [[239, 0], [240, 0]]:
        nonpain.append(np.mean(signalss_zscore[SE][se][:,ROI] > THR))  
    for SE, se in [[239, 1], [240, 1], [239, 2], [240, 2]]:
        pain.append(np.mean(signalss_zscore[SE][se][:,ROI] > THR))
        
plt.plot(np.mean(signalss_zscore[239][1], axis=1))
plt.plot(np.mean(signalss_zscore[239][2], axis=1))
plt.plot(np.mean(signalss_zscore[240][1], axis=1))
plt.plot(np.mean(signalss_zscore[240][2], axis=1))


plt.plot(np.mean(signalss_zscore[SE][se], axis=1))

ROI = 3

SE = 245
plt.plot(np.mean(signalss_v2[SE][0], axis=1))
plt.plot(np.std(signalss_v2[SE][0], axis=1))

plt.plot(np.mean(signalss_v2[SE][1], axis=1))
plt.plot(np.std(signalss_v2[SE][1], axis=1))

plt.plot(np.mean(signalss_v2[SE][2], axis=1))
plt.plot(np.std(signalss_v2[SE][2], axis=1))


thrkey = np.zeros((1000,10)) * np.nan

thrkey[242,0] = 170
thrkey[242,1] = 165
thrkey[242,2] = 170
thrkey[241,0] = 160
thrkey[241,1] = 170
thrkey[241,2] = 170
thrkey[243,0] = 150
thrkey[243,1] = 160
thrkey[243,2] = 180
thrkey[244,0] = 150
thrkey[244,1] = 170
thrkey[244,2] = 160
thrkey[245,0] = 200
thrkey[245,1] = 175
thrkey[245,2] = 140

for SE in [241, 242]:
    for se in range(3):
        a = np.std(signalss_raw[SE][se], axis=1)
        b = np.mean(signalss_raw[SE][se], axis=1)
        c = a/b
        plt.figure()
        # plt.plot(a)
        plt.plot(b)
        # plt.plot(c)
        plt.title(str(SE)+'_'+str(se))
        
        vix = np.where(b > thrkey[SE,se])[0]
        
        
        R = stats.pearsonr(a[vix], b[vix])[0]
        
        print(SE, se, R, np.mean(a[vix])/np.mean(b[vix]))



SE, se, ROI = 239, 0, 2
plt.plot(signalss_raw[SE][0][:,ROI])
plt.plot(signalss_raw[SE][1][:,ROI])
plt.plot(signalss_raw[SE][2][:,ROI])


def ms_smooth(mssignal=None, ws=None):
    msout = np.zeros(len(mssignal)) * np.nan
    for t in range(len(mssignal)):
        s = np.max([t-ws, 0])
        e = np.min([t+ws, len(mssignal)])
        msout[t] = np.mean(mssignal[s:e])
    return msout



msout = ms_smooth(mssignal=signalss_raw[240][1][:,ROI], ws=40)
plt.plot(msout)

#%% activity distribution

pslGroup
shamGroup

WS = int(round(FPS * 10))
BINS = shamGroup

mslist = pslGroup
mssave = msFunction.msarray([N,MAXSE])
mssave2 = np.zeros((N,MAXSE)) * np.nan

for SE in mslist:
    vix = np.where(movement_syn[SE][0] > 0)[0]
    starndard = np.mean(signalss_raw[SE][0][vix,:], axis=0)
    starndard = starndard / np.sum(starndard)
    
    for se in range(3):
        vix = np.where(movement_syn[SE][se] > 0)[0]
        target = np.mean(signalss_raw[SE][se][vix,:], axis=0)
        target = target / np.sum(target)
        r = np.mean(np.abs(starndard-target))

        mssave2[SE,se] = r
        print(SE, se, mssave2[SE,se])

print(np.nanmedian(mssave2[mslist,:3], axis=0))


#%% Parkinson's pain
PATH = 'D:\\mscore\\syncbackup\\google_syn\\'
pickle_save_tmp = PATH + 'mspickle_PD.pickle'    
with open(pickle_save_tmp, 'rb') as f:  # Python 3: open(..., 'rb')
    signalss_PD = pickle.load(f)    


MAXSE = 10
targetGroup = list(range(len(signalss_PD)))
signalss = list(signalss_PD)

ratio = 0.3
stanse = 1

mslist = list(range(0, 11))
mssave = np.zeros((N,MAXSE,2)) * np.nan
for stanse in [0, 1]:
    for SE in mslist:
        meansignal = np.mean(signalss_PD[SE][stanse], axis=1)
        vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
        starndard = np.mean(signalss_PD[SE][stanse][vix,:], axis=0)
        starndard = starndard / np.sum(starndard)
        
        for se in range(len(signalss_PD[SE])):
            if len(signalss_PD[SE][se]) > 0:
                meansignal = np.mean(signalss_PD[SE][se], axis=1)
                vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
                target = np.mean(signalss_PD[SE][se][vix,:], axis=0)
                target = target / np.sum(target)
                
                r = np.mean(np.abs(starndard-target))
                mssave[SE,se,stanse] = r
                
        if stanse == 0: nmr = 1
        if stanse == 1: nmr = 0
        mssave[SE,:,stanse] = mssave[SE,:,stanse]/mssave[SE,nmr,stanse]

exp = mslist = list(range(0, 8))
print('exp', np.nanmean(np.nanmean(mssave[mslist,:10,:], axis=2), axis=0))

con = mslist = list(range(8, 11))
print('con', np.nanmean(np.nanmean(mssave[mslist,:10,:], axis=2), axis=0))

#%% khupsl

ratio = 1
stanse = 1

mslist = np.sort(list(set(highGroup3) - set(list(range(230,239)))))
mssave = np.zeros((N,MAXSE,2)) * np.nan
for stanse in [0, 1]:
    for SE in mslist:
        meansignal = np.mean(signalss_raw[SE][stanse], axis=1)
        vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
        starndard = np.mean(signalss_raw[SE][stanse][vix,:], axis=0)
        starndard = starndard / np.sum(starndard)
        
        for se in range(len(signalss_raw[SE])):
            if len(signalss_raw[SE][se]) > 0:
                meansignal = np.mean(signalss_raw[SE][se], axis=1)
                vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
                target = np.mean(signalss_raw[SE][se][vix,:], axis=0)
                target = target / np.sum(target)
                
                r = np.mean(np.abs(starndard-target))
                mssave[SE,se,stanse] = r
                
        if stanse == 0: nmr = 1
        if stanse == 1: nmr = 0
        mssave[SE,:,stanse] = mssave[SE,:,stanse]/mssave[SE,nmr,stanse]

target = np.nanmean(mssave, axis=2)
A = target[mslist]

pain_time = msFunction.msarray([MAXSE])
nonpain_time = msFunction.msarray([MAXSE])

# target = np.array(mssave)
# for row in range(len(target)):
#     target[row,:] = target[row,:] - target[row,0]
#     print(target[row,:4])

nonpain1, nonpain2, pain = [], [], []
for SE in range(N):
    if SE in mslist: # filter
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test:
                nonpain_time[se].append(target[SE,se])
            if [SE, se] in group_pain_test:
                pain_time[se].append(target[SE,se])

def to_prism(target):
    Aprism = pd.DataFrame([])
    for row in range(len(target)):
        Aprism = pd.concat((Aprism, pd.DataFrame(target[row])), ignore_index=True, axis=1)
    return Aprism

Aprism_nonpain = to_prism(nonpain_time)
Aprism_pain = to_prism(pain_time)

# ROC 판정용 - 직렬화
def to_linear(target):
    linear = []
    for row in range(len(target)):
        linear += target[row]
    return linear

nonpain = to_linear(nonpain_time)
pain = to_linear(pain_time)

print(np.nanmean(nonpain), np.nanmean(pain))
accuracy, roc_auc = msROC(nonpain, pain)
print(accuracy, roc_auc)           


#%% SNU psl pain, self nmr

forlist = pslGroup + shamGroup + PSLgroup_khu
stanse = 0
ratio = 1
WS, bins = int(round(FPS*40)), 1 # int(round(FPS*2))
mssave = np.zeros((N,MAXSE)) * np.nan
# SE = forlist[0]
for SE in forlist:
    meansignal = np.mean(signalss_raw[SE][stanse], axis=1)
    vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
    starndard = np.mean(signalss_raw[SE][stanse][vix,:], axis=0)
    starndard = starndard / np.sum(starndard)
    
    msbins = list(range(0, len(signalss_raw[SE][stanse])-WS, bins))
    d1 = []
    for t in msbins:
        target = np.mean(signalss_raw[SE][stanse][t:t+WS,:], axis=0)
        target = target / np.sum(target)
        r = np.mean(np.abs(starndard-target))
        d1.append(r)
    nmr = np.mean(d1)

    for se in range(len(signalss_raw[SE])):
        if len(signalss_raw[SE][se]) > 0:
            meansignal = np.mean(signalss_raw[SE][se], axis=1)
            vix = np.argsort(meansignal)[::-1][:int(len(meansignal)*ratio)]
            target = np.mean(signalss_raw[SE][se][vix,:], axis=0)
            target = target / np.sum(target)
            
            r = np.mean(np.abs(starndard-target))
            mssave[SE,se] = r / (nmr * 1)


print(np.mean(mssave[pslGroup,:3], axis=0))

print(np.mean(mssave[shamGroup,:3], axis=0))


print(np.mean(mssave[PSLgroup_khu,:3], axis=0))


#%% featrue_extraction


inputsignal = signalss_raw[70]

# inputsignal = signalss_raw_PD[SE]
# inputsignal2 = signalss_PD[SE]
def feature_extraction(inputsignal = None, inputsignal2 = None):   
    WS = 40
    BINS = 10
    msout = []
    
    def mslinear_regression(x,y):
        x = np.array(x); y = np.array(y); 
        x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
        
        n = x.shape[0]
        r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
        m = r*(np.std(y)/np.std(x))
        b = np.mean(y) - np.mean(x)*m
        return m, b # bx+a
    
    def ms_smooth(mssignal=None, ws=None):
        msout = np.zeros(len(mssignal)) * np.nan
        for t in range(len(mssignal)):
            s = np.max([t-ws, 0])
            e = np.min([t+ws, len(mssignal)])
            msout[t] = np.mean(mssignal[s:e])
        return msout
    
    
    basese=[]
    for se in range(len(inputsignal)):
        masave=[]
        for ROI in range(inputsignal[se].shape[1]):
            t = mssignal=inputsignal[se][:,ROI]
            t2 = ms_smooth(t, ws=5)
            masave.append(t2)
        inputsignal2 = np.transpose(np.array(masave))
        
        meansignal = np.mean(inputsignal2, axis=1)
        vix = np.argsort(meansignal)[:int(len(meansignal)*0.30)]
        msmin = np.mean(meansignal[vix])
        basese.append(msmin)
            
    for se in range(len(inputsignal)):
        nmr = np.mean(basese)-basese[se]
        mssignal = inputsignal[se]
        
        vix = np.argsort(np.mean(mssignal, axis=0))[::-1][:int(len(mssignal)*1)]
        mssignal = np.array(mssignal)[:,vix] + nmr
        
        meansignal2 = np.mean(mssignal, axis=1)
        
        msbins = range(0, len(meansignal2)-WS, BINS)
        msplot = msFunction.msarray([2])
        for f in msbins:
            tmp = meansignal2[f:f+WS]
            if len(tmp) == WS:
                x1 = np.mean(tmp)
                x2 = np.std(np.mean(mssignal[f:f+WS,:], axis=0))
                msplot[0].append(x1)
                msplot[1].append(x2)
       
        vix = np.argsort(msplot[0])[::-1][:int(len(msplot[0])*1)]
        msplot[0] = np.array(msplot[0])[vix]
        msplot[1] = np.array(msplot[1])[vix]
        
        x = msplot[0]/np.max(msplot[0])
        y = msplot[1]/np.max(msplot[1])
        
        m, b = mslinear_regression(x, y)
        
        import scipy.stats
        
        r, p = scipy.stats.pearsonr(x, y)
        
        minx, miny = np.min(x), np.min(y)
        
        # featrue2
        sig = inputsignal2[0]
        stand = np.mean(sig, axis=0) / np.mean(sig)
        for se in range(len(inputsignal2)):
            sig = inputsignal2[se]
            exp = np.mean(sig, axis=0) / np.mean(sig)
            fn2 = np.mean(np.abs(exp - stand) > 0.22)

        msout.append([m, b, r, p, minx, miny, fn2])
        
        if False:
            plt.figure()
            plt.scatter(x, y)
            plt.title(str(SE)+'_'+str(se))
            plt.xlim([0,1])
            plt.ylim([0,1])
            
    return msout
        


if False:
    tlist = pslGroup + shamGroup + morphineGroup
    target = np.zeros((N,MAXSE)) * np.nan
    
    for SE in tlist:
        print(SE)
        msout = feature_extraction(signalss_raw[SE])
        for se in range(len(msout)):
            target[SE,se] = msout[se][0]
            
    for SE in range(len(target)):
        target[SE,:] = target[SE,:] / target[SE,0]
        target[SE,:] = 1- target[SE,:]

    
    ### 평가 - SNUPSL
    
    # print(target[tlist,:3])
    accuracy, roc_auc = msROC(target[shamGroup,1:3].flatten(), target[pslGroup,1:3].flatten())
    print(roc_auc)
    
    ### 평가 - KHUPSL_Morphine
    nonpain, pain = [], []
    nonpain += list(target[morphineGroup,1])
    pain += list(target[morphineGroup,2:10])
    accuracy, roc_auc = msROC(nonpain, pain)
    print('morphine', roc_auc)
    
    A = target[morphineGroup,:13]
    
    # min X, Y
    # pearsonR = p, r 
    # linear-regression = m, b


#%% signalss featrue extraction  -> save

fn_extraction = msFunction.msarray([N])
for SE in range(N):
    print(SE)
    if len(signalss_raw[SE][0]) > 0:
        msout = feature_extraction(signalss_raw[SE], signalss[SE])
        fn_extraction[SE] = msout

savename = 'C:\\mass_save\\fn_extraction.pickle'
with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(fn_extraction, f, pickle.HIGHEST_PROTOCOL)
    print(savename, '저장되었습니다.')


#%% PD : signalss featrue extraction  -> save

loadpath = 'C:\\mass_save\\PDpain\\mspickle_PD.pickle'  
with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
    msdict = pickle.load(f)
   
signalss_raw_PD = msdict['signalss_raw_PD']
signalss_PD = msdict['signalss_PD']
behav_raw_PD = msdict['behav_raw_PD']


# fn_extraction = msFunction.msarray([len(signalss_PD)])
# for SE in range(len(signalss_PD)):
#     print(SE)
#     if len(signalss_raw_PD[SE][0]) > 0:
#         msout = feature_extraction(signalss_raw_PD[SE], signalss_PD[SE])
#         fn_extraction[SE] = msout

# savename = 'C:\\mass_save\\fn_extraction_PD.pickle'
# with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(fn_extraction, f, pickle.HIGHEST_PROTOCOL)
#     print(savename, '저장되었습니다.')

#%%
# pslGroup
# shamGroup


# signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열
# signalss_raw

mssave = []

for stanse in [0, 1]:
    for THR in [0.22]: 
        target_sig = list(signalss_PD)
        forlist = list(range(len(target_sig)))
        
        # target_sig = list(signalss)
        # forlist = pslGroup + shamGroup
        
        matrix = np.zeros((len(target_sig),MAXSE)) * np.nan
        for SE in forlist:
            sig = target_sig[SE][stanse]
            stand = np.mean(sig, axis=0) / np.mean(sig)
            
            for se in range(len(target_sig[SE])):
                sig = target_sig[SE][se]
                exp = np.mean(sig, axis=0) / np.mean(sig)
                
                result = np.mean(np.abs(exp - stand) > THR)
                if not(result==0): matrix[SE,se] = result
                
        # accuracy, roc_auc = msROC(matrix[shamGroup,1:3].flatten(), matrix[pslGroup,1:3].flatten())
        # print(roc_auc)
    
        for r in range(len(matrix)):
            matrix[r,:] = matrix[r,:] #  - np.nanmean(matrix[r,0:2])
    
        nonpain = list(matrix[8:,:].flatten()) + list(matrix[0:8,:2].flatten())
        nonpain = list(matrix[8:,2:].flatten())
        pain = matrix[0:8,4:].flatten()
        accuracy, roc_auc = msROC(nonpain, pain)
        
        print(THR, roc_auc)
        mssave.append(matrix)
        
print('final', np.max(mssave))


matrix = np.nanmean(mssave, axis=0)
plt.plot(np.nanmean(matrix[:8,:], axis=0))
plt.plot(np.nanmean(matrix[8:,:], axis=0))

matrix2 = []
for c in list(range(0, 16, 2)):
    matrix2.append(np.max(matrix[:,c:c+2], axis=1))

matrix2 = np.transpose(np.array(matrix2))
nonpain = matrix[8:,2:].flatten()
pain = matrix[0:8,2:].flatten()
accuracy, roc_auc = msROC(nonpain, pain)
print(THR, roc_auc)
#%%

mssave = []

for stanse in [0]:
    for THR in [0.22]: 
        # target_sig = list(signalss_PD)
        # forlist = list(range(len(target_sig)))
        
        target_sig = list(signalss)
        forlist = PSLgroup_khu + morphineGroup
        
        matrix = np.zeros((len(target_sig),MAXSE)) * np.nan
        for SE in forlist:
            sig = target_sig[SE][stanse]
            stand = np.mean(sig, axis=0) / np.mean(sig)
            
            for se in range(len(target_sig[SE])):
                sig = target_sig[SE][se]
                exp = np.mean(sig, axis=0) / np.mean(sig)
                
                result = np.mean(np.abs(exp - stand) > THR)
                if not(result==0): matrix[SE,se] = result
                
        # accuracy, roc_auc = msROC(matrix[shamGroup,1:3].flatten(), matrix[pslGroup,1:3].flatten())
        # print(roc_auc)
    
        nonpain = list(matrix[morphineGroup,:2].flatten())
        pain = list(matrix[morphineGroup,2:10].flatten())
        accuracy, roc_auc = msROC(nonpain, pain)
        
        # print(THR, roc_auc)
        mssave.append(matrix)
        
matrix = mssave

pain1 = matrix[PSLgroup_khu,:3].flatten()
pain2 = matrix[morphineGroup,2:10].flatten()

np.nanmean(list(pain1) + list(pain2))


nonpain2 = matrix[morphineGroup,10:].flatten()
print(np.nanmean(matrix[PSLgroup_khu,1].flatten()))
print(np.nanmean(matrix[PSLgroup_khu,2].flatten()))


#%%

mssave = []

for stanse in [0]:
    for THR in [0.22]: 
        # target_sig = list(signalss_PD)
        # forlist = list(range(len(target_sig)))
        
        target_sig = list(signalss)
        forlist = [247, 248, 250, 251, 257, 258, 259, 262]
        
        matrix = np.zeros((len(target_sig),MAXSE)) * np.nan
        for SE in forlist:
            sig = target_sig[SE][stanse]
            stand = np.mean(sig, axis=0) / np.mean(sig)
            
            for se in range(len(target_sig[SE])):
                sig = target_sig[SE][se]
                exp = np.mean(sig, axis=0) / np.mean(sig)
                
                result = np.mean(np.abs(exp - stand) > THR)
                if not(result==0): matrix[SE,se] = result
                
        # accuracy, roc_auc = msROC(matrix[shamGroup,1:3].flatten(), matrix[pslGroup,1:3].flatten())
        # print(roc_auc)
    
        nonpain = list(matrix[morphineGroup,:2].flatten())
        pain = list(matrix[morphineGroup,2:10].flatten())
        accuracy, roc_auc = msROC(nonpain, pain)
        
        # print(THR, roc_auc)
        mssave.append(matrix)
        
matrix = mssave

pain1 = matrix[PSLgroup_khu,:3].flatten()
pain2 = matrix[morphineGroup,2:10].flatten()

np.nanmean(list(pain1) + list(pain2))


nonpain2 = matrix[morphineGroup,10:].flatten()
print(np.nanmean(matrix[PSLgroup_khu,1].flatten()))
print(np.nanmean(matrix[PSLgroup_khu,2].flatten()))


np.mean(matrix[forlist,:6], axis=0)


#%% data import, preprocessing

def ms_syn(target_signal=None, target_size=None):
    downratio = target_signal.shape[0] / target_size
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
    return allo

loadpath = 'C:\\mass_save\\PDpain\\mspickle_PD.pickle'  
with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
    msdict = pickle.load(f)
   
signalss_raw_PD = msdict['signalss_raw_PD']
signalss_PD = msdict['signalss_PD']
behav_raw_PD = msdict['behav_raw_PD']

MAXSE = 20
signalss2 = msFunction.msarray([len(signalss_raw_PD)])
behavss2 = msFunction.msarray([len(signalss_raw_PD)])

for SE in range(len(signalss_raw_PD)):
    for se in range(len(signalss_raw_PD[SE])):
        matrix = []
        for ROI in range(signalss_raw_PD[SE][se].shape[1]):
            s = signalss_raw_PD[SE][se][:,ROI]
            s2 = s/np.mean(s) - 1
            matrix.append(s2)
        matrix = np.transpose(np.array(matrix))  
        m =  ms_syn(target_signal=behav_raw_PD[SE][se], target_size=signalss_raw_PD[SE][se].shape[0])
   
        signalss2[SE].append(matrix)
        behavss2[SE].append(m)
            
#%% keras setup

lr = 1e-3 # learning rate
    
n_hidden = int(8*1) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8*1) # fully conneted laye node 갯수 # 8 # 원래 6 
    
l2_rate = 0.0
dropout_rate1 = 0.2 # dropout rate
dropout_rate2 = 0.1 # 
    

from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import BatchNormalization

from numpy.random import seed as nseed #
import tensorflow as tf
from keras.layers import Conv1D
from keras.layers import Flatten


def keras_setup(lr=0.01, batchnmr=False, seed=1, ROInum=None):
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras

    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    input1 = keras.layers.Input(shape=(ROInum-1))
    input2 = keras.layers.Input(shape=1)
    
    input_cocat = keras.layers.Concatenate(axis=1)([input1, input2])

    input10 = Dense(ROInum-1+1, kernel_initializer = 'normal', kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input_cocat)
    input10 = Dense(ROInum-1+1, kernel_initializer = 'normal', kernel_regularizer=regularizers.l2(l2_rate))(input10)
    input10 = Dense(ROInum-1+1, kernel_initializer = 'normal', kernel_regularizer=regularizers.l2(l2_rate))(input10)
    
    # if batchnmr: input10 = BatchNormalization()(input10)
    # input10 = Dropout(dropout_rate1)(input10) # dropout
    
     

    merge_4 = Dense(1, kernel_initializer = init)(input10) # fully conneted layers, relu

    model = keras.models.Model(inputs=[input1, input2], outputs=merge_4) # input output 선언
    model.compile(loss='mean_squared_error', optimizer='adam') # optimizer
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup(lr=lr, seed=0, ROInum=100)
print(model.summary())



#%% estimation model


target_sig = list(signalss2)
target_sig2 = list(behavss2)
forlist = list(range(len(target_sig)))

WS = 200
BINS = 20


#%%

RESULT_SAVE_PATH = 'D:\\mscore\\syncbackup\\Project\\박하늬선생님_PD_painimaging\\models\\'
SE, ROI = 0, 0
mssave = msFunction.msarray([16,10])
for SE in range(0, 16):
    mssignal = target_sig[SE][0]
    roinum = mssignal.shape[1]
    for ROI in range(roinum):
        final_weightsave = RESULT_SAVE_PATH + str(SE) + '_' + str(ROI) + '_final.h5'
        xtmp, xtmp2, ytmp, ztmp = [], [] ,[], []
        if not(os.path.isfile(final_weightsave)) or True:
            for trse in [0, 1]:
                mssignal = target_sig[SE][trse]
                msbins = np.arange(0, mssignal.shape[0]-WS, BINS)

                for t in msbins:
                    x = np.mean(mssignal[t:t+WS,:])
                    y = np.mean(mssignal[t:t+WS,ROI], axis=0)
                    # x = np.delete(x, ROI, axis=0)
                    x2 = np.mean(target_sig2[SE][trse][t:t+WS])
                    
                    xtmp.append(x)
                    xtmp2.append(x2)
                    ytmp.append(y)
                    ztmp.append([SE,se,t,ROI])
                    
            xtmp, xtmp2, ytmp, ztmp = np.array(xtmp), np.array(xtmp2), np.array(ytmp), np.array(ztmp)
            model = keras_setup(lr=lr, seed=0, ROInum=2)
            # print(model.summary())
            
            hist = model.fit([xtmp, xtmp2], ytmp, batch_size=2**11, epochs=2000, verbose=0)
            model.save_weights(final_weightsave)
        
    model = keras_setup(lr=lr, seed=0, ROInum=2)
    for tese in range(len(target_sig[SE])):
        test = target_sig[SE][tese]
        if len(test) > 0:
            ROIsave = []
            nmr = np.mean(test, axis=0)
            for ROI in range(roinum):
                xtmp_test, xtmp_test2, ytmp_test, ztmp_test = [], [], [] ,[]
                for t in msbins:
                    # x_test = np.mean(test[t:t+WS,:], axis=0)
                    # y_test = x_test[ROI]
                    
                    x_test = np.mean(test[t:t+WS,:])
                    y_test = np.mean(test[t:t+WS,ROI], axis=0)
                    
                    # x_test = np.delete(x_test, ROI, axis=0)
                    x_test2 = np.mean(target_sig2[SE][tese][t:t+WS])
                    
                    xtmp_test.append(x_test)
                    xtmp_test2.append(x_test2)
                    ytmp_test.append(y_test)
                    ztmp_test.append([SE,se,t,ROI])
                xtmp_test, xtmp_test2, ytmp_test, ztmp_test = np.array(xtmp_test), np.array(xtmp_test2), np.array(ytmp_test), np.array(ztmp_test)
                
                final_weightsave = RESULT_SAVE_PATH + str(SE) + '_' + str(ROI) + '_final.h5'
                model.load_weights(final_weightsave)
                predict = model.predict([xtmp_test, xtmp_test2])
                
                if False:
                    plt.plot(ytmp_test)
                    plt.plot(predict)
                
                loss = np.mean(((ytmp_test - predict)**2)**0.5)
                diff = (ytmp_test - predict[:,0]) > 0.1
                ROIsave.append(diff)
            ROIsave = np.array(ROIsave)
            
            f = np.mean(ROIsave)
            print(SE, tese, f)
            mssave[SE][tese] = f

#%%




#%%

t4 = np.zeros((16, 10))
for SE in range(16):
    for se in range(len(target_sig[SE])): 
        t4[SE,se] = np.mean(target_sig[SE][se])

plt.figure()
plt.plot(np.mean(t4[0:8,:], axis=0))
plt.plot(np.mean(t4[8:,:], axis=0))

movement = np.zeros((16, 10))
for SE in range(16):
    for se in range(len(target_sig2[SE])): 
        movement[SE,se] = np.mean(target_sig2[SE][se])

plt.figure()
plt.plot(np.mean(movement[0:8,:], axis=0))
plt.plot(np.mean(movement[8:,:], axis=0))


t1 = np.zeros((16, 10))
for SE in range(16):
    for se in range(len(target_sig[SE])): 
        t1[SE,se] = np.mean(signalss_raw_PD[SE][se])/ np.mean(target_sig2[SE][se]> 0.15)

plt.figure()
plt.plot(np.median(t1[0:8,:], axis=0))
plt.plot(np.median(t1[8:,:], axis=0))

















