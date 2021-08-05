# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:16:40 2021

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
from tqdm import tqdm
from scipy import stats
import scipy

# for se in range(13):
#     print(signalss[181][se].shape)

MAXSE = 20
#%% mFunction

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

def gaus(mu=None, sigma=None):
        x1 = mu - sigma * 7
        x2 = mu + sigma * 7
        x = np.linspace(x1, x2, 1000)
        y = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / (2 * sigma**2))
        return x, y 

#%% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
behavss = msdata_load['behavss2']   # 움직임 정보
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
KHUsham = msGroup['KHUsham']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

# signals_raw에서 직접 수정할경우
# signalss = msFunction.msarray([N])
# for SE in PSLgroup_khu + morphineGroup:
#     for se in range(len(signalss_raw[SE])):
#         allo = np.zeros(signalss_raw[SE][se].shape) * np.nan
#         for ROI in range(signalss_raw[SE][se].shape[1]):
#             matrix = signalss_raw[SE][se][:,ROI]
#             if len(bahavss[SE][se][0]) > 0:
#                 bratio = (1-np.mean(bahavss[SE][se][0] > 0.15)) * 0.3
#             else: bratio = 0.3
#             base = np.sort(matrix)[0:int(round(matrix.shape[0]*bratio))]
#             base_mean = np.mean(base)
#             matrix2 = matrix/base_mean
#             allo[:,ROI] = matrix2
#             # plt.plot(matrix2)
#         signalss[SE].append(allo)

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = behavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = downsampling(behav_tmp, signalss[SE][se].shape[0])

#%%
# FPS = 5.13 # PD
WS = 1040
BINS = 200

tlist = list(range(2, 6))
vlist = [[2,3], [4,5], [6,7], [8,9]]

mssave = np.zeros((N, 10)) * np.nan
mssave_tr = np.zeros((N, 10)) * np.nan

SE = 286
#%%

THR = 0.24
mssave = np.zeros((N,MAXSE)) * np.nan
for SE in PDpain + PDnonpain:
    roiNum = signalss[SE][0].shape[1]
    seNum = len(signalss_raw[SE])
    
    stanse = 0
    sig = signalss[SE][stanse] 
    stand = np.mean(sig, axis=0) / np.mean(sig)
    
    for se in range(seNum):
        if se != stanse:
            sig = signalss[SE][se]
            exp = np.mean(sig, axis=0) / np.mean(sig)
            f1 = np.mean(np.abs(exp - stand) > THR)
            mssave[SE,se] = f1

plt.figure()

msplot = mssave[PDpain,:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = mssave[PDnonpain,:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')


#%%
            # estimation
            
            for se in range(seNum):
                mssignal = np.array(signalss[SE][se])
                ac = np.mean(signalss[SE][se][bn:bn+WS, :], axis=0)
                ac = ac / np.sum(ac)
                
                pain_matrix  base_matrix
                
                  
                # for ROI in range(roiNum):
                #    mu = np.mean(base_matrix[:,ROI])
                #    sigma = np.std(base_matrix[:,ROI])
                #    x, y = gaus(mu = mu, sigma = sigma)
                   
                #    mu2 = np.mean(pain_matrix[:,ROI])
                #    sigma2 = np.std(pain_matrix[:,ROI])
                #    x2, y2 = gaus(mu = mu2, sigma = sigma2)
                   
                #    if False:
                #        plt.plot(x,y)
                #        plt.plot(x2,y2)
                #        plt.figure()
                #        xplot = list(base_matrix[:,ROI])
                #        plt.scatter(len(xplot)*[0], xplot)
                #        xplot = list(pain_matrix[:,ROI])
                #        plt.scatter(len(xplot)*[1], xplot)
                   
                #    p0 = stats.norm.pdf(ac[ROI], mu, sigma)
                #    p1 = stats.norm.pdf(ac[ROI], mu2, sigma2)
                   
                #    if p0 < p1:
                #        if se in telist: pmatrix[cv, se, ROI] = ac[ROI] * roiNum
                #        if se in basese + painse: pmatrix_tr[cv, se, ROI] = ac[ROI] * roiNum
    
    msin = np.nanmean(np.nanmean(pmatrix, axis=0), axis=1)
    mssave[SE,:len(msin)] = msin
    
    msin = np.nanmean(np.nanmean(pmatrix_tr, axis=0), axis=1)
    mssave_tr[SE,:len(msin)] = msin
    
plt.figure()
plt.plot(np.nanmean(mssave[PDpain,:], axis=0))
plt.plot(np.nanmean(mssave[PDnonpain,:], axis=0))


plt.figure()
plt.plot(np.nanmean(mssave_tr[PDpain,:], axis=0))
plt.plot(np.nanmean(mssave_tr[PDnonpain,:], axis=0))











































