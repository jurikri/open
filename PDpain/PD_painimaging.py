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
#%% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
behavss = msdata_load['behavss']   # 움직임 정보
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

# for SE in range(N):
#     for se in range(len(signalss_raw[SE])):
#         tmp = np.array(signalss_raw[SE][se])
#         signalss[SE][se] = tmp / np.mean(tmp)

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = behavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = msFunction.downsampling(behav_tmp, signalss[SE][se].shape[0])[0,:]
            if np.isnan(np.mean(movement_syn[SE][se])): movement_syn[SE][se] = []
#%%
signalss2 = msFunction.msarray([N,MAXSE])

THR = 0.2
mssave = msFunction.msarray([N,MAXSE])
mssave_up = msFunction.msarray([N,MAXSE])
mssave_down = msFunction.msarray([N,MAXSE])

for SE in PDpain + PDnonpain:
    roiNum = signalss[SE][0].shape[1]
    seNum = len(signalss_raw[SE])
    
    # ROI ex
    exnum = int(round(roiNum * 0.02)) # %
    
    tmp = []
    for se in range(seNum):
        tmp.append(np.mean(signalss[SE][se], axis=0))
    msrank = np.mean(np.array(tmp), axis=0)
    maxix = np.argsort(msrank)[::-1][:exnum]
    minix = np.argsort(msrank)[::][:exnum]
    for se in range(seNum):
        signalss2[SE][se] = np.delete(signalss[SE][se], list(maxix)+list(minix), axis=1)
        signalss2[SE][se] = signalss[SE][se]
    
    selist = [0, 1] 
    for stanse in selist:
        # bthr = behavss[SE][stanse][1]
        # if np.isnan(np.mean(movement_syn[SE][stanse])): continue
        # vix2 = np.where(movement_syn[SE][stanse] <= bthr)[0]
        
        sig = signalss2[SE][stanse] # [vix2,:]
        sig2 = np.mean(sig, axis=0)
        roiNum = signalss2[SE][stanse].shape[1]
        stand2 = sig2  / np.sum(sig2) * roiNum
        
        for se in range(2, seNum):
            # bthr = behavss[SE][se][1]
            # if np.isnan(np.mean(movement_syn[SE][se])): continue
            # vix2 = np.where(movement_syn[SE][se] <= bthr)[0]
            
            sig = signalss2[SE][se] # [vix2,:]
            sig2 = np.mean(sig, axis=0)
            roiNum = signalss2[SE][se].shape[1]
            exp = sig2  / np.sum(sig2) * roiNum
            
            f1 = np.mean(np.abs(exp - stand2) > THR)
            mssave[SE][se].append(f1)
            
            msup = (exp - stand2) > THR
            mssave_up[SE][se].append(msup)
            
            msdown = (stand2 - exp) > THR
            mssave_down[SE][se].append(msdown)
            
            
mssave2 = np.zeros((N,MAXSE)) * np.nan
mssave2_up = np.zeros((N,MAXSE)) * np.nan
mssave2_down = np.zeros((N,MAXSE)) * np.nan
mov = np.zeros((N,MAXSE)) * np.nan
t4 = np.zeros((N,MAXSE)) * np.nan
for row in range(N):
    for col in range(MAXSE):
        mssave2[row, col] = np.nanmean(mssave[row][col])
        mssave2_up[row, col] = np.nanmean(mssave_up[row][col])
        mssave2_down[row, col] = np.nanmean(mssave_down[row][col])
        t4[row, col] = np.nanmean(signalss2[row][col])
        if not(np.isnan(np.mean(movement_syn[row][col]))):
            bthr = behavss[SE][stanse][1]
            mov[row, col] = np.mean(movement_syn[row][col] > bthr)
            
        
#%%

plt.figure()
msplot = mssave2[PDnonpain,0:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = mssave2[PDpain,0:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

#%%
msmatrix = []
for i in list(range(0,10,2)):
    msmatrix.append(np.nanmax(mssave2[:,i:i+1], axis=1))
msmatrix = np.transpose(np.array(msmatrix))

msplot = msmatrix[PDnonpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = msmatrix[PDpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

Aprism_pdpain =  msmatrix[PDpain,:5]
Aprism_pdnonpain =  msmatrix[PDnonpain,:5]


#%%

msmatrix2 = np.zeros(msmatrix.shape)
for i in range(len(msmatrix)):
    msmatrix2[i,:] = msmatrix[i,:] / msmatrix[i,1]

msplot = msmatrix2[PDnonpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = msmatrix2[PDpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')





#%%
#%%
target = np.array(t4)

plt.figure()
msplot = target[PDnonpain,0:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = target[PDpain,0:10]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msmatrix = []
for i in list(range(0,10,2)):
    msmatrix.append(np.nanmax(target[:,i:i+1], axis=1))
msmatrix = np.transpose(np.array(msmatrix))

plt.figure()
msplot = msmatrix[PDnonpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

msplot = msmatrix[PDpain,:5]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o')

Aprism_pdpain =  msmatrix[PDpain,:5]
Aprism_pdnonpain =  msmatrix[PDnonpain,:5]


sham = msmatrix[PDnonpain,:5]
PD = msmatrix[PDpain,:5]

Aprsim = np.concatenate((np.transpose(sham), np.transpose(PD)), axis=1)
























