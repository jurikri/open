# -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:24:00 2021

@author: MSBak
"""

import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode\\')
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
import time

MAXSE = 20

#%% data import

gsync = 'D:\\2p_pain\\'
# gsync = 'C:\\mass_save\\PSLpain\\'
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
KHU_CFA = msGroup['KHU_CFA']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

#%%
for SE in range(N):
    for se in range(len(signalss_raw[SE])):
        tmp = np.array(signalss_raw[SE][se])
        signalss[SE][se] = tmp / np.mean(tmp)
            
movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = behavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = msFunction.downsampling(behav_tmp, signalss[SE][se].shape[0])[0,:]
            if np.isnan(np.mean(movement_syn[SE][se])): movement_syn[SE][se] = []
            
t4, mov = np.zeros((N, MAXSE)), np.zeros((N, MAXSE))
for SE in range(N):
    for se in range(len(signalss[SE])):
        mov[SE,se] = np.mean(movement_syn[SE][se])
#%%
savepath = 'D:\\2p_pain\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    
mssave = np.mean(np.array(repeat_save), axis=0)
    

#%% snu psl

# regrouping
GBVX_nonpain_d3, GBVX_nonpain_d10 = [], []
GBVX = [164, 166, 167, 172, 174, 177, 179, 181]
for SE in [164, 166, 167, 172, 174, 177, 179, 181]:
    d3c, d10c = [], []
    for se in range(12):
        d10c.append(SE in [164, 166] and se in [2,3])
        
        d3c.append(SE in [167] and se in [4,5])
        d10c.append(SE in [167] and se in [6,7])
        
        d3c.append(SE in [172] and se in [4,5])
        d10c.append(SE in [172] and se in [8,9])
        
        d3c.append(SE in [174] and se in [4,5])
        
        d3c.append(SE in [177,179,181] and se in [2,3])
        d10c.append(SE in [177,179] and se in [6,7])

        if np.sum(np.array(d3c)) > 0: GBVX_nonpain_d3.append(mssave[SE,se])
        if np.sum(np.array(d10c)) > 0: GBVX_nonpain_d10.append(mssave[SE,se])
        
GBVX_nonpain_d3 = msFunction.nanex(GBVX_nonpain_d3)
GBVX_nonpain_d10 = msFunction.nanex(GBVX_nonpain_d10)

# plotting

msplot = mssave[shamGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

msplot = mssave[pslGroup,1:3]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[ipsaline_pslGroup + ipclonidineGroup,:][:,[1,3]]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='g')

msplot = np.zeros((50, 2)) * np.nan
msplot[:len(GBVX_nonpain_d3),0] = GBVX_nonpain_d3
msplot[:len(GBVX_nonpain_d10),1] = GBVX_nonpain_d10
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='k')

# saving, ROC

psl_d3 = list(mssave[pslGroup,1]) + list(mssave[ipsaline_pslGroup + ipclonidineGroup,1])
psl_d10 = list(mssave[pslGroup,2]) + list(mssave[ipsaline_pslGroup + ipclonidineGroup,3])
psl_d3_GBVX = list(GBVX_nonpain_d3)
psl_d10_GBVX = list(GBVX_nonpain_d10)
sham_d3 = list(mssave[shamGroup,1])
sham_d10 = list(mssave[shamGroup,2])


Aprism2 = pd.DataFrame(sham_d3)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3_GBVX)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(sham_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10_GBVX)), ignore_index=True, axis=1)

Aprism = msFunction.msarray([6])
Aprism[0] += list(msFunction.nanex(sham_d3))
Aprism[1] += list(msFunction.nanex(sham_d10))
Aprism[2] += list(msFunction.nanex(psl_d3))
Aprism[3] += list(msFunction.nanex(psl_d10))
Aprism[4] += list(msFunction.nanex(psl_d3_GBVX))
Aprism[5] += list(msFunction.nanex(psl_d10_GBVX))

Aprism_info = np.zeros((len(Aprism),3))
for i in range(len(Aprism)):
    Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

nonpain = list(sham_d3) + list(sham_d10)
pain = list(psl_d3) + list(psl_d10)
accuracy, roc_auc, fig_data_psl = msFunction.msROC(nonpain, pain)
    
#%% snu oxali (d3-pain, d7-pain, d10-nonpain)

 # painc.append(SE in oxaliGroup and se in [1])
 # painc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [2])
 nonpainc.append(SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [3])
 nonpainc.append(SE in [188, 189, 200, 201] and se in [2])
 nonpainc.append(SE in glucoseGroup and se in [0,1,2,3,4])

oxali = msFunction.msarray([4])
for SE in oxaliGroup:
    for se in range(MAXSE):
        if SE in oxaliGroup and se in [0]:
            oxali[0].append(mssave[SE,se])
        if SE in oxaliGroup and se in [1]:
            oxali[1].append(mssave[SE,se])
        if SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [2]: 
            oxali[2].append(mssave[SE,se])
        if SE in list(range(192,200)) + [202, 203, 220, 221]  and se in [3]:
            oxali[3].append(mssave[SE,se])
        if SE in [188, 189, 200, 201] and se in [2]:
            oxali[3].append(mssave[SE,se])

msplot = np.zeros((50, len(oxali))) * np.nan
for i in range(len(oxali)):
    tmp = msFunction.nanex(oxali[i])
    msplot[:len(tmp),i] = tmp

msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[glucoseGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

oxali_d3 = oxali[1]
oxali_d7 = oxali[2]
oxali_d10 = oxali[3]
glucose = msFunction.nanex(mssave[glucoseGroup,:].flatten())

# Aprism2 = pd.DataFrame(sham_d3)
# Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3)), ignore_index=True, axis=1)
# Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d3_GBVX)), ignore_index=True, axis=1)
# Aprism2 = pd.concat((Aprism2, pd.DataFrame(sham_d10)), ignore_index=True, axis=1)
# Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10)), ignore_index=True, axis=1)
# Aprism2 = pd.concat((Aprism2, pd.DataFrame(psl_d10_GBVX)), ignore_index=True, axis=1)

# Aprism = msFunction.msarray([6])
# Aprism[0] += list(msFunction.nanex(sham_d3))
# Aprism[1] += list(msFunction.nanex(sham_d10))
# Aprism[2] += list(msFunction.nanex(psl_d3))
# Aprism[3] += list(msFunction.nanex(psl_d10))
# Aprism[4] += list(msFunction.nanex(psl_d3_GBVX))
# Aprism[5] += list(msFunction.nanex(psl_d10_GBVX))

# Aprism_info = np.zeros((len(Aprism),3))
# for i in range(len(Aprism)):
#     Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

accuracy, roc_auc, fig_data_oxali = msFunction.msROC(glucose, oxali_d3 + oxali_d7)

plt.plot(fig_data_psl[0], fig_data_psl[1], label='PSL (AUC = %0.2f)' % fig_data_psl[2])
plt.plot(fig_data_oxali[0], fig_data_oxali[1], label='oxaliplatin (AUC = %0.2f)' % fig_data_oxali[2])
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend(loc="lower right")
figsavepath = 'D:\\2p_pain\\weight_saves\\211122\\AUC2.png'
plt.savefig(figsavepath, dpi=1000)

#%% snu formaline

msplot = mssave[highGroup + highGroup2,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[midleGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[ketoGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[yohimbineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[lidocaineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[salineGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

# saving, ROC

high = list(mssave[highGroup + highGroup2,1])
midle = list(mssave[midleGroup,1])
keto = list(mssave[ketoGroup,1])
lido = list(mssave[lidocaineGroup,1])
saline = list(msFunction.nanex(mssave[salineGroup,:].flatten()))

Aprism2 = pd.DataFrame(saline)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(high)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(midle)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(keto)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(lido)), ignore_index=True, axis=1)

Aprism = msFunction.msarray([6])
Aprism[0] += list(msFunction.nanex(sham_d3))
Aprism[1] += list(msFunction.nanex(sham_d10))
Aprism[2] += list(msFunction.nanex(psl_d3))
Aprism[3] += list(msFunction.nanex(psl_d10))
Aprism[4] += list(msFunction.nanex(psl_d3_GBVX))
Aprism[5] += list(msFunction.nanex(psl_d10_GBVX))

Aprism_info = np.zeros((len(Aprism),3))
for i in range(len(Aprism)):
    Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

accuracy, roc_auc, fig_data_formalin = msFunction.msROC(saline, high)

#%% snu cfa, cap

msplot = mssave[CFAgroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

msplot = mssave[capsaicinGroup,:]
msplot_mean = np.nanmean(msplot, axis=0)
e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

# saving, ROC

cfa2 = list(mssave[CFAgroup,1])
cfa3 = list(mssave[CFAgroup,2])
cap = list(mssave[capsaicinGroup,1])

Aprism2 = pd.DataFrame(cfa2)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(cfa3)), ignore_index=True, axis=1)
Aprism2 = pd.concat((Aprism2, pd.DataFrame(cap)), ignore_index=True, axis=1)


Aprism = msFunction.msarray([6])
Aprism[0] += list(msFunction.nanex(sham_d3))
Aprism[1] += list(msFunction.nanex(sham_d10))
Aprism[2] += list(msFunction.nanex(psl_d3))
Aprism[3] += list(msFunction.nanex(psl_d10))
Aprism[4] += list(msFunction.nanex(psl_d3_GBVX))
Aprism[5] += list(msFunction.nanex(psl_d10_GBVX))

Aprism_info = np.zeros((len(Aprism),3))
for i in range(len(Aprism)):
    Aprism_info[i, :] = np.nanmean(Aprism[i]), scipy.stats.sem(Aprism[i], nan_policy='omit'), len(Aprism[i])

accuracy, roc_auc, fig_data_cfa = msFunction.msROC(saline, cfa2 + cfa3)
accuracy, roc_auc, fig_data_cap = msFunction.msROC(saline, cap)

plt.plot(fig_data_formalin[0], fig_data_formalin[1], label='formalin (AUC = %0.2f)' % fig_data_formalin[2])
plt.plot(fig_data_cap[0], fig_data_cap[1], label='capsaicin (AUC = %0.2f)' % fig_data_cap[2])
plt.plot(fig_data_cfa[0], fig_data_cfa[1], label='CFA (AUC = %0.2f)' % fig_data_cfa[2])
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.legend(loc="lower right")
figsavepath = 'D:\\2p_pain\\weight_saves\\211122\\AUC.png'
plt.savefig(figsavepath, dpi=1000)



#%%

























































