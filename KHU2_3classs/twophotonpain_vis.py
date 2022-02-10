 # -*- coding: utf-8 -*-
"""
Created on Thu May 27 13:24:00 2021

@author: MSBak
"""

import sys; 
sys.path.append('D:\\mscore\\code_lab\\')
sys.path.append('C:\\mscode\\')
sys.path.append('C:\\Users\\skklab\\Documents\mscode\\')
sys.path.append('K:\\mscode_m2\\')

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

MAXSE = 40

gsync = 'C:\\SynologyDrive\\2p_data\\'
if os.path.isdir('K:\\mscode_m2'): gsync = 'K:\\SynologyDrive\\2p_data\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
behavss = msdata_load['behavss']   # 움직임 정보
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = np.array(msdata_load['signalss']) # 투포톤 이미징데이터 -> 시계열
# signalss_df = np.array(msdata_load['signalss']) # 투포톤 이미징데이터 -> 시계열
signalss_raw = np.array(msdata_load['signalss_raw'])

signalss2 = msdata_load['signalss2']
movement_syn = msdata_load['movement_syn']

inter_corr = msdata_load['inter_corr']

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
KHU_saline = msGroup['KHU_saline']
PSLgroup_khu =  msGroup['PSLgroup_khu']
morphineGroup = msGroup['morphineGroup']
KHUsham = msGroup['KHUsham']
KHU_CFA = msGroup['KHU_CFA']

PDpain = msGroup['PDpain']
PDnonpain = msGroup['PDnonpain']

PDmorphine = msGroup['PDmorphine']
KHU_PSL_magnolin = msGroup['KHU_PSL_magnolin']

# pdmorphine = list(range(325, 332))

#%% 검증

# array2d = mssave3

def msbarplot(array2d): # 2d arrary (sample x group)
    x = np.arange(array2d.shape[1])
    fig, ax = plt.subplots()
    y = np.nanmean(array2d, axis=0)
    e = scipy.stats.sem(array2d, axis=0, nan_policy='omit')
    plt.bar(x, y, yerr=e, align='center', alpha=0.5, ecolor='black', capsize=10)
    plt.xticks(x)
    
    for xi in range(len(x)):
        scatter_y = msFunction.nanex(array2d[:,xi])
        scatter_x = np.ones(len(scatter_y)) * xi
        plt.scatter(scatter_x, scatter_y, alpha=0.5)
    
# mssave_p = mssave[:,:,1]
def ms_report(mssave):
    mssave_p = np.array(mssave)
    plt.figure()
    msplot = mssave_p[morphineGroup,2:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    msplot = mssave_p[KHUsham,2:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    msplot = mssave_p[PSLgroup_khu,2:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='orange')
    
    #SS
    same_days = [[2,3,4,5],[6,7,8,9],[10,11,12]]
    target = morphineGroup
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout = pd.DataFrame(mssave3[target])
    
    same_days = [[2,3,4,5],[6,7,8,9],[10,11,12]]
    target = KHUsham
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout = pd.concat((prismout, pd.DataFrame(mssave3[target])), axis=1, ignore_index=True)
    
    same_days = [[2,3],[4,5]]
    target = PSLgroup_khu
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout = pd.concat((prismout, pd.DataFrame(mssave3[target])), axis=0, ignore_index=True)
    
    #SS
    same_days = [[2,3],[4,5],[6,7],[8,9],[10,11,12]]
    target = morphineGroup
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout_saline = pd.DataFrame(mssave3[target])
    
    same_days = [[2,3],[4,5],[6,7],[8,9],[10,11,12]]
    target = KHUsham
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout_saline = pd.concat((prismout_saline, pd.DataFrame(mssave3[target])), axis=1, ignore_index=True)
    
    same_days = [[2,3],[],[4,5]]
    target = PSLgroup_khu
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave_p[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    prismout_saline = pd.concat((prismout_saline, pd.DataFrame(mssave3[target])), axis=0, ignore_index=True)
    
    return prismout, prismout_saline
    
def ms_report_2d(mssave, target):
    def ms_2dscatter(x,y,c):
        plt.scatter(np.nanmean(x), np.nanmean(y), c=c)
        plt.errorbar(np.nanmean(x), np.nanmean(y), yerr=scipy.stats.sem(x, nan_policy='omit'), fmt="o", c=c)
        plt.errorbar(np.nanmean(x), np.nanmean(y), xerr=scipy.stats.sem(y, nan_policy='omit'), fmt="o", c=c)

    
    plt.figure()
    x_pain, x_nonpain, x_drug = [], [], []
    y_pain, y_nonpain, y_drug = [], [], []
    for SE in target:
        for se in range(MAXSE):
            if [SE, se] in group_pain_training: 
                y_pain.append(mssave[SE,se,:][0])
                x_pain.append(mssave[SE,se,:][1])
                
            if [SE, se] in group_nonpain_training: 
                y_nonpain.append(mssave[SE,se,:][0])
                x_nonpain.append(mssave[SE,se,:][1])
                
            if [SE, se] in group_drug_training: 
                y_drug.append(mssave[SE,se,:][0])
                x_drug.append(mssave[SE,se,:][1])
                
    x = list(x_pain)
    y = list(y_pain)
    c = 'r'
    ms_2dscatter(x,y,c)
    
    x = list(x_nonpain)
    y = list(y_nonpain)
    c = 'b'
    ms_2dscatter(x,y,c)
    
    x = list(x_drug)
    y = list(y_drug)
    c = 'g'
    ms_2dscatter(x,y,c)

def ms_report_cfa(mssave):
    KHU_CFA_100 = KHU_CFA[:7]
    KHU_CFA_50 = KHU_CFA[7:]

    target_group = list(KHU_CFA_50)

    plt.figure()
    msplot = mssave[target_group,0:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    target_group = list(KHU_CFA_100)

    plt.figure()
    msplot = mssave[target_group,0:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    # savefig
    same_days = [[0,1,2,3], [4,5], [6,7], [8,9],[10,11],[12,13]]
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[KHU_CFA_50,i] = np.nanmean(mssave[KHU_CFA_50,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    
    Aprism_50 = mssave3[KHU_CFA_50,:]
    
    same_days = [[0,1,2,3], [4,5], [6,7], [8,9,10]]
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[KHU_CFA_100,i] = np.nanmean(mssave[KHU_CFA_100,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)
    
    Aprism_100 = mssave3[KHU_CFA_100,:]
    
    return Aprism_100, Aprism_50
    
def ms_report_snu_chronic(mssave):
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


    GBVX_nonpain_d3, GBVX_nonpain_d10 = [], []
    for SE in [164, 166, 167, 172, 174, 177, 179, 181]:
        d3c, d10c = [], []
        for se in range(12):
            d10c.append(SE in [164, 165, 166] and se in [2,3])
            
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

    msplot = np.zeros((50, 2)) * np.nan
    msplot[:len(GBVX_nonpain_d3),0] = GBVX_nonpain_d3
    msplot[:len(GBVX_nonpain_d10),1] = GBVX_nonpain_d10
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='k')

    #%SNU oxaliplatin
    plt.figure()
    msplot = mssave[oxaliGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')

    msplot = mssave[glucoseGroup,1:3]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
def ms_report_khu_magnolin(mssave):    
    plt.figure()
    msplot = mssave[KHU_PSL_magnolin,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    #
    same_days = [[0,1,2,3],[4,5,6,7],[8,9],[10,11],[12,13]]
    target = KHU_PSL_magnolin
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[target,i] = np.nanmean(mssave[target,:][:, same_days[i]], axis=1)
    msbarplot(mssave3)

    return mssave3[target,:]

# mssave2 = mssave[:,:,1]
def msplot_PD(mssave2):
    PDmorphineA = [325, 326]
    PDmorphineB = [327, 328]
    PDmorphineC = [329, 330]
    PDmorphineD = [331]
    
    visse = 9
    PDmorphine_matrix = msFunction.msarray([len(PDmorphine), visse])
    
    for ix, SE in enumerate(PDmorphine):
        for se in range(MAXSE):
            if SE in PDmorphine and se in [0,1,2,3]:
                PDmorphine_matrix[ix][0].append(mssave2[SE,se])
            
            if SE in PDmorphineA and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [6,7]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [8,9]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineA and se in [12,13,14,15]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
                
            if SE in PDmorphineB and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [6,7,8,9]:
                PDmorphine_matrix[ix][2].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [10,11]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [12,13]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [14,15]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [18,19]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineB and se in [20,21]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
                
            if SE in PDmorphineC and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [6,7,8,9]:
                PDmorphine_matrix[ix][2].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [12,13]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [14,15]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [18,19]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
            if SE in PDmorphineC and se in [20,21]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
                
            if SE in PDmorphineD and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [6,7]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [8,9]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [12,13,14,15]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [16,17]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [18,19]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in PDmorphineD and se in [20,21]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
                
            if SE in [339] and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in [339] and se in [6,7]:
                PDmorphine_matrix[ix][3].append(mssave2[SE,se])
            if SE in [339] and se in [8,9]:
                PDmorphine_matrix[ix][4].append(mssave2[SE,se])
            if SE in [339] and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in [339] and se in [12,13,14,15]:
                PDmorphine_matrix[ix][6].append(mssave2[SE,se])
                
            if SE in [340, 341] and se in [4,5]:
                PDmorphine_matrix[ix][1].append(mssave2[SE,se])
            if SE in [340, 341] and se in [6,7,8,9]:
                PDmorphine_matrix[ix][2].append(mssave2[SE,se])
            if SE in [340, 341] and se in [10,11]:
                PDmorphine_matrix[ix][5].append(mssave2[SE,se])
            if SE in [340, 341] and se in [12,13]:
                PDmorphine_matrix[ix][7].append(mssave2[SE,se])
            if SE in [340, 341] and se in [14,15]:
                PDmorphine_matrix[ix][8].append(mssave2[SE,se])
                

    PDmorphine_matrix2 = np.zeros((len(PDmorphine), visse)) * np.nan
    for ix in range(len(PDmorphine)):
        for se in range(visse):
            PDmorphine_matrix2[ix,se] = np.mean(PDmorphine_matrix[ix][se])

    plt.figure()
    msplot = mssave2[PDpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    plt.plot(msplot_mean, c='r')
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    msplot = mssave2[PDnonpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    plt.plot(msplot_mean, c='b')
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    #
    
    same_days = [[2,3], [4,5], [6,7], [8,9]]
    mssave3 = np.zeros((N, len(same_days))) * np.nan
    for i in range(len(same_days)):
        mssave3[:,i] = np.nanmean(mssave2[:, same_days[i]], axis=1)
   
    plt.figure()
    msplot = mssave3[PDpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')
    
    msplot = mssave3[PDnonpain,:]
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')

    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    prismout = pd.DataFrame([])
    prismout = pd.concat((prismout, pd.DataFrame(mssave3[PDpain,:])), axis=0, ignore_index=True)
    prismout = pd.concat((prismout, pd.DataFrame(mssave3[PDnonpain,:])), axis=1, ignore_index=True)
    
    # + morphine
    
    plt.figure()
    plt.plot(np.nanmean(PDmorphine_matrix2, axis=0))
    msplot = PDmorphine_matrix2
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    # same_days = [[0], [1,2], [3,4], [5,6], [7,8]]
    # mssave3 = np.zeros((PDmorphine_matrix2.shape[0], len(same_days))) * np.nan
    # for i in range(len(same_days)):
    #     mssave3[:,i] = np.nanmean(PDmorphine_matrix2[:, same_days[i]], axis=1)
        
    # plt.plot()
    # msplot = mssave3
    # msplot_mean = np.nanmean(msplot, axis=0)
    # e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    # plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='r')

    return prismout, PDmorphine_matrix2


def ROC_merge(merge_dict):
    sz = 1
    lw = 2
    fig = plt.figure(1, figsize=(7*sz, 5*sz))
    for i in range(len(merge_dict)):
        fpr = merge_dict[i]['fpr']
        tpr = merge_dict[i]['tpr']
        roc_auc = merge_dict[i]['roc_auc']
        plt.plot(fpr, tpr, lw=lw, label= str(i) + ' (area = %0.2f)' % roc_auc) 
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")

def msreport_SNU_pslGBVX(mssave_p):
    group_save = msFunction.msarray([7])
    # 0 - base
    # 1 - sham d3
    # 2 - sham d10
    # 3 - psl d3
    # 4 - psl d10
    # 5 - psl d3 + GB/VX
    # 6 - psl d10 + GB/VX
    # psl
    group_save[3].append(mssave_p[[73, 80, 87, 93, 94],:][:,1])
    group_save[4].append(mssave_p[[73, 80, 87, 93, 94],:][:,2])
    
    group_save[3].append(np.nanmean(mssave_p[[70, 71, 75, 76, 77, 78, 79],:][:,1:3], axis=1))
    group_save[4].append(np.nanmean(mssave_p[[70, 71, 75, 76, 77, 78, 79],:][:,3:7], axis=1))
    
    # shamGroup
    group_save[1].append(mssave_p[shamGroup,:][:,1])
    group_save[2].append(mssave_p[shamGroup,:][:,2])
    
    # ipsaline_pslGroup
    group_save[4].append(np.nanmean(mssave_p[[141, 142, 143],:][:,1:3], axis=1))
    group_save[3].append(np.nanmean(mssave_p[[144, 145, 150, 152],:][:,[2,3,4,5]], axis=1))
    group_save[4].append(np.nanmean(mssave_p[[144, 145, 150, 152],:][:,[6,7,8,9]], axis=1))
    group_save[3].append(np.nanmean(mssave_p[[146, 158],:][:,[2,3,4,5]], axis=1))
    
    # ipclonidineGroup
    group_save[3].append(np.nanmean(mssave_p[ipclonidineGroup,:][:,[2,3]], axis=1))
    group_save[4].append(np.nanmean(mssave_p[ipclonidineGroup,:][:,[6,7]], axis=1))
    
    # GBVX
    group_save[6].append(np.nanmean(mssave_p[[164,165,166],:][:,[2,3]], axis=1))
    group_save[5].append(np.nanmean(mssave_p[[167, 172, 174],:][:,[4,5]], axis=1))
    group_save[6].append(np.nanmean(mssave_p[[167],:][:,[6,7]], axis=1))
    group_save[6].append(np.nanmean(mssave_p[[172],:][:,[8,9]], axis=1))
    group_save[5].append(np.nanmean(mssave_p[[177, 179, 181],:][:,[2,3]], axis=1))
    group_save[6].append(np.nanmean(mssave_p[[177, 179],:][:,[6,7]], axis=1))
    
    group_save2 = msFunction.msarray([7])
    sizesave = []
    for i in range(len(group_save)):
        for j in range(len(group_save[i])):
            group_save2[i] += list(group_save[i][j])
        group_save2[i] = msFunction.nanex(group_save2[i])
        sizesave.append(len(group_save2[i]))
        print(np.mean(group_save2[i]))
    
    mssave_p2 = np.zeros((np.max(sizesave), 7)) * np.nan
    for i in range(len(group_save2)):
        mssave_p2[:len(group_save2[i]), i] = group_save2[i]
    
    msplot = mssave_p2
    msplot_mean = np.nanmean(msplot, axis=0)
    e = scipy.stats.sem(msplot, axis=0, nan_policy='omit')
    plt.errorbar(range(len(msplot_mean)), msplot_mean, e, linestyle='None', marker='o', c='b')
    
    return mssave_p2


#%% dataload


filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220208_morphine'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)

Aprism, _ = ms_report(mssave[:,:,1])
Aprism2 = np.array(Aprism)

nonpain = Aprism2[:,[3,4]].flatten()
pain = Aprism2[:,[0,1]].flatten()
_,_,_, msdict = msFunction.msROC(nonpain, pain, figsw=True)

drug = Aprism2[:,2]
pain = Aprism2[:,0:2].flatten()
_,_,_, msdict2 = msFunction.msROC(drug, pain, figsw=True)

ROC_merge([msdict, msdict2])

# KHU PSL (drug-saline model)
filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220207_morphine_control'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)

_, Aprism = ms_report(mssave[:,:,1])
Aprism2 = np.array(Aprism)

pain = Aprism2[:,[0,2]].flatten()
pain_saline = Aprism2[:,[1,3]].flatten()

msFunction.msROC(pain_saline, pain, figsw=True)

drug = Aprism2[:,2]
pain = Aprism2[:,0:2].flatten()
msFunction.msROC(drug, pain, figsw=True)



# MPTP
filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220208_PD'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)


Aprism, Aprism2 = msplot_PD(mssave[:,:,1])

# MPTP - total
pain = list(np.array(Aprism)[:,0:4].flatten()) + list(Aprism2[:,[1,2,5,6]].flatten())
nonpain = list(np.array(Aprism)[:,4:8].flatten()) + list(Aprism2[:,0])
_,_,_, msdict1 = msFunction.msROC(nonpain, pain, figsw=True)

# MPTP - first
pain = list(np.array(Aprism)[:,0:4].flatten()) 
nonpain = list(np.array(Aprism)[:,4:8].flatten()) 
_,_,_, msdict2 = msFunction.msROC(nonpain, pain, figsw=True)

# MPTP - second
pain = list(Aprism2[:,[1,2,5,6]].flatten())
nonpain = list(Aprism2[:,0])
_,_,_, msdict3 = msFunction.msROC(nonpain, pain, figsw=True)

ROC_merge([msdict1, msdict2, msdict3])


# MPTP = second - MPTP vs morphine

pain = list(Aprism2[:,[1,2,5,6]].flatten())
drug = list(Aprism2[:,[3,4,7,8]].flatten())
_,_,_, msdict3 = msFunction.msROC(drug, pain, figsw=True)


# SNU - PSL
filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220208_GBVX'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)

mssave_p2 = msreport_SNU_pslGBVX(mssave[:,:,1])

# PSL vs sham
pain = mssave_p2[:,[3,4]].flatten()
nonpain = mssave_p2[:,[1,2]].flatten()
_,_,_, msdict1 = msFunction.msROC(nonpain, pain, figsw=True)

# PSL vs GBVX
pain = mssave_p2[:,[4]].flatten()
drug = mssave_p2[:,[6]].flatten()
_,_,_, msdict2 = msFunction.msROC(drug, pain, figsw=True)

pain = mssave_p2[:,[3]].flatten()
drug = mssave_p2[:,[5]].flatten()
_,_,_, msdict3 = msFunction.msROC(drug, pain, figsw=True)

ROC_merge([msdict1, msdict2, msdict3])

# magnolin
filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220208_magnolin2'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)

Aprism = ms_report_khu_magnolin(mssave[:,:,1])

pain = Aprism[:,1]
nonpain = Aprism[:,0]
_,_,_, msdict1 = msFunction.msROC(nonpain, pain, figsw=True)

pain = Aprism[:,1]
drug = Aprism[:,4]
_,_,_, msdict2 = msFunction.msROC(drug, pain, figsw=True)

ROC_merge([msdict1, msdict2])


# keto
filepath = 'C:\\SynologyDrive\\2p_data\\'
RESULT_SAVE_PATH = filepath + 'model5_20220207_keto'
savepath = RESULT_SAVE_PATH + '\\repeat_save.pickle'
with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
    repeat_save = pickle.load(f)
    mssave = np.nanmean(np.array(repeat_save), axis=0)

Aprism_100, Aprism_50 = ms_report_cfa(mssave[:,:,1])


pain = list(Aprism_100[:,[1,3]].flatten()) + list(Aprism_50[:,[1,3,4,5]].flatten())
nonpain = list(Aprism_100[:,0]) + list(Aprism_50[:,0])
_,_,_, msdict1 = msFunction.msROC(nonpain, pain, figsw=True)

pain = list(Aprism_100[:,[1,3]].flatten())
drug = Aprism_100[:,2]
_,_,_, msdict2 = msFunction.msROC(drug, pain, figsw=True)

pain = list(Aprism_50[:,[1,3,4,5]].flatten())
drug = Aprism_50[:,2]
_,_,_, msdict3 = msFunction.msROC(drug, pain, figsw=True)


ROC_merge([msdict1, msdict2, msdict3])








