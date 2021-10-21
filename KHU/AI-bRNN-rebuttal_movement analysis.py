# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:43:31 2021

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

MAXSE = 20
#%% data import

gsync = 'C:\\mass_save\\PSLpain\\'
with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['behavss2']   # 움직임 정보
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

movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        behav_tmp = bahavss[SE][se][0]
        if len(behav_tmp) > 0:
            movement_syn[SE][se] = msFunction.downsampling(behav_tmp, signalss[SE][se].shape[0])
            if np.isnan(np.mean(movement_syn[SE][se])): movement_syn[SE][se] = []

movement = np.zeros((N, MAXSE)) * np.nan
t4 = np.zeros((N, MAXSE)) * np.nan
for SE in range(N):
    for se in range(len(signalss[SE])):
        movement[SE,se] = np.mean(movement_syn[SE][se] >  bahavss[SE][se][1])
        t4[SE,se] = np.mean(signalss[SE][se])

# movement        

movement_saline = pd.Series(movement[salineGroup, 1].flatten())
movement_formalin = pd.Series(movement[highGroup + midleGroup, 1])
movement_cap = pd.Series(movement[capsaicinGroup, 1])
movement_cfa_1d = pd.Series(movement[CFAgroup, 1])
movement_psl_3d = pd.Series(movement[pslGroup, 1])

movement_df = pd.DataFrame([])
movement_df = pd.concat([movement_df, movement_saline], axis=1, ignore_index=True)
movement_df = pd.concat([movement_df, movement_formalin], axis=1, ignore_index=True)
movement_df = pd.concat([movement_df, movement_cap], axis=1, ignore_index=True)
movement_df = pd.concat([movement_df, movement_cfa_1d], axis=1, ignore_index=True)
movement_df = pd.concat([movement_df, movement_psl_3d], axis=1, ignore_index=True)

# t4

t4_saline = pd.Series(t4[salineGroup, 1].flatten())
t4_formalin = pd.Series(t4[highGroup + midleGroup, 1])
t4_cap = pd.Series(t4[capsaicinGroup, 1])
t4_cfa_1d = pd.Series(t4[CFAgroup, 1])
t4_psl_3d = pd.Series(t4[pslGroup, 1])

t4_df = pd.DataFrame([])
t4_df = pd.concat([t4_df, t4_saline], axis=1, ignore_index=True)
t4_df = pd.concat([t4_df, t4_formalin], axis=1, ignore_index=True)
t4_df = pd.concat([t4_df, t4_cap], axis=1, ignore_index=True)
t4_df = pd.concat([t4_df, t4_cfa_1d], axis=1, ignore_index=True)
t4_df = pd.concat([t4_df, t4_psl_3d], axis=1, ignore_index=True)
























