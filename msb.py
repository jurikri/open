# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:38:56 2019

@author: msbak
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
import os

try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    savepath = ''; # os.chdir(savepath);
print('savepath', savepath)
#

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']
behavss2 = msdata_load['behavss2']
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

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]
# 최종 평가 함수 
def accuracy_cal(pain, non_pain, fsw):
    pos_label = 1; roc_auc = -np.inf
    
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
        fig = plt.figure(1, figsize=(5*sz, 5*sz))
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

# In[] pain 때 뜬 cell은 움직임에 의해 우연히 뜰 확률에 못미침.
# pain cell이 있다고 말하기 힘듬 

movement2 = np.zeros((N,5))
SE = 0; se = 0
for SE in range(N):
    for se in range(5):
        movement2[SE,se] = np.mean(bahavss[SE][se][0:int(round(497/4.3*64))])
        
print(movement2)

def newCell_calc(thr=0, figsw=False):
    target = np.zeros((N,5))
    
    SE = 0; se = 0
    for SE in range(N):
        for se in range(5):
            signal = np.array(signalss[SE][se][0:497,:])
            
            target[SE,se] = np.sum(np.mean(signal, axis=0) > thr) / signal.shape[1]
    
    axiss = []; [axiss.append([]) for i in range(4)]
    for SE in range(N):
        for se in range(5):
            for se2 in range(5):
                if se < se2:          
                    if SE in notpainGroup:
                        dMovement = movement2[SE,se2]-movement2[SE,se]
                        dNeuron = target[SE,se2]-target[SE,se]
                    
                        axiss[0].append(dMovement)
                        axiss[1].append(dNeuron)
                            
        if SE in painGroup:
            dMovement = movement2[SE,1]-movement2[SE,0]
            dNeuron = target[SE,1]-target[SE,0]
        
            axiss[2].append(dMovement)
            axiss[3].append(dNeuron)
            
    m, b = mslinear_regression(axiss[0],axiss[1])

    #    print(m, b)
    
    m2, b2 = mslinear_regression(axiss[2],axiss[3])
    
    if figsw:
        plt.figure(figsize = (9.7,6))
        plt.title('newcell_population_estimation_notpain')
        plt.scatter(axiss[0],axiss[1])
        plt.xlabel('dMovement')
        plt.ylabel('dNeuron %')
        xaxis = np.arange(np.min(axiss[0]),np.max(axiss[0]),np.max(axiss[0])/10)
        plt.plot(xaxis, m * xaxis + b, c = 'orange')
        plt.savefig('newcell_population_estimation_notpain.png')
        
        plt.figure(figsize = (9.7,6))
        plt.title('newcell_population_estimation_pain')
        plt.scatter(axiss[2],axiss[3])
        plt.xlabel('dMovement')
        plt.ylabel('dNeuron %')
        xaxis = np.arange(np.min(axiss[2]),np.max(axiss[2]),np.max(axiss[2])/10)
        plt.plot(xaxis, m * xaxis + b, c = 'orange')
        plt.savefig('newcell_population_estimation_pain.png')

    dslope = m2-m
    
    return thr, dslope, m, b, m2, b2, target
    
painGroup = highGroup + midleGroup + yohimbineGroup # + ketoGroup + capsaicinGroup
notpainGroup = salineGroup # + lidocainGroup

axiss2 = []; [axiss2.append([]) for i in range(4)]
for thr in np.arange(0,2,0.05):
    thr, dslope, _, _, _, _, _ = newCell_calc(thr=thr, figsw=False)
    
    axiss2[0].append(thr)
    axiss2[1].append(dslope)

plt.figure(figsize = (9.7,6))
plt.plot(axiss2[0],axiss2[1])
plt.xlabel('thr')
plt.ylabel('dNeuron %')
plt.savefig('newcell_population_on_var_thr.png')

thr = 0.3
thr, dslope, m, b, m2, b2, target = newCell_calc(thr=thr, figsw=True)

experimental = []; bychance = []
for SE in range(N):
    if SE in painGroup:
        dMovement = movement2[SE,1]-movement2[SE,0]
        dNeuron = target[SE,1]-target[SE,0]
        
        experimental.append(dNeuron)
        bychance.append(dMovement * m + b)
        
plt.figure(figsize = (6,9.7))
plt.scatter(np.zeros(len(bychance)), bychance)
plt.xlabel('chance-level vs real')
plt.ylabel('dNeuron # %')
plt.scatter(0, np.mean(bychance), marker='_', s=1000)
plt.scatter(np.ones(len(experimental)), experimental)
plt.scatter(1, np.mean(experimental), marker='_', s=1000)
plt.savefig('chance-level vs real.png')

Aprism_bychance = np.array(bychance)
Aprism_experimental = np.array(experimental)

# In[] Total acitivity :497

def msGrouping_nonexclude(msmatrix): # base 예외처리 없음, goruping된 sample만 뽑힘
    target = np.array(msmatrix)
    
    df3 = pd.DataFrame(target[highGroup]) 
    df3 = pd.concat([df3, pd.DataFrame(target[midleGroup]), \
                     pd.DataFrame(target[salineGroup]), \
                     pd.DataFrame(target[ketoGroup]), pd.DataFrame(target[lidocainGroup]), \
                     pd.DataFrame(target[yohimbineGroup]), pd.DataFrame(target[capsaicinGroup][:,0:3])], \
                        ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

totalActivity = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        totalActivity[SE,se] = np.mean(signalss[SE][se][0:497,:])

Aprism_totalActivity = msGrouping_nonexclude(totalActivity)

# ROC by totalActivity 
target = np.array(totalActivity)

painGroup = highGroup + midleGroup + yohimbineGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup

pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy, roc_auc, fig = accuracy_cal(pain, nonpain, True)
fig.savefig('test.png', dpi=1000)

# Movement - Total corr

axiss = []; [axiss.append([]) for i in range(4)]
for SE in range(N):
    for se in range(5):
        if SE in painGroup and se == 1:
            axiss[2].append(movement2[SE,se])
            axiss[3].append(totalActivity[SE,se])
            
        elif SE in nonpainGroup or (SE in painGroup and (se==0 or se==2)):
            axiss[0].append(movement2[SE,se])
            axiss[1].append(totalActivity[SE,se])

plt.figure(1, figsize=(9.7, 6)) 

plt.scatter(axiss[0], axiss[1])
plt.scatter(axiss[2], axiss[3])
msmax = np.max(axiss[0]+axiss[2])
xaxis = np.arange(0, msmax, msmax/10)

m, b = mslinear_regression(axiss[0], axiss[1])
plt.plot(xaxis, xaxis*m + b, label='not-painSession')

m, b = mslinear_regression(axiss[2], axiss[3])
plt.plot(xaxis, xaxis*m + b, label='painSession')

plt.xlabel('movement ratio', fontsize=18)
plt.ylabel('(ΔF/ F0) / (ROIs * Frame)', fontsize=18)
plt.legend(fontsize=18)
plt.savefig('test2.png', dpi=1000)
# In[] movement :497
movement2 = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        movement2[SE,se] = np.mean(bahavss[SE][se][0:int(round(497/4.3*64))])

Aprism_movement = msGrouping_nonexclude(movement2)






















































