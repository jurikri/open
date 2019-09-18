# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:09:53 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
print('savepath', savepath)

# 저장경로 유효성 test
#import pandas as pd
import csv

df2 = [['SE', 'se', '%']]
df2.append([1, 1, 1])

csvfile = open('mscsvtest.csv', 'w', newline='')
csvwriter = csv.writer(csvfile)
for row in range(len(df2)):
    csvwriter.writerow(df2[row])

csvfile.close()

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
#with open('mspickle_msdict.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
#    msdict = pickle.load(f)
#    msdict = msdict['msdict']
    
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

movement2 = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        movement2[SE,se] = np.mean(bahavss[SE][se][0:int(round(497/4.3*64))])


# behav2 최적 thr 찾

axiss = []; [axiss.append([]) for i in range(2)]
for thr in np.arange(0,13,0.1):
    loss = 0
    for SE in range(N):
        for se in range(5):
            b1 = np.mean(bahavss[SE][se][0:int(round(497/4.3*64))])
            b2 = np.mean(behavss2[SE][se][0:497] > thr)
            
            loss += np.abs(b1-b2)
            

#    print(thr, loss)
    
    axiss[0].append(thr)
    axiss[1].append(loss)
    
plt.plot(axiss[0],axiss[1])

optimized_behavs2_thr = axiss[0][np.argmin(np.array(axiss[1]))]
print('optimized_behavs2_thr', optimized_behavs2_thr)



# In[] Trace 시각화 및 저장 
savepath2 = 'E:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\heatmap\\'
print('savepath2', savepath2)
if not os.path.exists(savepath2):
    os.mkdir(savepath2)

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

SE=0;se=0
signalss2 = []; [signalss2.append([]) for i in range(N)]
for SE in range(N):
    signalss2[SE]; [signalss2[SE].append([]) for i in range(5)]
    for se in range(5):
        pre_order = np.array(signalss[SE][se][:497,:])
        oredered_ix = np.argsort(np.mean(pre_order, axis=0))[::-1]
        
        signalss2[SE][se] = pre_order[:, oredered_ix]


#plt.imshow(np.transpose(pre_order))
#plt.figure()
#plt.imshow(np.transpose(pre_order[:, oredered_ix]))
     
# In[] heatmap 시각화 (session 별로 나누어봅시다 )
#import seaborn

for SE in range(N):
    if SE in grouped_total_list:
        signals = signalss2[SE]
        behavs = bahavss[SE]
        
        msplot_merge = np.transpose(np.array(signals[0]))
        for se in range(1,5):
            msplot = np.transpose(np.array(signals[se]))
            msplot_merge = np.concatenate((msplot_merge, msplot), 1)
        
        msmax = np.max(np.mean(msplot_merge,0))
        
        fig = plt.figure(1, figsize=(9.7*1.5, 6*1.5))
        plt.subplots_adjust(hspace = 0.3, wspace = 0.05)
        
        for se in range(5):
            msplot = np.transpose(np.array(signals[se]))
        
            ax1 = plt.subplot(2,5,1 + se)
            
        
            im1 = ax1.imshow(msplot, cmap='jet', aspect='auto', vmin=0, vmax=msmax)
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            
            if se > 0:
                ax1.axes.get_yaxis().set_visible(False)         
                ax1.spines['left'].set_visible(False)
            
        fig.colorbar(im1)
            
        for se in range(5):
            msplot = np.transpose(np.array(signals[se]))
            ax2 = plt.subplot(2,5,6 + se, sharex = ax1)
            ax2.plot(np.mean(msplot,0), c = 'black')
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            
            ax2.set_ylim([0,msmax])
            
            if se > 0:
                ax2.axes.get_yaxis().set_visible(False)
                ax2.spines['left'].set_visible(False)
                
            for xindex in np.where(behavss2[SE][se][:497]>7)[0]:
                try:
                    ax2.fill_between([xindex-0.5,xindex+0.5], 0, msmax, color = 'lightskyblue', alpha=0.5)
                except:
                    try:
                        ax2.fill_between([xindex,xindex+0.5], 0, msmax, color = 'lightskyblue', alpha=0.5)
                    except:
                        ax2.fill_between([xindex-0.5,xindex], 0, msmax, color = 'lightskyblue', alpha=0.5)
                        
        plt.savefig(savepath2 + str(SE) + '_heatmap.png', dpi=1000)
        plt.close()

# In[] venn diagram으로 signal의 overlap %를 시각화함.
# 이 계산방식은 사실상 idnex1과 동일하지만, 정규화를 base, session 간의 %로 한다는 차이가 존재함.
# 실제 index1은 전체 %가 아니라, signal intensity 자체를 사용함. 

import matplotlib_venn as mv

vennsave = np.zeros((N,5,3))        
for SE in range(N):
    signals = signalss[SE]
    base = np.array(basess[SE])
    
    for se in range(5):
        signal = np.array(signals[se])   
        
        basetotal = np.sum(np.sum(base))
        signaltotal = np.sum(np.sum(signal))
        
        mergetotal = list()
        for n in range(signal.shape[1]):
            mergetotal.append(np.min([np.sum(base[:,n]), np.sum(signal[:,n])]))
            
        mergetotal = np.array(mergetotal); mergetotal = np.sum(mergetotal)
        
        total = basetotal + signaltotal - mergetotal
        
        vennsave[SE,se,0] = (basetotal - mergetotal) / signal.shape[1]
        vennsave[SE,se,1] = mergetotal / signal.shape[1]
        vennsave[SE,se,2] = (signaltotal - mergetotal) / signal.shape[1]

def venn_mean(vennsave, see, state, group, testsw): 
    # see : session state 0 for non-pain, 1 for pain
    # sate : 0 for base(pure), 1 for merge, 2 for session(pure)
    # group : 1 or 0, pain group or nonpain group
    # testsw : pain grouping
    
    if testsw == 1:
        testpaingroup = msGroup['highGroup'] + msGroup['midleGroup'] + msGroup['restrictionGroup']
    elif testsw == 2:
        testpaingroup = msGroup['restrictionGroup']
        
    if group == 1:
        if see == 1: # pain group의 early 및 late 
            msreturn = np.mean(np.concatenate((vennsave[testpaingroup,1,state],vennsave[testpaingroup,3,state])))
            
        elif see == 0: # pain group의 inter
            msreturn = np.mean(np.concatenate((vennsave[testpaingroup,2,state],vennsave[testpaingroup,4,state])))
        
    elif group == 0:
        msreturn = np.mean(vennsave[msGroup['salineGroup'],:,state][baseindex[msGroup['salineGroup'],:]==0])
        
    return msreturn


# for salinegroup
see = 1; group = 0; testsw = 1
ven_merge = venn_mean(vennsave, see, 1, group, testsw)
ven_B = venn_mean(vennsave, see, 2, group, testsw)
ven_A = venn_mean(vennsave, see, 0, group, testsw)

plt.figure(0)
mv.venn2(subsets = (round(ven_A,2), round(ven_B,2), round(ven_merge,2)), set_labels = ('baseline', 'phase')) 
plt.title("Saline group")

# for painGroup_early & late 
see = 1; group = 1; testsw = 1
ven_merge = venn_mean(vennsave, see, 1, group, testsw) 
ven_B = venn_mean(vennsave, see, 2, group, testsw) 
ven_A = venn_mean(vennsave, see, 0, group, testsw)

plt.figure(1)
mv.venn2(subsets = (round(ven_A,2), round(ven_B,2), round(ven_merge,2)), set_labels = ('baseline', 'phase')) 
plt.title("painGroup_early & late")
        
# for painGroup_inter & recover
see = 0; group = 1; testsw = 1
ven_merge = venn_mean(vennsave, see, 1, group, testsw)
ven_B = venn_mean(vennsave, see, 2, group, testsw)
ven_A = venn_mean(vennsave, see, 0, group, testsw)

plt.figure(2)
mv.venn2(subsets = (round(ven_A,2), round(ven_B,2), round(ven_merge,2)), set_labels = ('baseline', 'phase')) 
plt.title("painGroup_inter & recover")



























