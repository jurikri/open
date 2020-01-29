# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:12:16 2019

@author: msbak
"""


import os
import pickle 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import metrics
import random

try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

with open(savepath + 'msGroup_ksh.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    msGroup = pickle.load(f)
    print('msGroup_ksh.pickle load')


capsaicinGroup = msGroup['capsaicinGroup']
vehicleGroup = msGroup['vehicleGroup'] 
signalss = msGroup['signalss']
signalss_pc = msGroup['signalss_pc']
behavss = msGroup['behavss']
bg = msGroup['bg']

N = len(signalss)

##

def msacc(target, figsw=False):
    target = np.array(target)
    pain = []; nonpain = []
    for SE in range(N):
        for se in range(3):
            c1 = SE in capsaicinGroup and se in [0]
            c2 = SE in vehicleGroup and se in [0,1,2]
            
            if SE in capsaicinGroup and se in [1]:
                pain.append(target[SE,se])
                
            elif c1 or c2:
                nonpain.append(target[SE,se])
                
    pvalue = stats.ttest_ind(pain, nonpain)[1]
    
    pos_label = 1; roc_auc = -np.inf; fig = None
    while roc_auc < 0.5:
        pain = np.array(pain); nonpain = np.array(nonpain)
        pain = pain[np.isnan(pain)==0]; nonpain = nonpain[np.isnan(nonpain)==0]
        
        anstable = list(np.ones(pain.shape[0])) + list(np.zeros(nonpain.shape[0]))
        predictValue = np.array(list(pain)+list(nonpain)); predictAns = np.array(anstable)
        #            
        fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
        
        maxix = np.argmax((1-fpr) * tpr)
        specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
        accuracy = ((pain.shape[0] * sensitivity) + (nonpain.shape[0]  * specificity)) / (pain.shape[0] + nonpain.shape[0])
        roc_auc = metrics.auc(fpr,tpr)
        
        if roc_auc < 0.5:
            pos_label = 0
            
    if figsw:
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
            
    return pvalue, roc_auc, accuracy, fig

def msacc2(pain, nonpain):
    pos_label = 1; roc_auc = -np.inf
    
    pain = np.array(pain); nonpain = np.array(nonpain)
    pain = pain[np.isnan(pain)==0]; nonpain = nonpain[np.isnan(nonpain)==0]
    
    anstable = list(np.ones(pain.shape[0])) + list(np.zeros(nonpain.shape[0]))
    predictValue = np.array(list(pain)+list(nonpain)); predictAns = np.array(anstable)
    #            
    fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
    
    maxix = np.argmax((1-fpr) * tpr)
    specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
    accuracy = ((pain.shape[0] * sensitivity) + (nonpain.shape[0]  * specificity)) / (pain.shape[0] + nonpain.shape[0])
    roc_auc = metrics.auc(fpr,tpr)
      
    return accuracy

def smoothListGaussian(array1,window):  
     window = round(window)
     degree = (window+1)/2
     weight=np.array([1.0]*window)  
     weightGauss=[]  

     for i in range(window):  
         i=i-degree+1  
         frac=i/float(window)  
         gauss=1/(np.exp((4*(frac))**2))  
         weightGauss.append(gauss)  

     weight=np.array(weightGauss)*weight  
     smoothed=[0.0]*(array1.shape[0]-window)
     
     weight = weight / np.sum(weight) # nml

     for i in range(len(smoothed)):  
         smoothed[i]=sum(np.array(array1[i:i+window])*weight)/sum(weight)  

     return smoothed 

# downsampling
def downsampling(signalss, wanted_size):
    downratio = signalss[0][0].shape[0]/wanted_size
    signalss2 = []; [signalss2.append([]) for u in range(N)]
    for SE in range(N):
        [signalss2[SE].append([]) for u in range(3)]
        for se in range(3):
            signal = np.array(signalss[SE][se])
            
            downsignal = np.zeros(wanted_size)
            downsignal[:] = np.nan
            for frame in range(wanted_size):
                s = int(round(frame*downratio))
                e = int(round(frame*downratio+downratio))
                downsignal[frame] = np.mean(signal[s:e])
                
            signalss2[SE][se] = downsignal
    return np.array(signalss2)

def mslinear_regression(x,y):
    x = np.array(x); y = np.array(y); 
    x = x[np.isnan(x)==0]; y = y[np.isnan(y)==0]
    
    n = x.shape[0]
    r = (1/(n-1)) * np.sum(((x - np.mean(x))/np.std(x)) * ((y - np.mean(y))/np.std(y)))
    m = r*(np.std(y)/np.std(x))
    b = np.mean(y) - np.mean(x)*m

    return m, b # bx+a

##
# In[] movement, t4, roc

behavss = np.array(msGroup['behavss'])
behavss2 = []; [behavss2.append([]) for u in range(N)]  
for SE in range(N):
    print('behav, smooth', SE)
    [behavss2[SE].append([]) for u in range(3)]
    for se in range(3):
        behavss2[SE][se] = np.array(smoothListGaussian(behavss[SE][se],40))
        
        if False: # plot save 
            plt.plot(behavss2[SE][se])
            savename = str(SE) + '_' + str(se) + '.png'
            savepath1 = 'E:\\ksh_perkinje\\visualization_save\\movement_filter\\'
            plt.savefig(savepath1 + savename, dpi=1000)
            plt.close()
        
## threshold binarization
for SE in range(N):
    for se in range(3):
        behavss2[SE][se] = behavss2[SE][se] > 100 # behav thr
               
signalss = np.array(msGroup['signalss'])
signalss2 = []; [signalss2.append([]) for u in range(N)]
for SE in range(N):
    print('signal, smooth', SE)
    [signalss2[SE].append([]) for u in range(3)]
    for se in range(3):
        signalss2[SE][se] = np.array(smoothListGaussian(signalss[SE][se],40))

wanted_size = np.min([behavss2[0][0].shape[0], signalss2[0][0].shape[0]])
signalss2 = downsampling(signalss2, wanted_size)
behavss2 = downsampling(behavss2, wanted_size)

#    print(window, behavss2[0][0].shape[0], signalss[0][0].shape[0])

# In[]
movement = np.zeros((N,3))
for SE in range(N):
    for se in range(3):
#        mssignal = signalss2[SE][se]
        msbehav = behavss2[SE][se]
        
        movement[SE,se] = np.mean(msbehav)
    
    
t4 = np.zeros((N,3))
for SE in range(N):
    for se in range(3):
        mssignal = signalss2[SE][se]
#        msbehav = behavss[SE][se]
        
        t4[SE,se] = np.mean(mssignal)

pvalue, roc_auc, accuracy, fig = msacc(movement, figsw=True)   
print(pvalue, roc_auc)
plt.title('movement (total)')
fig.savefig(savepath + 'ksh_result\\' + 'ksh_movement_roc.png', dpi=1000)

plt.close()

pvalue, roc_auc, accuracy, fig = msacc(t4, figsw=True)   
print(pvalue, roc_auc)
plt.title('activity (total)')
fig.savefig(savepath + 'ksh_result\\' + 'ksh_activity_roc.png', dpi=1000)
plt.close()
# In[] movement, t4, 2mins version
##
movement_2mins = np.zeros((N,3))
for SE in range(N):
    for se in range(3):
#        mssignal = signalss2[SE][se]
        msbehav = behavss2[SE][se]
        movement_2mins[SE,se] = np.mean(msbehav[:int(msbehav.shape[0]/5)])
    
    
t4_2mins = np.zeros((N,3))
for SE in range(N):
    for se in range(3):
        mssignal = signalss2[SE][se]
#        msbehav = behavss[SE][se]
        
        t4_2mins[SE,se] = np.mean(mssignal[:int(mssignal.shape[0]/5)])

pvalue, roc_auc, accuracy, fig = msacc(movement_2mins, figsw=True)   
print(pvalue, roc_auc)
plt.title('movement (2mins)')
fig.savefig(savepath + 'ksh_result\\' + 'ksh_movement_2mins_roc.png', dpi=1000)
plt.close()

pvalue, roc_auc, accuracy, fig = msacc(t4_2mins, figsw=True)   
print(pvalue, roc_auc)
plt.title('activity (2mins)')
fig.savefig(savepath + 'ksh_result\\' + 'ksh_activity_2mins_roc.png', dpi=1000)
plt.close()


# In[] import bRNN
with open(savepath + 'msGroup_ksh_bRNN.pickle', 'rb') as f:  # Python 3: open(..., 'wb')
    bRNN_mean = np.array(pickle.load(f))
    print('msGroup_ksh_bRNN.pickle load')

pvalue, roc_auc, accuracy, fig = msacc(bRNN_mean , figsw=True)   
print(pvalue, roc_auc)
plt.title('bRNN')
fig.savefig(savepath + 'ksh_result\\' + 'ksh_bRNN_roc.png', dpi=1000)
plt.close()

# In[] 움직임 noise 문제 해결하기 - new

# 랜덤한 갯수로 nonpain을 뽑아서 평균 movement와, 그 그룹만 썻을때 accuracy를 2d plot
# 서로 상관관계가 있는지.
    # bRNN_mean
    
def movement_shuffle(t4):
    target_n = np.array(t4)
    movement_n =  np.array(movement)
    
    pain = []; nonpain = []
    pain_movement = []; nonpain_movement = []
    for SE in range(N):
        for se in range(3):
            c1 = SE in capsaicinGroup and se in [0]
            c2 = SE in vehicleGroup and se in [0,1,2]
            
            if SE in capsaicinGroup and se in [1]:
                pain.append(target_n[SE,se])
                pain_movement.append(movement_n[SE,se])
                
            elif c1 or c2:
                nonpain.append(target_n[SE,se])
                nonpain_movement.append(movement_n[SE,se])
    
    axiss = []; [axiss.append([]) for i in range(2)]
    totalNum = len(pain)
    epochs = 10000
    for epoch in range(epochs):
        if epoch % int(epochs/10) == 1:
            print(epoch, '/', epochs)
        random_N = random.randrange(totalNum)
        ixlist = random.sample(list(range(totalNum)), random_N)
        
        accuracy = msacc2(pain, np.array(nonpain)[ixlist])   
        axiss[0].append(np.mean(np.array(nonpain_movement)[ixlist], axis=0))
        axiss[1].append(accuracy)
    
    for d in range(2):
        axiss[d] = np.array(axiss[d])
        axiss[d] = axiss[d][np.isnan(axiss[d])==0];
        
#    maxvalue = np.max([np.concatenate((pain, nonpain))])
        
    m, b = mslinear_regression(axiss[0], axiss[1])
    print('slope', m)
    
    fig = plt.figure()
    plt.scatter(axiss[0], axiss[1], s = 1)
    plt.xlabel('movement ratio mean')
    plt.ylabel('accuracy by movement as feature')
    
    xaxis = np.arange(0, np.max(movement_n), np.max(movement_n)/100)
    plt.plot(xaxis, xaxis*m + b, c = 'orange')
    plt.xlim([0,np.max(movement_n)]), plt.ylim([0,1])
    
    return fig
    
fig = movement_shuffle(bRNN_mean)
print('bRNN_mean', pvalue, roc_auc)
plt.title('bRNN')
plt.savefig(savepath + 'ksh_result\\' + 'ksh_movement_bRNN_mean_shuffle.png', dpi=1000)
plt.close()

fig = movement_shuffle(movement)
print('movement', pvalue, roc_auc)
plt.title('movement (+ control)')
plt.savefig(savepath + 'ksh_result\\' + 'ksh_movement_movement_shuffle.png', dpi=1000)
plt.close()

# In[]












#

