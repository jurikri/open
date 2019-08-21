
import pandas as pd
#import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

import pickle
import os
try:
    savepath = 'C:\\Users\\user\\Google 드라이브\\BMS Google drive\\희라쌤\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\msbak\\Documents\\tensor\\'; os.chdir(savepath);
    except:
        savepath = ''; # os.chdir(savepath);
print('savepath', savepath)
#

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
with open('mspickle_msdict.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdict = pickle.load(f)
    msdict = msdict['msdict']
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']
behavss2 = msdata_load['behavss2']
#baseindex = msdata_load['baseindex']
#basess = msdata_load['basess']
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
#
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
        plt.figure()
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
        
    return accuracy, roc_auc 
    
def pain_nonpain_sepreate(target, painGroup, nonpainGroup):
    valuetable = np.array(target)
    painset = []; nonpainset =[]

    for i in mouseGroup:
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


# In data load하여 pain nonpain grouping
savepath = 'C:\\Users\\user\\Google 드라이브\\BMS Google drive\\희라쌤\\save\\tensorData\\'; os.chdir(savepath)
model_name = '0819_test' # <<<<<<<<<<<<<<<<<<<
# 0725_12_simpleRNN
# 0731_semisupervised_v4
#def model_data_import():


mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))
      
#mouseGroup.remove(63)      
    # In[] gee
       
rawsave = []
for i in range(N):
    rawsave.append([])

for repeat in range(1,20):
    msloadpath = savepath + 'result\\' + model_name + '_' + str(repeat) + '\\'; printsw=True
    
    try:
        os.chdir(msloadpath + 'exp_raw')
        if printsw:
            print('repeat', repeat, 'is loaded'); printsw=False
    except:
        continue
        
    for SE in mouseGroup:
        try:
            df1 = pd.read_csv('biRNN_raw_' + str(SE) + '.csv', header = None)
        except:
            print(repeat, 'repeat에', SE, 'data 없음')
            continue
            
        df2 = np.array(df1)
        rawsave[SE].append(df2)
        
for SE in range(N):
    rawsave[SE] = np.mean(np.array(rawsave[SE]), axis=0)
    
def ms_thresholding(optimalThr = 0):
    target = np.zeros((N,5)); target[:] = np.nan
    for SE in mouseGroup:
        ROInum = signalss[SE][0].shape[1]
        for se in range(5):
#            print('SE', se)
    
            # 현재: ROI를 frame 상관없이 모두 모아서 '확률을' 평균낸뒤, thr 이하면 0으로 처리한 다음 ROI들을 모두 평균낸다. 
            valuesave = np.zeros(ROInum); valuesave[:] = np.nan 
            for ROI in range(ROInum):
                
                rowindex = np.where((np.array(rawsave[SE][:,1]==se) * np.array( rawsave[SE][:,2]==ROI)) == True)[0]
                valuesave[ROI] = np.mean(rawsave[SE][rowindex,4]>0.5) # 4 for pain probability 
                # >0.5 를 사용하면 loss -> acc로 변환
                
#            valuesave = valuesave > optimalThr # binary
            valuesave[valuesave < optimalThr] = 0 # thr 이하의 값은 noise로 취급, 0으로 처리
                
            target[SE,se] = np.mean(valuesave) # 데이터들 수정후, mean으로 교체
    return target

    
# In[]

optimized_thr = 0
print('optimized_thr', optimized_thr)


target = ms_thresholding(optimized_thr)

# 최종 평가
print('______________ biRNN ______________')

painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

print('highGroup', np.nanmean(target[highGroup,1]))
print('midleGroup', np.nanmean(target[midleGroup,1]))
print('lowGroup', np.nanmean(target[lowGroup,1]))
print('restrictionGroup', np.nanmean(target[restrictionGroup,1]))
print('ketoGroup', np.nanmean(target[ketoGroup,1]))
print('salineGroup', np.nanmean(target[salineGroup,1]))
print('lidocainGroup', np.nanmean(target[lidocainGroup,1]))
print('capsaicinGroup ', np.nanmean(target[capsaicinGroup ,1]))

# control, 그룹내, 그룹외 추가 요망

# 개별평가
# formalin - within
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('formalin - within')
accuracy_cal(pain, nonpain_within, True)

# formalin - between
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('formalin - between')
accuracy_cal(pain, nonpain_between, True)

# capsaicin - within
painGroup = capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('capsaicin - within')
accuracy_cal(pain, nonpain_within, True)

# capsaicin - between
painGroup = capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
print('capsaicin - between')
accuracy_cal(pain, nonpain_between, True)

# In[] test용 임시구문

# 개별평가
if False:
    tartget_tmp = t10
    
    # formalin - within
    painGroup = highGroup + midleGroup + ketoGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('formalin - within')
    accuracy_cal(pain, nonpain_within, True)
    
    # formalin - between
    painGroup = highGroup + midleGroup + ketoGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('formalin - between')
    accuracy_cal(pain, nonpain_between, True)
    
    # capsaicin - within
    painGroup = capsaicinGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('capsaicin - within')
    accuracy_cal(pain, nonpain_within, True)
    
    # capsaicin - between
    painGroup = capsaicinGroup
    nonpainGroup = salineGroup + lidocainGroup
    pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(tartget_tmp, painGroup, nonpainGroup)
    print('capsaicin - between')
    accuracy_cal(pain, nonpain_between, True)

# In[]
painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
print('movment--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(movement, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)

painGroup = highGroup + midleGroup + ketoGroup
nonpainGroup = salineGroup + lidocainGroup
print('total--------------------------------')
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(msdict['total'], painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)
accuracy_cal(pain, nonpain, True)
# In[] 취약점 분석
# predict_matrix_total
painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup

targetMatrix = np.array(target)
pain, nonpain_within, nonpain_between = pain_nonpain_sepreate(targetMatrix, painGroup, nonpainGroup)
nonpain = np.concatenate((nonpain_within, nonpain_between), axis=0)

pain1 = np.array(pain); non_pain1 = np.array(nonpain)
pain1 = pain1[np.isnan(pain1)==0]; non_pain1 = non_pain1[np.isnan(non_pain1)==0]
maxvalue = np.max([np.concatenate((pain1, non_pain1))])
print('maxvalue', maxvalue)

xaxis = []; yaxis = []
for thr in np.arange(0,maxvalue,maxvalue/1000):
    TP = np.sum(pain1 >= thr)
    FN = np.sum(pain1 < thr)
    FP = np.sum(non_pain1 >= thr)
    TN = np.sum(non_pain1 < thr)
    msaccuracy = (TP + TN) / (TP + TN + FN + FP)
    
    xaxis.append(thr)
    yaxis.append(msaccuracy)
    
#    plt.plot(xaxis, yaxis)
    
optimized_thr = xaxis[np.argmax(yaxis)]
print('optimized_thr', optimized_thr)

Z = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        Z[SE,se] = SE*100 + se

painGroup = highGroup + midleGroup + ketoGroup; painGroup = np.array(painGroup)
nonpainGroup = salineGroup + lidocainGroup

FNlist = Z[painGroup,1][np.where(targetMatrix[painGroup,1] < optimized_thr)[0]]

FPlist_ix = np.where(np.concatenate((targetMatrix[painGroup,0], targetMatrix[painGroup,2], targetMatrix[nonpainGroup,:].flatten())) > optimized_thr)[0]
FPlist = np.concatenate((Z[painGroup,0], Z[painGroup,2], Z[nonpainGroup,:].flatten()))[FPlist_ix]

for i in FNlist:
    SE = int(i // 100)
    se = int(i % 100)
    print(SE, se, 'False Negative', targetMatrix[SE,se])

print('____________________________________________')

for i in FPlist:
    SE = int(i // 100)
    se = int(i % 100)
    print(SE, se, 'False Positive', targetMatrix[SE,se])
    
# In[] 움직임 noise 문제 해결하기 - new

# 랜덤한 갯수로 nonpain을 뽑아서 평균 movement와, 그 그룹만 썻을때 accuracy를 2d plot
# 서로 상관관계가 있는지.
    
import random

painGroup = highGroup + midleGroup + ketoGroup + capsaicinGroup
nonpainGroup = salineGroup + lidocainGroup

pain_movement, nonpain_within_movement, nonpain_between_movement = pain_nonpain_sepreate(movement, painGroup, nonpainGroup)
nonpain_movement = np.concatenate((nonpain_within_movement, nonpain_between_movement), axis=0)


axiss = []; [axiss.append([]) for i in range(2)]
totalNum = nonpain_movement.shape[0]
epochs = 5000
for epoch in range(epochs):
    if epoch % int(epochs/10) == 1:
        print(epoch, '/', epochs)
    random_N = random.randrange(totalNum)
    ixlist = random.sample(list(range(totalNum)), random_N)
    
    accuracy, _ = accuracy_cal(pain_movement, nonpain_movement[ixlist], False)
    axiss[0].append(np.mean(nonpain_movement[ixlist], axis=0))
    axiss[1].append(accuracy)

for d in range(2):
    axiss[d] = np.array(axiss[d])
    axiss[d] = axiss[d][np.isnan(axiss[d])==0];
    
maxvalue = np.max([np.concatenate((pain1, non_pain1))])
    
m, b = mslinear_regression(axiss[0], axiss[1])

plt.figure()
plt.scatter(axiss[0], axiss[1], s = 0.4)
plt.xlabel('movement ratio mean')
plt.ylabel('accuracy by movement as feature')

xaxis = np.arange(0,0.5,0.5/100)
plt.plot(xaxis, xaxis*m + b, c = 'orange')
plt.xlim([0,0.5]), plt.ylim([0.5,1])

# 대조군

pain_target, nonpain_within_target, nonpain_between_target = pain_nonpain_sepreate(target, painGroup, nonpainGroup)
nonpain_target = np.concatenate((nonpain_within_target, nonpain_between_target), axis=0)


axiss = []; [axiss.append([]) for i in range(2)]
totalNum = nonpain_movement.shape[0]

#epochs = 50000
for epoch in range(epochs):
    if epoch % int(epochs/10) == 1:
        print(epoch, '/', epochs)
        
    random_N = random.randrange(totalNum)
    ixlist = random.sample(list(range(totalNum)), random_N)
    
    accuracy, _ = accuracy_cal(pain_target, nonpain_target[ixlist], False)
    axiss[0].append(np.mean(nonpain_movement[ixlist], axis=0))
    axiss[1].append(accuracy)

for d in range(2):
    axiss[d] = np.array(axiss[d])
    axiss[d] = axiss[d][np.isnan(axiss[d])==0];
    
maxvalue = np.max([np.concatenate((pain1, non_pain1))])
    
m, b = mslinear_regression(axiss[0], axiss[1])

plt.figure()
plt.scatter(axiss[0], axiss[1], s = 0.4)
plt.xlabel('movement ratio mean')
plt.ylabel('accuracy by movement as feature')

xaxis = np.arange(0,0.5,0.5/100)
plt.plot(xaxis, xaxis*m + b, c = 'orange')
plt.xlim([0,0.5]), plt.ylim([0.5,1])

# In[] to Prism

def msGrouping_nonexclude(msmatrix): # base 예외처리 없음, goruping된 sample만 뽑힘
    target = np.array(msmatrix)
    
    df3 = pd.DataFrame(target[highGroup]) 
    df3 = pd.concat([df3, pd.DataFrame(target[midleGroup]), pd.DataFrame(target[lowGroup]), \
                     pd.DataFrame(target[restrictionGroup]), pd.DataFrame(target[salineGroup]), \
                     pd.DataFrame(target[ketoGroup]), pd.DataFrame(target[lidocainGroup]), \
                     pd.DataFrame(target[capsaicinGroup])], ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

Aprism = msGrouping_nonexclude(target)















