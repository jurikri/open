# msbak, 2019. 09. 02.
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
try: import pickle5 as pickle
except: import pickle
import os
import random
from scipy import stats
import scipy

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

shortlist = []; longlist = []
def msGrouping_nonexclude(msdata): 
    df3 = pd.concat([pd.DataFrame(msdata[salineGroup,0:4]) \
                                  ,pd.DataFrame(msdata[highGroup + highGroup2,0:4]) \
                                  ,pd.DataFrame(msdata[midleGroup,0:4]) \
                                  ,pd.DataFrame(msdata[ketoGroup,0:4]) \
                                  ,pd.DataFrame(msdata[lidocainGroup,0:4])] \
                                  ,ignore_index=True, axis=1)
    
    df3 = np.array(df3)
    return df3

def msGrouping_pslOnly(psl): # psl만 처리
    psldata = np.array(psl)
    
    df3 = pd.DataFrame(psldata[shamGroup,0:3]) 
    df3 = pd.concat([df3, pd.DataFrame(psldata[pslGroup,0:3]), \
                     pd.DataFrame(psldata[adenosineGroup,0:3])], ignore_index=True, axis = 1)
        
    df3 = np.array(df3)
    
    return df3

def msGrouping_yohimbine(msdata):
    msdata = np.array(msdata)
    df3 = np.array(pd.concat([pd.DataFrame(msdata[salineGroup,1:4]) \
                                  ,pd.DataFrame(msdata[yohimbineGroup,1:4])] \
                                  ,ignore_index=True, axis=1))
    df3 = np.array(df3)
    return df3


try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
    gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
except:
    try:
        savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)
#

# var import
with open('D:\\mscore\\syncbackup\\google_syn\\mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
# bahavss = msdata_load['bahavss']
# behavss2 = msdata_load['behavss2']
msGroup = msdata_load['msGroup']
msdir = msdata_load['msdir']
signalss = msdata_load['signalss']
    
highGroup = msGroup['highGroup']
highGroup2 = msGroup['highGroup2']
highGroup3 = msGroup['highGroup3']
midleGroup = msGroup['midleGroup']
lowGroup = msGroup['lowGroup']
salineGroup = msGroup['salineGroup']
restrictionGroup = msGroup['restrictionGroup']
ketoGroup = msGroup['ketoGroup']
lidocainGroup = msGroup['lidocaineGroup']
capsaicinGroup = msGroup['capsaicinGroup']
yohimbineGroup = msGroup['yohimbineGroup']
pslGroup = msGroup['pslGroup']
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
PSLscsaline = msGroup['PSLscsaline']
glucoseGroup = msGroup['glucoseGroup']

msset = msGroup['msset']; msset = np.array(msset)
msset2 = msGroup['msset2']; msset2 = np.array(msset2)
del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

# se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup \
# + itSalineGroup + itClonidineGroup # for test only

# pslset = pslGroup + shamGroup + adenosineGroup + itSalineGroup + itClonidineGroup
fset  = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
baseonly = lowGroup + lidocainGroup + restrictionGroup
gababase = list(range(164,169)) + list(range(172,176)) + list(range(177,183)) + [226,227]

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]
totaldataset = grouped_total_list

msdata_load['msset_total'] = msset_total
msdata_load['fset'] = fset
msdata_load['baseonly'] = baseonly
msdata_load['gababase'] = gababase
msdata_load['totaldataset'] = totaldataset

pslset = pslGroup + shamGroup + adenosineGroup

# 제외 roi 적용 (mouse 통합)
roi_del_ix = msdata_load['roi_del_ix_save']
skiplist = []
for SE in range(N):
    if SE in skiplist: continue
    setnum = [SE]
    if SE in msset_total.flatten():
        row = np.where(msset_total==SE)[0][0]
        setnum = msset_total[row,:]
    skiplist += list(setnum)
    
    tmp = []
    for SE2 in setnum:
        tmp += list(roi_del_ix[SE2])
    tmp = list(set(tmp))  
    
    for SE2 in setnum:
        for se in range(5):  
            tix = list(range(signalss[SE2][se].shape[1]))
            ix = list(set(tix) - set(tmp))
            signalss[SE2][se] = signalss[SE2][se][:,ix]

def msGrouping_pain_vs_itch(msdata): # psl만 처리
    msdata = np.array(msdata)
    
    df3 = pd.DataFrame(msdata[highGroup,1]) 
    df3 = pd.concat([df3, pd.DataFrame(msdata[chloroquineGroup,1:3])], ignore_index=True, axis = 1)
    df3 = np.array(df3)
    
    return df3

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

#Aprism_foramlin_pain = dict_gen(model2, msset='formalin', figsw=False, figsw2=True)

def nanex(array1):
    array1 = np.array(array1)
    array1 = array1[np.isnan(array1)==0]
    return array1

def msGrouping_base_vs_itch(msdata): # psl만 처리
    msdata = np.array(msdata)
    
    df3 = pd.DataFrame(msdata[salineGroup,:]) 
    df3 = pd.concat([df3, pd.DataFrame(msdata[chloroquineGroup,0:3])], ignore_index=True, axis = 1)
    df3 = np.array(df3)
    return df3

# 제외된 mouse 확인용, mouseGroup
mouseGroup = []
for i in list(msGroup.keys()):
    mouseGroup += msGroup[i]
print('현재 grouping된 mouse #...', len(set(mouseGroup)), '/', str(N))
      
# long, short separate
#msshort = 42; mslong = 97; 
bins = 10

t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
        # 개별 thr로 relu 적용되어있음. frame은 signal과 syn가 다름
                    
        # In[] model1 load
if False:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model1\\'               
    project_list = []

    project_list.append(['model1_roiroi_formalin_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model1 = np.nanmean(testsw3_mean, axis=2)
    

#%% Formalin CV-- model2 (AI)
 
if True:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\kerasdata\\model2\\'               
    project_list = []

    project_list.append(['foramlin_only_1', 100, None])
    project_list.append(['foramlin_only_2', 200, None]) 
    project_list.append(['foramlin_only_3', 300, None]) 
    project_list.append(['foramlin_only_4', 400, None])
    project_list.append(['foramlin_only_5', 500, None]) 
  
    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model2 = np.nanmean(testsw3_mean, axis=2)
    
    import paindecoder_grouping
    ins = paindecoder_grouping.grouping(msdata_load)
    _, _ = ins.dict_gen(target = model2, msset='psl', legendsw=True, figsw=True)
  
#%% In Formalin CV-- model2 - mean (AA)
  
if False:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model2\\'               
    project_list = []

    project_list.append(['foramlin_only_1', 100, None])
    project_list.append(['foramlin_only_2', 200, None]) 
    project_list.append(['foramlin_only_3', 300, None]) 
    project_list.append(['foramlin_only_4', 400, None])
    project_list.append(['foramlin_only_5', 500, None]) 
  
    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + 'mean.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model2_mean = np.nanmean(testsw3_mean, axis=2)
    
# In Formalin CV-- model2_roi_roi (II)
 
if False:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model2-roi\\'               
    project_list = []

    project_list.append(['model2_roitraining_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model2roi_roi = np.nanmean(testsw3_mean, axis=2)
    
# In Formalin CV-- model2_roi_eman (IA)
 
if False:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model2-roi\\'               
    project_list = []

    project_list.append(['model2_roitraining_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + 'mean.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model2roi_mean = np.nanmean(testsw3_mean, axis=2)  
                
    # In[] PSL용 load - model3
if True:
    t = 10
    testsw3_mean = np.zeros((N,5,t)); testsw3_mean[:] = np.nan         
    for i in range(t):
        path1 = 'D:\\mscore\\syncbackup\\google_syn\kerasdata\\model3\\'
        path2 = 'fset + baseonly + CFAgroup + capsaicinGroup_0.69_0415_t' + str(i) + '.h5'
        path3 = path1+ path2
        
        if os.path.isfile(path3):
            with open(path3, 'rb') as f:  # Python 3: open(..., 'rb')
                testsw3 = pickle.load(f)
                testsw3_mean[:testsw3.shape[0],:,i] = testsw3
    model3 = np.nanmean(testsw3_mean, axis=2)
    
    import paindecoder_grouping
    ins = paindecoder_grouping.grouping(msdata_load)
    _, _ = ins.dict_gen(target = model3, msset='psl', legendsw=True, figsw=True)
    



    
        # In[] model4 load
if False:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\model4\\'               
    project_list = []

    project_list.append(['model4_roiroi_formalin_cap_cfa_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model4 = np.nanmean(testsw3_mean, axis=2)
    
        # In[] model5 load
if True:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\result\\'               
    project_list = []

    project_list.append(['model5_1', 100, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw5_mean = np.zeros((N,5,len(model_name))); testsw5_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw5 = pickle.load(f)
                testsw5_mean[SE,:,ix] = testsw5[SE,:]
    model5 = np.nanmean(testsw5_mean, axis=2)

# In[] model_basic _ oxaliplatin
    
msdir2 = 'D:\\mscore\\code_lab\\'
import sys; sys.path.append(msdir2); 
import msFunction

if True:
    savepath = 'D:\\mscore\\syncbackup\\google_syn\\kerasdata\\'               
    project_list = ['20201020_basic_1', '20201020_basic_2', '20201020_basic_3', '20201020_basic_4', '20201020_basic_5']
    
    mssave = msFunction.msarray([N,5])
    for i in range(len(project_list)):
        loadpath = savepath + project_list[i] + '\\exp\\result_matrix.h5'
        if os.path.isfile(loadpath):
          with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
                resultmatrix = pickle.load(f) # [SE][se][epoch] # epoch 평균내어 사용
        for SE in range(N):
            for se in range(5):
                mssave[SE][se].append(np.nanmean(resultmatrix[SE][se]))

model_basic = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        model_basic[SE,se] = np.nanmean(mssave[SE][se])

import paindecoder_grouping
target = np.array(model_basic)
aprism_oxali = paindecoder_grouping.dict_gen(target, msset='oxali', legendsw=True, figsw=True, figsw2=False)
# In[] raw test (구버전) - with model3
    
if True:
    thr = 0.5
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\result\\0331_CFA_selection\\exp_raw\\'               
    project_list = range(10)

    model3_mean_overtime = []; [model3_mean_overtime.append([]) for u in range(N)]
    for i in range(N):
        [model3_mean_overtime[i].append([]) for u in range(5)]
        
    for SE in range(N):
        for se in range(5):
            matrixsave=[]
            for i in range(len(project_list)):
                loadpath_mean = savepath + 'PSL_result_' + str(SE) + '_' + str(project_list[i]) + '.pickle' 
                if os.path.isfile(loadpath_mean):
                    with open(loadpath_mean, 'rb') as f:  # Python 3: open(..., 'rb')
                        PSL_result_save = pickle.load(f)
                        
                    binum = len(PSL_result_save[SE][se])
                    if binum == 0: continue          
                    ROInum = len(PSL_result_save[SE][se][0])
                    
                    binROI_matrix = np.zeros((ROInum, binum)); binROI_matrix[:] = np.nan
                    
                    for col in range(binum):
                        for row in range(ROInum):
                            binROI_matrix[row,col] = PSL_result_save[SE][se][col][row][0][1]
                    matrixsave.append(np.array(binROI_matrix))
            matrixsave = np.array(matrixsave)
                            
            model3_mean_overtime[SE][se] = np.mean(matrixsave, axis=0)
            print(SE, se, model3_mean_overtime[SE][se].shape)

                        
    # heatmatplot
    
    nonpains2 = []
    nonpains2.append(np.mean(model3_mean_overtime[167][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[168][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[173][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[173][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[75][0] > thr, axis=0)[:55]) # 5
    nonpains2.append(np.mean(model3_mean_overtime[76][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[85][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[87][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[88][0] > thr, axis=0)[:55])
    nonpains2.append(np.mean(model3_mean_overtime[94][1] > thr, axis=0)[:55]) # 10
    nonpains2 = np.array(nonpains2)
    
    pains = []
    pains.append(np.mean(model3_mean_overtime[71][1] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[72][2] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[73][1] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[73][2] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[75][1] > thr, axis=0)) # 5
    pains.append(np.mean(model3_mean_overtime[76][1] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[85][2] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[87][1] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[88][1] > thr, axis=0))
    pains.append(np.mean(model3_mean_overtime[94][1] > thr, axis=0)) # 10
    pains = np.array(pains)[:,:55]
    
    nonpains = []
    nonpains.append(np.mean(model3_mean_overtime[167][2] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[168][2] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[172][2] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[173][2] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[174][2] > thr, axis=0)) # 5
    nonpains.append(np.mean(model3_mean_overtime[175][2] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[177][1] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[178][1] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[179][1] > thr, axis=0))
    nonpains.append(np.mean(model3_mean_overtime[180][1] > thr, axis=0)) # 10
    nonpains = np.array(nonpains)[:,:55]
    
    inter = np.ones((1,55))
    msplot = np.concatenate((nonpains2, inter, pains, inter, nonpains), axis=0) 
    
    plt.imshow(msplot, cmap='hot')
    
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    
    plt.colorbar()
    
    #%%
    plt.figure(figsize=(6,1))
    plt.plot(np.mean(model3_mean_overtime[167][0] > thr, axis=0)[:55])
    savepath = 'C:\\SynologyDrive\\worik in progress\\20220114 - EMM revision\\figsave.png'
    plt.savefig(savepath, dpi=500)
    
    

# In[] label 재정렬 movement 
t4 = np.zeros((N,5)); movement = np.zeros((N,5))
for SE in range(N):
    for se in range(5):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not
    
target = np.array(movement)
for SE in range(N):
    if SE in [141,142,143]:
        target[SE,1:3] = target[SE,3:5] 
        target[SE,1:3] = np.nan
        
    if SE in [146,149,158,159]:
        target[SE,3:] = np.nan

movement = target     
movement_filter = np.array(movement)
        
# In[]
# target = np.array(model3); fsw=True
def msacc(class0, class1, mslabel='None', figsw=False, fontsz=15, fontloc="lower right", legendsw=True, figsw2=False):
    pos_label = 1; roc_auc = -np.inf; fig = None
    while roc_auc < 0.5:
        class0 = np.array(class0); class1 = np.array(class1)
        class0 = class0[np.isnan(class0)==0]; class1 = class1[np.isnan(class1)==0]
        
        anstable = list(np.ones(class1.shape[0])) + list(np.zeros(class0.shape[0]))
        predictValue = np.array(list(class1)+list(class0)); predictAns = np.array(anstable)
        #            
        fpr, tpr, thresholds = metrics.roc_curve(predictAns, predictValue, pos_label=pos_label)
        
        maxix = np.argmax((1-fpr) * tpr)
        specificity = 1-fpr[maxix]; sensitivity = tpr[maxix]
        accuracy = ((class1.shape[0] * sensitivity) + (class0.shape[0]  * specificity)) / (class1.shape[0] + class0.shape[0])
        roc_auc = metrics.auc(fpr,tpr)
        
        if roc_auc < 0.5:
            pos_label = 0
            
    if figsw:
        sz = 0.9
        fig = plt.figure(1, figsize=(7*sz, 5*sz))
        lw = 2
        plt.plot(fpr, tpr, lw=lw, label = (mslabel + ' ' + str(round(roc_auc,2))), alpha=1)
        plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        if legendsw:
            plt.legend(loc="lower right", prop={'size': fontsz})
            
    if figsw2:
        sz = 0.5
        fig = plt.figure(1, figsize=(5*sz, 5*sz))
        lw = 2
        plt.plot(fpr, tpr, lw=lw, label = (mslabel + ' ' + str(round(roc_auc,2))), alpha=1)
        plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        print('roc_auc', roc_auc)
        if legendsw:
            plt.legend(loc="lower right", prop={'size': fontsz})
            
    return roc_auc, accuracy, fig


target = model2
def dict_gen(target, msset=None, legendsw=None, figsw=False, figsw2=False):
    if msset is None:
        print('set mssset')
        pass
    
    target = np.array(target)
#    print(target.shape, movement_filter.shape)
#    
    if msset in ['psl'] and False:
        print('movement > 0.5 filter')
        ix = np.where(movement[:,0] > 0.5)[0]
        print('mov out', ix)
        target[ix,:] = np.nan
        
    # subset 평균처리        
    subset_mean = np.zeros((N,5)); subset_mean[:] = np.nan
    for SE in range(N):
        if SE in np.array(msset_total)[:,0]:
            settmp = np.array(msset_total)[np.where(np.array(msset_total)[:,0]==SE)[0][0],:]
            subset_mean[SE,:] = np.nanmean(target[settmp,:],axis=0)
    #        print('set averaging', settmp)
        elif SE not in np.array(msset_total).flatten(): 
            subset_mean[SE,:] = target[SE,:]
            
    for SE in range(N):
        if SE in [141,142,143]:
            subset_mean[SE,3:5] = subset_mean[SE,1:3]
            subset_mean[SE,1:3] = np.nan
    
    # grouping
    high0 = nanex(subset_mean[highGroup+highGroup2,0])
    high1 = nanex(subset_mean[highGroup+highGroup2,1])
    
    midle0 = nanex(subset_mean[midleGroup,0])
    midle1 = nanex(subset_mean[midleGroup,1])
    
    keto0 = nanex(subset_mean[ketoGroup,0])
    keto1 = nanex(subset_mean[ketoGroup,1])
    
    lido0 = nanex(subset_mean[lidocainGroup,0])
    lido1 = nanex(subset_mean[lidocainGroup,1])
    
    saline0 = nanex(subset_mean[salineGroup,0])
    saline1 = nanex(subset_mean[salineGroup,1])
    
    cap0 = nanex(subset_mean[capsaicinGroup,0])
    cap1 = nanex(subset_mean[capsaicinGroup,1])
    
    CFA0 = nanex(subset_mean[CFAgroup,0])
    CFA1 = nanex(subset_mean[CFAgroup,1])
    CFA2 = nanex(subset_mean[CFAgroup,2])
    
    sham0 = nanex(subset_mean[shamGroup,0])
    sham1 = nanex(subset_mean[shamGroup,1])
    sham2 = nanex(subset_mean[shamGroup,2])
           
    psl0 = nanex(subset_mean[pslGroup,0])
    psl1 = nanex(subset_mean[pslGroup,1])
    psl2 = nanex(subset_mean[pslGroup,2])
    
    
#    ipsaline_pslGroup
    
    ipsaline0 = nanex(subset_mean[ipsaline_pslGroup,0])
    ipsaline1 = nanex(subset_mean[ipsaline_pslGroup,1])
    ipsaline2 = nanex(subset_mean[ipsaline_pslGroup,2])
    ipsaline3 = nanex(subset_mean[ipsaline_pslGroup,3])
    ipsaline4 = nanex(subset_mean[ipsaline_pslGroup,4])
    
    ipclonidine0 = nanex(subset_mean[ipclonidineGroup,0])
    ipclonidine1 = nanex(subset_mean[ipclonidineGroup,1])
    ipclonidine2 = nanex(subset_mean[ipclonidineGroup,2])
    ipclonidine3 = nanex(subset_mean[ipclonidineGroup,3])
    ipclonidine4 = nanex(subset_mean[ipclonidineGroup,4])

    
    gaba120_0 = nanex(np.mean(subset_mean[[164,165,166],0:2], axis=1))
    gaba120_1 = nanex(subset_mean[[177,179],2]) # psl_d3, GB/VX_i.p._120m 
    gaba120_2 = nanex(np.mean(subset_mean[[164,165,166],2:4], axis=1)) # psl_d10, GB/VX_i.p._120m 
    
    
    gaba30_0 = nanex(subset_mean[[167,168,172,174,177,179,182], 0])
    gaba30_0 = np.concatenate((gaba30_0, [np.nanmean(subset_mean[181,0:2])]), axis=0)
    add = np.mean(subset_mean[[185,186],0:2], axis=1)
    gaba30_0 = np.concatenate((gaba30_0, add), axis=0)     
    
    gaba30_1 = nanex(subset_mean[[167,168], 1]) # GB/VX (d3)
    gaba30_1 = np.concatenate((gaba30_1, nanex(subset_mean[[172,174], 2])), axis=0)  # GB/VX (d3)
    gaba30_1 = np.concatenate((gaba30_1, nanex(subset_mean[[177,179], 1])), axis=0)  # GB/VX (d3)
    add = np.nanmean(subset_mean[181, [2,3]])
    gaba30_1 = np.concatenate((gaba30_1, [add]), axis=0)
    add = np.nanmean(subset_mean[182, [1,2]])
    gaba30_1 = np.concatenate((gaba30_1, [add]), axis=0)
    
    
    gaba30_2 = nanex(subset_mean[[167,168], 2]) # lidocaine (d2)
    gaba30_2 = np.concatenate((gaba30_2, nanex(subset_mean[[172,174], 3])), axis=0)  # lidocaine (d2)
    
    gaba30_3 = nanex(np.mean(subset_mean[[169,170,171],0:2], axis=1)) # GB/VX (d10~)
    gaba30_3 = np.concatenate((gaba30_3, [np.mean(subset_mean[176,0:2])]), axis=0) 
    add = np.mean(subset_mean[[183,184],0:2], axis=1)
    gaba30_3 = np.concatenate((gaba30_3, add), axis=0)
    add = np.mean(subset_mean[[185,186],2:4], axis=1)
    gaba30_3 = np.concatenate((gaba30_3, add), axis=0) 
    
    gaba30_4 = nanex(np.mean(subset_mean[[169,170,171],2:4], axis=1)) # lidocaine (d10~)
    
    scsalcine = nanex(subset_mean[[172,174], 1])
    
    itsaline0 = nanex(subset_mean[itSalineGroup,0])
    itsaline1 = nanex(subset_mean[itSalineGroup,1])
    itsaline2 = nanex(subset_mean[itSalineGroup,2])
    
    itclonidine0 = nanex(subset_mean[itClonidineGroup,0])
    itclonidine1 = nanex(subset_mean[itClonidineGroup,1])
    itclonidine2 = nanex(subset_mean[itClonidineGroup,2])
    
    #
    oxali0 = nanex(subset_mean[oxaliGroup,0])
    oxali1 = nanex(subset_mean[oxaliGroup,1])
    oxali2 = nanex(subset_mean[[list(range(192,200)) + [202,203,220,221]],2]) # for pain
    tmp1 = nanex(subset_mean[[188,189,200,201],2]) # for nonpain
    tmp2 = nanex(subset_mean[list(range(192,198)) + [202,203,220,221],3])
    oxali3 = nanex(np.concatenate((tmp1, tmp2), axis=0))
    
    glucose0 = nanex(np.concatenate((subset_mean[[204,205],0],subset_mean[list(range(206,216)),0]), axis=0))
    glucose1 = nanex(np.concatenate((subset_mean[[204,205],1],subset_mean[list(range(206,216)),1]), axis=0))
    glucose2 = nanex(subset_mean[list(range(206,216)),2])
    glucose3 = nanex(np.concatenate((subset_mean[[204,205],2],subset_mean[list(range(206,216)),3]), axis=0))
    
    scSaline0 = nanex(subset_mean[list(range(216,220)) + [204,205],0])
    scSaline1 = nanex(subset_mean[list(range(216,220)) + [204,205],1:3])
    
    if msset == 'formalin':
        name='formalin'
        pain = np.concatenate((high1, midle1), axis=0)
        nonpain = np.concatenate((high0, midle0, saline0, saline1), axis=0)
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        base_merge = np.concatenate((saline0, saline1), axis=0)
        Aprism = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(high0), pd.DataFrame(high1) \
                                       , pd.DataFrame(midle0), pd.DataFrame(midle1) \
                                       , pd.DataFrame(keto0), pd.DataFrame(keto1)
                                       , pd.DataFrame(lido0), pd.DataFrame(lido1)] \
                                       , ignore_index=True, axis=1)
        
    elif msset == 'capcfa':
        base_merge = np.concatenate((saline0, saline1), axis=0)
        name='capcfa'
        pain = np.concatenate((cap1, CFA1, CFA2), axis=0)
        nonpain = base_merge
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        Aprism = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(cap1), pd.DataFrame(CFA1) \
                                       , pd.DataFrame(CFA2)], ignore_index=True, axis=1)
            
    elif msset == 'cap':
        base_merge = np.concatenate((saline0, saline1), axis=0)
        name='cap'
        pain = cap1
        nonpain = base_merge
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        Aprism = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(cap1)], ignore_index=True, axis=1)
            
    elif msset == 'cfa':
        base_merge = CFA0
        name='cfa'
        pain = np.concatenate((CFA1, CFA2), axis=0)
        nonpain = base_merge
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        Aprism = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(CFA1), pd.DataFrame(CFA2)], ignore_index=True, axis=1)
        
    elif msset == 'psl':
        name='psl'
        pain = np.concatenate((psl1, psl2), axis=0)
        nonpain = np.concatenate((psl0, sham0, sham1, sham2), axis=0)
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        base_merge = np.concatenate((sham0, psl0, ipsaline0, ipclonidine0), axis=0)
        psl3_merge = np.concatenate((psl1, ipsaline1, ipclonidine1), axis=0)
        psl10_merge = np.concatenate((psl2, ipsaline3, ipclonidine3), axis=0)

        Aprism = [[], [], [], []]
        
        # base, sham, psl
        Aprism[0] = pd.concat([pd.DataFrame(sham0), pd.DataFrame(psl0), pd.DataFrame(sham1), pd.DataFrame(psl1), \
                            pd.DataFrame(sham2), pd.DataFrame(psl2)], \
                            ignore_index=True, axis=1)
        
        # psl+ipsaline, psl+ipclonidine, psl+ipGB/VX
        base_merge = np.concatenate((ipsaline0, ipclonidine0, gaba120_0, gaba30_0), axis=0)
        Aprism[1] = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(ipsaline2), pd.DataFrame(ipsaline4), pd.DataFrame(ipclonidine2), \
                    pd.DataFrame(ipclonidine4), pd.DataFrame(gaba120_1), pd.DataFrame(gaba30_1), pd.DataFrame(gaba30_3), pd.DataFrame(gaba30_2), \
                    pd.DataFrame(gaba30_4), pd.DataFrame(scsalcine)], \
                    ignore_index=True, axis=1)
        
        # psl+itsaline, psl+itclonidine
        base_merge = np.concatenate((itsaline0, itclonidine0), axis=0)
        Aprism[2] = pd.concat([pd.DataFrame(base_merge), pd.DataFrame(itsaline1), pd.DataFrame(itclonidine1), \
              pd.DataFrame(itsaline2), pd.DataFrame(itclonidine2)],ignore_index=True, axis=1)
        
        # base, sham, psl(psl+ipsaline), psl+GB/VX
        base_merge = np.concatenate((sham0, psl0, ipsaline0, gaba30_0, gaba120_0), axis=0)
  
        Aprism[3] = pd.concat([pd.DataFrame(base_merge), \
              pd.DataFrame(sham1), pd.DataFrame(psl3_merge), pd.DataFrame(gaba30_1), pd.DataFrame(gaba120_1), \
              pd.DataFrame(sham2), pd.DataFrame(psl10_merge), pd.DataFrame(gaba30_3), pd.DataFrame(gaba120_2)],ignore_index=True, axis=1)
        
    elif msset == 'oxali':
        name='oxali'
        pain = np.concatenate((oxali1, oxali2), axis=0)
        nonpain = np.concatenate((oxali0, oxali3, glucose0, glucose1, glucose2, glucose3), axis=0)
        
        
        # 3일차 oxali vs 3일차 vehicle
        pain = np.concatenate((oxali1, []), axis=0)
        nonpain = np.concatenate((glucose1, []), axis=0)
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        
        Aprism = pd.concat([pd.DataFrame(glucose0), pd.DataFrame(glucose1), pd.DataFrame(glucose2), pd.DataFrame(glucose3), \
                            pd.DataFrame(oxali0), pd.DataFrame(oxali1), pd.DataFrame(oxali2), pd.DataFrame(oxali3)], ignore_index=True, axis=1)
            
    elif msset == 'scSaline':
        name='scSaline'
        pain = np.concatenate((scSaline1, []), axis=0)
        nonpain = np.concatenate((scSaline0, []), axis=0)
        roc_auc, _, _ = msacc(nonpain, pain, mslabel= name + ', AUC:', figsw=figsw, figsw2=figsw2, legendsw=legendsw)
        
        Aprism = pd.concat([pd.DataFrame(scSaline0), pd.DataFrame(scSaline1)], ignore_index=True, axis=1)
             
    return Aprism

# In[]

legendsw = False; figsw2 = True

Aprism_capcfa_pain = dict_gen(model2, msset='cap', legendsw=legendsw, figsw2=figsw2)
Aprism_capcfa_pain = dict_gen(model2, msset='cfa', legendsw=legendsw, figsw2=figsw2)
Aprism_capcfa_pain = dict_gen(model_basic, msset='oxali', legendsw=legendsw, figsw2=figsw2)

savepath2 = 'D:\\mscore\\syncbackup\\google_syn\\prismdata\\'
plt.savefig(savepath2 + 'roc_cap_cfa_oxali.png', dpi=1000)#


Aprism_capcfa_pain = dict_gen(model3, msset='psl', legendsw=legendsw, figsw2=figsw2)

savepath2 = 'D:\\mscore\\syncbackup\\google_syn\\prismdata\\'
plt.savefig(savepath2 + 'roc_psl.png', dpi=1000)#

#model1_dict = dict_gen(model1)
#model3_dict = dict_gen(model3)
#model4_dict = dict_gen(model4)

#fsw=True
#model2_dict = dict_gen(model2)
#model2mean_dict = dict_gen(model2_mean)
#model2roi_roi_dict = dict_gen(model2roi_roi)
#model2roi_mean_dict = dict_gen(model2roi_mean)

# 이름결정
legendsw = True
Aprism_capcfa_pain = dict_gen(model2, msset='cap', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'cap_aionly', dpi=1000)#

Aprism_capcfa_pain = dict_gen(model2, msset='cfa', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'cfa_aionly', dpi=1000)#

legendsw = True
Aprism_capcfa_pain = dict_gen(model2_mean, msset='cap', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2_mean, msset='cfa', legendsw=legendsw)


# capsaicin 4
legendsw = True
legendsw = False
Aprism_capcfa_pain = dict_gen(model2, msset='cap', legendsw=legendsw)
vAprism_capcfa_pain = dict_gen(model2_mean, msset='cap', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_roi, msset='cap', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_mean, msset='cap', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'cap_4ways', dpi=1000)#

# cfa 4
legendsw = True
legendsw = False
Aprism_capcfa_pain = dict_gen(model2, msset='cfa', legendsw=legendsw)
vAprism_capcfa_pain = dict_gen(model2_mean, msset='cfa', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_roi, msset='cfa', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_mean, msset='cfa', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'cfa_4ways', dpi=1000)#

# psl 4
legendsw = True
legendsw = False
Aprism_capcfa_pain = dict_gen(model2, msset='psl', legendsw=legendsw)
vAprism_capcfa_pain = dict_gen(model2_mean, msset='psl', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_roi, msset='psl', legendsw=legendsw)
Aprism_capcfa_pain = dict_gen(model2roi_mean, msset='psl', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'psl_4ways', dpi=1000)#

# Psl AI, model2, model3


#### formalin pain
legendsw = False
if False:
    Aprism_foramlin_pain = dict_gen(model2, msset='formalin', legendsw=legendsw, figsw2=True)
    savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
    plt.savefig(savepath2 + 'formalin_aionly', dpi=1000)#
    
    Aprism_foramlin_pain = dict_gen(model2, msset='cap', legendsw=legendsw, figsw2=True)
    savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
    plt.savefig(savepath2 + 'cap_aionly', dpi=1000)#
    
    Aprism_foramlin_pain = dict_gen(model2, msset='cfa', legendsw=legendsw, figsw2=True)
    savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
    plt.savefig(savepath2 + 'cfa_aionly', dpi=1000)#
    
    Aprism_foramlin_pain = dict_gen(model3, msset='psl', legendsw=legendsw, figsw2=True)
    savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
    plt.savefig(savepath2 + 'psl_aionly', dpi=1000)#    
    
    _ = dict_gen(model2_mean, msset='formalin', legendsw=legendsw, figsw2=True)
    _ = dict_gen(model2roi_roi, msset='formalin', legendsw=True)
    _ = dict_gen(model2roi_mean, msset='formalin', legendsw=True)
    
    # formalin movement
    Aprism_foramlin_movement = dict_gen(movement, msset='formalin', legendsw=True)
    
    # capcfa pain
    Aprism_capcfa_pain = dict_gen(model2, msset='capcfa', legendsw=legendsw)
    _ = dict_gen(model2_mean, msset='capcfa', legendsw=legendsw)
    _ = dict_gen(model2roi_roi, msset='capcfa', legendsw=legendsw)
    _ = dict_gen(model2roi_mean, msset='capcfa', legendsw=legendsw)
#    _ = dict_gen(model5, msset='capcfa', legendsw=legendsw)
    savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
    plt.savefig(savepath2 + 'capcfa_roc', dpi=1000)#
    
    # capcfa movement
    Aprism_capcfa_movement = dict_gen(movement, msset='capcfa', legendsw=True)

legendsw = False
legendsw = True
Aprism_psl_pain = dict_gen(model2, msset='psl', legendsw=legendsw)
_ = dict_gen(model2_mean, msset='psl', legendsw=legendsw)
_ = dict_gen(model2roi_roi, msset='psl', legendsw=legendsw)
_ = dict_gen(model2roi_mean, msset='psl', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'psl_roc', dpi=1000)#

Aprism_psl_pain = dict_gen(model3, msset='psl', legendsw=legendsw)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'psl_roc', dpi=1000)#

Aprism_psl_pain = dict_gen(t4, msset='psl', legendsw=legendsw)
Aprism_psl_pain_ratio = dict_gen(engram_activity, msset='psl', legendsw=legendsw)

_ = dict_gen(model2, msset='psl', legendsw=legendsw)
_ = dict_gen(model5, msset='psl', legendsw=legendsw)
Aprism_psl_pain3 = dict_gen(model3, msset='psl', legendsw=legendsw)
plt.savefig(savepath2 + 'psl_roc_models', dpi=1000)#

# psl movement
legendsw = True
Aprism_psl_movement = dict_gen(movement, msset='capcfa', legendsw=legendsw)
#Aprism_psl_movement = dict_gen(movement, msset='psl', legendsw=True)
savepath2 = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\psl_visualization\\'
plt.savefig(savepath2 + 'capcfa_movement_roc', dpi=1000)#

legendsw = True
_ = dict_gen(model5, msset='psl', legendsw=legendsw)
_ = dict_gen(model5, msset='capcfa', legendsw=True)

Aprism_oxli_t4 = dict_gen(t4, msset='oxali', legendsw=True)
Aprism_oxli_movement = dict_gen(movement, msset='oxali', legendsw=True)
Aprism_oxli_semi_bRNN = dict_gen(model3, msset='oxali', legendsw=True)


# In[]
import os
os.sys.exit()

# In[]
# In[] cerebellum _ capsaicin
picklesavename = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\msGroup_ksh_bRNN.pickle'
with open(picklesavename, 'rb') as f:  # Python 3: open(..., 'wb')
    cerebellum_capsaicin = pickle.load(f)
    
pain = cerebellum_capsaicin[:7,1]
nonpain = np.concatenate((cerebellum_capsaicin[7:,:].flatten(), cerebellum_capsaicin[:7,0], cerebellum_capsaicin[:7,2]),axis=0)
roc_auc, _, _ = msacc(nonpain, pain, mslabel='AUC:', figsw2=True, legendsw=legendsw)
plt.savefig(savepath2 + 'cerebellum_ROC.png', dpi=1000)


# In[] itch vs non-itch
if True:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\result\\'    
    project_list = []

    project_list.append(['20200308_itch_vs_before', 111, None])
    project_list.append(['20200308_itch_vs_before2', 222, None])
    project_list.append(['20200308_itch_vs_before3', 333, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    for ix, p in enumerate(model_name):
        for SE in range(N):
            
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'testsw3_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                    testsw3 = pickle.load(f)
                testsw3_mean[SE,:,ix] = testsw3[SE,:]
    model_itch_vs_nonitch = np.nanmean(testsw3_mean, axis=2)
    
    fset2 = highGroup + midleGroup + highGroup2 
    Aprism_itch_late = model_itch_vs_nonitch[fset2,:]
    Aprism_itch_late2 = model2[fset2,:]
    
    Aprism_itch_late_yohimibine = model_itch_vs_nonitch[yohimbineGroup,:]
    Aprism_itch_late_yohimibine2 = model2[yohimbineGroup,:]
    
    
    pain = model_itch_vs_nonitch[chloroquineGroup,1]
    
    tmp1 = model_itch_vs_nonitch[chloroquineGroup,0]
    tmp2 = model_itch_vs_nonitch[salineGroup,1].flatten()
    nonpain = np.concatenate((tmp1, tmp2), axis=0)
    
#    nonpain = np.array(model_itch_vs_nonitch[chloroquineGroup,0])

    roc_auc, _, _ = msacc(nonpain, pain, mslabel='AUC:', figsw2=True, legendsw=legendsw)
    
    # saline inter를 사용할이유가?
    # saline을 모두 빼고, ROC 및 본 그래프 수정


    
# In[] itch vs pain
if False:
    savepath = 'D:\\mscore\\syncbackup\\save\\tensorData\\result\\'    
    project_list = []

    project_list.append(['20200302_painitch_1', 100, None])
    project_list.append(['20200302_painitch_2', 200, None]) 
    project_list.append(['20200302_painitch_3', 300, None]) # acc_thr 증가
    project_list.append(['20200302_painitch_4', 400, None])

    model_name2 = project_list 
                 
    model_name = np.array(model_name2)
    testsw3_mean = np.zeros((N,5,len(model_name))); testsw3_mean[:] = np.nan
    

    ## 
    for ix, p in enumerate(model_name):
        for SE in range(N):          
            loadpath5 = savepath + p[0] + '\\exp_raw\\' + 'PSL_result_' + str(SE) + '.pickle'
            if os.path.isfile(loadpath5):
                sessionNum = 5
    
                for se in range(sessionNum):
                    with open(loadpath5, 'rb') as f:  # Python 3: open(..., 'rb')
                        PSL_result_save = pickle.load(f)
                    PSL_result_save2 = np.array(PSL_result_save[SE][se]) # [BINS][ROI][bins] # BINS , full length 넘어갈때, # bins는 full length 안에서
                    if type(PSL_result_save2) == np.ndarray:
                        if len(PSL_result_save2) != 0:
                            testsw3_mean[SE,se,i] = np.nanmean(PSL_result_save2[:,:,:,1])
                            if np.isnan(np.mean(PSL_result_save2[:,:,:,1])):
                                print('nan')
        model_itch_vs_pain = np.nanmean(testsw3_mean, axis=2)
        
        
        a1 = model_itch_vs_pain[highGroup,1]
        model_itch_vs_pain[highGroup2,1]
        a2 = model_itch_vs_pain[chloroquineGroup,1]           
        
    # rawdata 직접 load
if True:
    loadpath = 'D:\\mscore\\syncbackup\\save\\tensorData\\result\\20200302_painitch_1\\rawdata.xlsx'
    df1 = np.array(pd.read_excel(loadpath))
        
    pain = df1[:,0]
    nonpain = df1[:,1]
    roc_auc, _, _ = msacc(nonpain, pain, mslabel='', figsw2=True, legendsw=legendsw)
    plt.savefig(savepath2 + 'itch_vs_pain_ROC.png', dpi=1000)


legendsw = True

pain = cerebellum_capsaicin[:7,1]
nonpain = np.concatenate((cerebellum_capsaicin[7:,:].flatten(), cerebellum_capsaicin[:7,0], cerebellum_capsaicin[:7,2]),axis=0)
roc_auc, _, _ = msacc(nonpain, pain, mslabel='Cerebellum, pain vs non-pain, AUC:', \
                      figsw=True, fontsz=12, fontloc="lower right", legendsw=legendsw)

pain = model_itch_vs_nonitch[chloroquineGroup,1]
nonpain = np.concatenate((model_itch_vs_nonitch[salineGroup,:2].flatten(), model_itch_vs_nonitch[chloroquineGroup,0]),axis=0)
roc_auc, _, _ = msacc(nonpain, pain, mslabel='S1, itch vs non-itch, AUC:', \
                      figsw=True, fontsz=12, fontloc="lower right", legendsw=legendsw)

# itch vs pain
pain = df1[:,0]
nonpain = df1[:,1]
roc_auc, _, _ = msacc(nonpain, pain, mslabel='S1, itch vs pain, AUC:', \
                      figsw=True, fontsz=12, fontloc="lower right", legendsw=legendsw)

plt.savefig(savepath2 + 'etc_ROC_HR.png', dpi=1000)


#%%








