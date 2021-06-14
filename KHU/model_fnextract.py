# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: msbak
"""




#%%

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

# set pathway
try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
    gsync = 'D:\\mscore\\syncbackup\\google_syn\\'
except:
    try:
        savepath = 'C:\\titan_savepath\\'; os.chdir(savepath);
        gsync = 'C:\\mass_save\\'
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

with open(gsync + 'mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)

FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
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
highGroup3 = list(set(highGroup3) - set([231, 232, 233, 234]))
PSLgroup_khu =  msGroup['PSLgroup_khu']

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]
totaldataset = grouped_total_list

msset = msGroup['msset']
msset2 = msGroup['msset2']

del msGroup['msset']; del msGroup['msset2']
msset_total = np.array(pd.concat([pd.DataFrame(msset), pd.DataFrame(msset2)], ignore_index=True, axis=0))

se3set = capsaicinGroup + pslGroup + shamGroup + adenosineGroup + CFAgroup + chloroquineGroup \
+ itSalineGroup + itClonidineGroup # for test only

pslset = pslGroup + shamGroup + ipsaline_pslGroup + ipclonidineGroup
fset = highGroup + midleGroup + yohimbineGroup + ketoGroup + highGroup2 
baseonly = lowGroup + lidocaineGroup + restrictionGroup
#%

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

# t4 = total activity, movement
MAXSE = 10
t4 = np.zeros((N,MAXSE)); movement = np.zeros((N,MAXSE))
for SE in range(N):
    for se in range(len(signalss[SE])):
        t4[SE,se] = np.mean(signalss[SE][se])
        movement[SE,se] = np.mean(bahavss[SE][se]>0) # binaryzation or not

movement_syn = msFunction.msarray([N])
for SE in range(N):
    tmp = []
    for se in range(len(signalss[SE])):
        tmp.append(downsampling(bahavss[SE][se], signalss[SE][se].shape[0]))
    movement_syn[SE] = tmp

#%% grouping

group_pain_training = []
group_nonpain_training = []
group_pain_test = []
group_nonpain_test = []

SE = 0; se = 0
for SE in range(N):
    for se in range(10):
        painc, nonpainc, test_only = [], [], []
        
        # SNU

        nonpainc.append(SE in highGroup + midleGroup + highGroup2 + CFAgroup  + capsaicinGroup and se in [0])
        painc.append(SE in highGroup + midleGroup + highGroup2 + CFAgroup  + capsaicinGroup and se in [1,2])
        
        nonpainc.append(SE in salineGroup + shamGroup and se in [0,1,2])
        
        # snu psl pain
        nonpainc.append(SE in pslGroup and se in [0])
        painc.append(SE in pslGroup and se in [1,2])
        
        # khu formalin
        painc.append(SE in list(range(230, 239)) and se in [1])
        painc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [5])
        painc.append(SE in [252]  + [253, 254, 256, 260, 261, 265, 266, 267] + [269, 272] and se in [2])
        
        nonpainc.append(SE in list(range(230, 239)) and se in [0])
        nonpainc.append(SE in list(range(247, 253)) + list(range(253,273)) and se in [0, 1])
        nonpainc.append(SE in list(range(247, 252)) + [255,257, 258, 259, 262, 263, 264] + [268, 270, 271] and se in [2])
        nonpainc.append(SE in [247,248,250,251] + [257, 258, 259, 262] and se in [3,4])
        
        # khu psl
        nonpainc.append(SE in PSLgroup_khu and se in [0])
        painc.append(SE in PSLgroup_khu and se in [1,2])
        
        # test only
#        test_only.append(SE in PSLgroup_khu and se in [1,2])
        
        if np.sum(np.array(painc)) > 0:
            group_pain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_pain_training.append([SE, se])
            
        if np.sum(np.array(nonpainc)) > 0:
            group_nonpain_test.append([SE, se])
            if np.sum(np.array(test_only)) == 0:
                group_nonpain_training.append([SE, se])

#%% khu t4 check
if False:
    nonpain1, nonpain2, pain = [], [], []
    
    for SE in highGroup3:
        for se in range(MAXSE):
            if [SE, se] in group_nonpain_test: nonpain1.append(t4[SE,se])
            if [SE, se] in group_pain_test: pain.append(t4[SE,se])
            
    print(np.mean(nonpain1), np.mean(nonpain2), np.mean(pain))
    accuracy, roc_auc = msROC(nonpain1 + nonpain2, pain)
    print(accuracy, roc_auc)

# t4 만으로 formalin이 나뉠수 있음.
# 하지만 nonpain mov와 t4의 corr이 있으면, mov 정도에 의해 얼마든지 오측정 될 가능성이 있음.

# mov 정리해야함, 이부분은 나중에 처리 (외장하드에 있음)

#%% data prep


savename = 'C:\\mass_save\\fn_extraction.pickle'    
if os.path.isfile(savename) and False:
    with open(savename, 'rb') as f:  # Python 3: open(...,l 'rb')
        fninput = pickle.load(f)
 
FNINPUT = fninput
    
def ms_sampling(forlist=range(N), seset=None, signalss=signalss, ROIsw=False, fixlabel=False):
# label 지정
    X, Y, Z = [], [], [];

    # SE = forlist[0]; se = sefor[0]
    for SE in forlist:
        if seset is None: sefor = range(len(signalss[SE]))
        if not seset is None: sefor = seset;
        
        for se in sefor:
            msclass = None
            if [SE, se] in group_pain_training: msclass = [0, 1] # for pain
            if [SE, se] in group_nonpain_training: msclass = [1, 0]  # for nonpain
            if fixlabel: msclass = [0, 1] # test 전용
            
            if not(msclass is None):      
                X.append(FNINPUT[SE][se]);
                Y.append(msclass); 
                Z.append([SE, se])
               
    X, Y, Z = np.array(X), np.array(Y), np.array(Z)
    return X, Y, Z

#Xsave, Ysave, Zsave = ms_sampling(forlist=[70])
# X_tmp = X_tr; Y_tmp = Y_tr; Z_tmp = Z_tr
def upsampling(X_tmp, Y_tmp, Z_tmp, offsw=False):
    dup = 10
    X = np.array(X_tmp)
    Y = np.array(Y_tmp)
    Z = np.array(Z_tmp)
    if not(offsw):
        while True:
            n_ix = np.where(Y[:,0]==1)[0]
            p_ix = np.where(Y[:,1]==1)[0]
            print('sample distributions', 'nonpain', n_ix.shape[0], 'pain', p_ix.shape[0])
            
            nnum = n_ix.shape[0]
            pnum = p_ix.shape[0]
            
            maxnum = np.max([nnum, pnum])
            minnum = np.min([nnum, pnum])
            
            print('ratio', maxnum / minnum)
            if maxnum / minnum > dup:
                print('1/' + str(dup) + ' down')
                downix = np.where(Y[:,np.argmax([nnum, pnum])]==1)[0]
                rix = random.sample(list(downix), int(len(downix)/dup*(dup-1)))
                
                tlist = list(range(len(Y)))
                vlist = list(set(tlist)-set(rix))
                
                X = X[vlist]
                Y = Y[vlist]
                Z = Z[vlist]
                continue

            addix = np.where(Y[:,np.argmin([nnum, pnum])]==1)[0]
            if not(maxnum == minnum):
                if maxnum // minnum > 1:
                    X = np.append(X, X[addix], axis=0)
                    Y = np.append(Y, Y[addix], axis=0)
                    Z = np.append(Z, Z[addix], axis=0)
                elif maxnum // minnum == 1:
                    rix = random.sample(list(addix), maxnum-minnum)
                    X = np.append(X, X[rix], axis=0)
                    Y = np.append(Y, Y[rix], axis=0)
                    Z = np.append(Z, Z[rix], axis=0)
            else: break
            print('data set num #', len(Y), np.mean(np.array(Y), axis=0))
    
    # shuffle
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    six = random.sample(list(range(len(X))), len(X))
    
    X = X[six]
    Y = Y[six]
    Z = Z[six]

    return X, Y, Z

#%% hyperparameter

mslength = np.zeros((N,MAXSE)) * np.nan
for SE in range(N):
    for se in range(len(signalss[SE])):
        signal = np.array(signalss[SE][se])
        mslength[SE,se] = signal.shape[0]
        
FS = int(np.nanmin(mslength))
print('full_sequence', FS, 'frames')

BINS = 10 # 최소 time frame 간격 # hyper

# learning intensity
epochs = 1 # 
lr = 5e-4 # learning rate

n_hidden = int(2**7) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(2**7) # fully conneted laye node 갯수 # 8 # 원래 6 

l2_rate = 1e-3
dropout_rate1 = 0.1 # dropout rate
dropout_rate2 = 0.1 # 


#%% keras setup
from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.layers import BatchNormalization

from numpy.random import seed as nseed #
import tensorflow as tf
from keras.layers import Conv1D
from keras.layers import Flatten


def keras_setup(lr=0.01, batchnmr=False, seed=1):
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras

    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌

    input1 = keras.layers.Input(shape=(4)) 
    input10 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input1) # fully conneted layers, relu
    if batchnmr: input10 = BatchNormalization()(input10)
    input10 = Dropout(dropout_rate1)(input10) # dropout
    
    for ln in range(2):
        if int(layer_1/(2**ln)) < 2: break
        input10 = Dense(int(layer_1/(2**ln)), kernel_initializer = init, \
                        kernel_regularizer=regularizers.l2(l2_rate), activation='relu')(input10) # fully conneted layers, relu
        if batchnmr: input10 = BatchNormalization()(input10)
        input10 = Dropout(dropout_rate1)(input10) # dropout
    
    merge_4 = Dense(2, kernel_initializer = init, activation='softmax')(input10) # fully conneted layers, relu
    model = keras.models.Model(inputs=input1, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, decay=1e-8, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

model = keras_setup(lr=lr, seed=0)
print(model.summary())

#%%     project_list
project_list = []
project_list.append(['20210614_fnx_1', 100]) # project name, seed

from keras.callbacks import EarlyStopping
Callback = EarlyStopping
class EarlyStopping_ms(Callback):
    def __init__(self, monitor='accuracy', value=0.7, verbose=1, baseline=0.):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.baseline = baseline

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        # print('current', current, self.value)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current > self.value:
            # print('current', current, 'over thr')
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
callbacks = [EarlyStopping_ms(monitor='accuracy', value=0.91, verbose=1)]   

#%%

q = project_list[0]; nix = 0
# engram_save_teman = []
for nix, q in enumerate(project_list):
    settingID = q[0]; seed = q[1]  # ; seed2 = int(seed+1)
    # continueSW = q[2]
    
    print('settingID', settingID, 'seed', seed)

    # set the pathway2
    RESULT_SAVE_PATH =  gsync + 'kerasdata\\'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)

    RESULT_SAVE_PATH = gsync + 'kerasdata\\' + settingID + '\\'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)
        
### wantedlist
    runlist = list(range(N))
    validlist =  PSLgroup_khu

#%% learning 
    mslog = msFunction.msarray([N]); k=0
    model = keras_setup(lr=lr, seed=seed)
    initial_weightsave = RESULT_SAVE_PATH + 'initial_weight.h5'
    model.save_weights(initial_weightsave)
    
    savepath_pickle = RESULT_SAVE_PATH + 'resultsave.pickle'
    if os.path.isfile(savepath_pickle) and False:
        with open(savepath_pickle, 'rb') as f:  # Python 3: open(...,l 'rb')
            mssave = pickle.load(f)
            
    elif not(os.path.isfile(savepath_pickle)):
        mssave = np.zeros((N, MAXSE)) * np.nan 

    for k in range(len(validlist)):
        stopsw = False
        vlist = validlist[k]
        if not(type(vlist)==list): vlist = [vlist]
        addset = []
        for w in [validlist[k]]:
            if not(stopsw):
                if validlist[k] in msset_total[:,0]: 
                    addset += list(msset_total[np.where(msset_total[:,0]==validlist[k])[0],:][0][1:])
                if validlist[k] in msset_total[:,1:].flatten(): stopsw = True
                
        if not(stopsw): 
            final_weightsave = RESULT_SAVE_PATH + str(validlist[k]) + '_final.h5'
            if not(os.path.isfile(final_weightsave)) or True:
                vlist += addset
                print('learning 시작합니다. validation mouse #', validlist[k])
            
                trlist = list(set(runlist) - set(vlist))

                # training set
                X_tr, Y_tr, Z_tr = ms_sampling(forlist = trlist)
                print('tr set num #', len(Y_tr), np.mean(np.array(Y_tr), axis=0))
                X_tr, Y_tr, Z_tr = upsampling(X_tr, Y_tr, Z_tr, offsw=False) # ratio 10 초과시 random down -> 1:1로 upsample, -> shuffle
                print('trainingset bias', np.mean(Y_tr, axis=0))
                
                # validation set
                X_te, Y_te, Z_te = ms_sampling(forlist = vlist, seset=[0,1])
                print('tr set num #', len(Y_te), np.mean(np.array(Y_te), axis=0))
                
                # model reset
                model = keras_setup(lr=lr, seed=seed)
                model.load_weights(initial_weightsave)
                
                # for j in range(10):
                hist = model.fit(X_tr, Y_tr, batch_size=2**11, \
                                 epochs=99999, verbose=1, validation_data = (X_te, Y_te), callbacks=callbacks)
              
                s_loss=[]; s_acc=[]; sval_loss=[]; sval_acc=[];
                s_loss += list(np.array(hist.history['loss']))
                s_acc += list(np.array(hist.history['accuracy']))
                sval_loss += list(np.array(hist.history['val_loss']))
                sval_acc += list(np.array(hist.history['val_accuracy']))
      
                # save
                mssave_tmp = {'s_loss': s_loss, 's_acc': s_acc, \
                              'sval_loss': sval_loss, 'sval_acc': sval_acc}

                model.save_weights(final_weightsave)
                
                savepath_log = RESULT_SAVE_PATH + str(k) + '_log.pickle'
                with open(savepath_log, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(mssave_tmp, f, pickle.HIGHEST_PROTOCOL)
                    print(savepath_log, '저장되었습니다.')
                
                for fignum in range(2):
                    plt.figure()
                    if fignum == 0:
                        figname = str(k) + '_loss_save.png'
                        plt.plot(s_loss)
                        plt.plot(sval_loss)
                    if fignum == 1:
                        figname = str(k) + '_acc_save.png'
                        plt.plot(s_acc)
                        plt.plot(sval_acc)
                    plt.savefig(RESULT_SAVE_PATH + figname)
                    plt.close()
                
                # test
                # valSE = vlist[0]
                for valSE in vlist:
                    model.load_weights(final_weightsave)
                    for valse in range(len(signalss[valSE])):
                        X_te, Y_te, Z_te = ms_sampling(forlist = [valSE], seset=[valse])
                        predict = model.predict(X_te)
                        pain = np.mean(predict[:,1])
                        print()
                        print('te set num #', len(Y_te), 'test result SE', valSE, 'se', valse, 'pain >>', pain)
                        mssave[valSE, valse] = pain

    with open(savepath_pickle, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(mssave, f, pickle.HIGHEST_PROTOCOL)
        print(savepath_pickle, '저장되었습니다.')


#%% 평가















