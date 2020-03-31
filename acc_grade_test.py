# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:05:33 2020

@author: MSBak
"""
import os  
import pickle
#import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from datetime import datetime
#import csv
#import random
#import time
#
#from keras import regularizers
#from keras.layers.core import Dropout
#from keras import initializers
#import keras
#from keras.layers.core import Dense, Activation
#from keras.layers.recurrent import LSTM
#from keras.layers.wrappers import Bidirectional
#from keras.optimizers import Adam
#from numpy.random import seed as nseed
#import tensorflow as tf
#from keras.layers import BatchNormalization

try:
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\skklab\\Google 드라이브\\save\\tensorData\\'; os.chdir(savepath);
    except:
        try:
            savepath = 'D:\\painDecorder\\save\\tensorData\\'; os.chdir(savepath);
        except:
            savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

project_list = []
project_list.append(['0330_batchnorm_1', 100, None])
project_list.append(['0330_batchnorm_2', 200, None])
project_list.append(['0330_batchnorm_3', 300, None])



mssave = []; [mssave.append([]) for u in range(3)]
q = project_list[0]
for nix, q in enumerate(project_list):
    settingID = q[0]; seed = q[1]; seed2 = int(seed+1)
    continueSW = q[2]
    
    print('settingID', settingID, 'seed', seed, 'continueSW', continueSW)

    # set the pathway2
    RESULT_SAVE_PATH = './result/'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)

    RESULT_SAVE_PATH = './result/' + settingID + '//'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)
    
    if not os.path.exists(RESULT_SAVE_PATH + 'exp/'):
        os.mkdir(RESULT_SAVE_PATH + 'exp/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'exp_raw/'):
        os.mkdir(RESULT_SAVE_PATH + 'exp_raw/') 
    
    if not os.path.exists(RESULT_SAVE_PATH + 'control/'):
        os.mkdir(RESULT_SAVE_PATH + 'control/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'control_raw/'):
        os.mkdir(RESULT_SAVE_PATH + 'control_raw/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'model/'):
        os.mkdir(RESULT_SAVE_PATH + 'model/')
    
    if not os.path.exists(RESULT_SAVE_PATH + 'tmp/'):
        os.mkdir(RESULT_SAVE_PATH + 'tmp/')

    filepath = RESULT_SAVE_PATH + 'exp_raw/'
    flist = os.listdir(filepath)
    
    for j in range(len(flist)):
        loadpath = filepath + flist[j]
        print(loadpath)
    
        with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
            pickle_load = pickle.load(f)
            mssave[j].append(pickle_load)
            
            # In[]
            
mssave = np.array(mssave)
for i in range(3):
    print(i)
    print(np.mean(mssave[i], axis=0)[128:,:3])
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
































