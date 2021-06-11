# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:16:40 2021

@author: MSBak
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
try: import pickle5 as pickle
except: import pickle

import sys
msdir = 'D:\\mscore\\code_lab'
sys.path.append('C:\\Users\\skklab\\Documents\\mscode')
sys.path.append('D:\\mscore\\code_lab')
import msFunction

SNU_FPS = 4.3650966869
def ms_syn(target_signal=None, FPS=None):
    downratio = FPS / SNU_FPS
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
    return allo

def errorCorrection(msraw): # turboreg로 발생하는 에러값을 수정함.
    sw = 0
    for col in range(msraw.shape[1]):
        for row in range(msraw.shape[0]):
            if msraw[row,col] > 10**4 or msraw[row,col] < -10**4 or np.isnan(msraw[row,col]):
                sw = 1
                print('at '+ str(row) + ' ' + str(col))
                print('turboreg error value are dectected... will process after correction')
                try:
                    msraw[row,col] = msraw[row+1,col]
                    print(msraw[row+1,col])
                except:
                    print('error, can not fullfil')                 
    return msraw, sw

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

#%% hyper

PATH = 'D:\\mscore\\syncbackup\\Project\\박하늬선생님_PD_painimaging\\raw\\'
OFFSET = 0
KHU_FPS = 5.13
MAXSE = 10
#gfiltersw, dfsw = True, True

#%%
#name = 's201229 MPTP_5.13Hz_512x512.xlsx'
# in -> 여러 sheet를 포함한 exel 1개 (SE)

def msfilepath(N):
    filename = None
    if N == 0: filename = 's201229 MPTP_5.13Hz_512x512.xlsx'
    if N == 1: filename = 's210202 MPTP_5.13Hz_512x512.xlsx'
    if N == 2: filename = 's210203 MPTP_5.13Hz_512x512.xlsx'
    if N == 3: filename = 's210216 MPTP_5.13Hz_512x512.xlsx'
    if N == 4: filename = 's210225_MPTP_5.13Hz_512x512.xlsx'
    if N == 5: filename = 's210226_MPTP_5.13Hz_512x512.xlsx'
    if N == 6: filename = 's210302_MPTP_5.13Hz_512x512.xlsx'
    if N == 7: filename = 's210405_MPTP_5.13Hz_512x512.xlsx'
    if N == 8: filename = 's210308_1_Saline_5.13Hz_512x512.xlsx'
    if N == 9: filename = 's210308_3_Saline_5.13Hz_512x512.xlsx'
    if N == 10: filename = 's210325_1_Saline_5.13Hz_512x512.xlsx'
    if N == 11: filename = 's210325_2_Saline_5.13Hz_512x512.xlsx'
    if N == 12: filename = 's210329_Saline_5.13Hz_512x512.xlsx'
    if N == 13: filename = 's210330_Saline_5.13Hz_512x512.xlsx'
    if N == 14: filename = 's210331_Saline_5.13Hz_512x512.xlsx'
    if N == 15: filename = 's210401_Saline_5.13Hz_512x512.xlsx'
    
    return filename

for n in range(999999):
    filename = msfilepath(n)
    if filename is None: N = n; print('total N', N); break

def signals_roidel_extract(name, gfiltersw=True, dfsw=True):
    loadpath = PATH + name
    array0, array2, k = [], [], -1
    while True:
        try:
            k += 1
            print('k', k, name)
            df = pd.read_excel(loadpath, sheet_name=k, header=None)
            array0.append(np.array(df))
        except:
            print('khu format으로 처리됩니다.', 'total session #', k)
            break
    
    for se in range(len(array0)):
        ROInum = array0[se].shape[1]
        for col in range(ROInum):
            if np.isnan(array0[se][0,col]):
                ROInum = col-1
                print('NaN value로 인하여 ROInum수정합니다.')
                break
        
        timeend = array0[se].shape[0]
        for row in range(timeend):
            if np.isnan(array0[se][row,0]):
                timeend = row
                print('NaN value로 인하여 data 수정합니다.')
                break
    
        array0[se] = np.array(array0[se][:timeend,:ROInum])
        array0[se] = array0[se] - OFFSET
        array0[se][np.where(array0[se]<0)] = 0
        
        savematrix = []
        for ROI in range(0, array0[se].shape[1]):
            savematrix.append(ms_syn(target_signal=array0[se][:,ROI], FPS=KHU_FPS))
        array0[se] = np.transpose(np.array(savematrix))
        
        print(se, ' max ' + str(np.max(np.max(array0[se]))) + \
              ' min ' +  str(np.min(np.min(array0[se]))))
        
        msraw = np.array(array0[se])
        while True:
            msraw, sw = errorCorrection(msraw)
            if sw == 0:
                break
        array0[se] = np.array(msraw)
        array2.append(np.array(array0[se][:,1:]))
        
    print(name + ' ROI >>>', array2[0].shape[1])
    
    # gaussian filter
    if gfiltersw:          
        array3 = [] # after gaussian filter
        for se in range(len(array2)):
            matrix = np.array(array2[se])
            tmp_matrix = []
            for neuronNum in range(matrix.shape[1]):
                tmp_matrix.append(smoothListGaussian(matrix[:,neuronNum], 10))
                
            tmp_matrix = np.transpose(np.array(tmp_matrix))
            
            array3.append(tmp_matrix)
    elif not(gfiltersw): print('no else'); import sys; sys.exit()
    
    # In F zero 계산
    if dfsw:
        array4 = []; se = 0; n = 0
        for se in range(len(array3)):
            matrix = np.array(array3[se])
            matrix = np.array(list(matrix[:,:]), dtype=np.float)
            
            f0_vector = list()
            for n in range(matrix.shape[1]):
                msmatrix = np.array(matrix[:,n])
                f0 = np.mean(np.sort(msmatrix)[0:int(round(msmatrix.shape[0]*0.3))])
                f0_vector.append(f0)
                
            f0_vector = np.array(f0_vector)   
            f_signal = np.zeros(matrix.shape)
            for frame in range(matrix.shape[0]):
                f_signal[frame,:] = (array2[se][frame, :] - f0_vector) / f0_vector
            array4.append(f_signal)
    if not(dfsw): print('no else'); import sys; sys.exit()
    
    # 이상감지1 - 급격한 변화
    rois = np.zeros(array4[0].shape[1]) 
    for se in range(len(array4)):
        wsw = True
        thr = 10 # df limit
        while wsw:
            wsw = False
            signal = np.array(array4[se])
            for n in range(signal.shape[1]):
                msplot = np.zeros(signal.shape[0]-1)
                for frame in range(signal.shape[0]-1):
                    msplot[frame] = np.abs(signal[frame+1,n] - signal[frame,n])
    
                    if msplot[frame] > thr and rois[n] < 20:
                        wsw = True
                        rois[n] += 1
    #                            print(SE, se, n, msplot[frame], frame+1)
                        array4[se][frame+1,n] = float(signal[frame,n]) # 변화가 급격한 경우 noise로 간주, 이전 intensity 값으로 대체함.
    
    # 이상감지2 - 낮은 신호량
    roi_del_ix = np.where(rois==20)[0]
    ROInum = np.array(array4[0]).shape[1]
    for ROI in range(ROInum):
        if not ROI in roi_del_ix:
            passsw = False
            for se in range(len(array4)):
                if np.max(array4[se][:,ROI]) > 0.3:
                    passsw = True
                    break
            if not(passsw):
                print(ROI, 'signal max가 0.3이하인 ROI가 존재함')
                print('ROI 에서 제거후 진행')
                tmp = list(roi_del_ix) + [ROI]
                roi_del_ix = tmp
                
    return array2, array4, roi_del_ix

#%% 개별 file 에서 signal extract 후 pickle 저장
file_list = os.listdir(PATH)
for SE in range(N):
    name = msfilepath(SE)
    pickle_save_tmp = PATH + name + '.pickle'
    if not(os.path.isfile(pickle_save_tmp)) or True:
        array2, array4, roi_del_ix = signals_roidel_extract(name, gfiltersw=True, dfsw=True)
        
        msdict = {'signals_raw': array2, 'signals': array4, 'roi_del_ix': roi_del_ix}
        with open(pickle_save_tmp, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
            print(pickle_save_tmp, '저장되었습니다.')
            
#%% 개별 file의 pickle을 불러와서 통합 + roi_del 적용
file_list = os.listdir(PATH) 
signalss = msFunction.msarray([N, MAXSE])
signalss_raw = msFunction.msarray([N, MAXSE])
SE = -1
for SE in range(N):
    name = msfilepath(SE)
    pickle_save_tmp = PATH + name + '.pickle'
    with open(pickle_save_tmp, 'rb') as f:  # Python 3: open(..., 'rb')
        msdict = pickle.load(f)
        signals = msdict['signals']
        signals_raw = msdict['signals_raw']
        roi_del_ix = msdict['roi_del_ix']
    
    ROInum = signals[0].shape[1]
    for se in range(len(signals)):
        signalss[SE][se] = np.array(signals[se])
        signalss_raw[SE][se] = np.array(signals_raw[se])
        if signalss[SE][se].shape[1] != ROInum:
            print(SE, se, name, 'session간 ROI num 불일치')
    
        pre_roinum = signalss[SE][se].shape[1]
        rois = list(range(pre_roinum))
        for j in range(len(roi_del_ix)):
            rois.remove(roi_del_ix[j])
        signalss[SE][se] = signalss[SE][se][:,rois]
        signalss_raw[SE][se] = signalss_raw[SE][se][:,rois]
        roinum = signalss[SE][se].shape[1]
    print(name, 'roinum', pre_roinum, '>>', roinum)

msdict = {'signalss_PD': signalss, 'signalss_raw_PD': signalss_raw}
pickle_save_tmp = 'C:\\mass_save\\PDpain\\' + 'mspickle_PD.pickle'    
with open(pickle_save_tmp, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
    print(pickle_save_tmp, '저장되었습니다.') 
















