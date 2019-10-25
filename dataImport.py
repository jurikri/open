# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:15:05 2019

@author: user
"""

"""
N값, Group 수정
N값 자동화함. Group 지정만, 
"""

# In[] Group 지정
highGroup =         [0,2,3,4,5,6,8,9,10,11,59] # 5%                 # exclude 7은 계속아픔. baseline도 아픔. 행동도 이상함 
# 1추가 제거
midleGroup =        [20,21,22,23,24,25,26,57] # 1%
restrictionGroup =  [27,28,29,30,43,44,45] # restriction 5%
lowGroup =          [31,32,33,35,36,37,38]  # 0.25%                  # exclude 34는 overapping이 전혀 안됨
salineGroup =       [12,13,14,15,16,17,18,19,47,48,52,53,56,58] # control
ketoGroup =         [39,40,41,42,46,49,50]
lidocaineGroup =    [51,54,55]
capsaicinGroup =    [60,61,62,64,65]
yohimbineGroup =    [63,66,67,68,69,74]
pslGroup =          [70,71,72,73,75,76,77,78,79]

msGroup = dict()
msGroup['highGroup'] = highGroup
msGroup['midleGroup'] = midleGroup
msGroup['restrictionGroup'] = restrictionGroup
msGroup['lowGroup'] = lowGroup 
msGroup['salineGroup'] = salineGroup
msGroup['ketoGroup'] = ketoGroup
msGroup['lidocaineGroup'] = lidocaineGroup
msGroup['capsaicinGroup'] = capsaicinGroup
msGroup['yohimbineGroup'] = yohimbineGroup
msGroup['pslGroup'] = pslGroup

import numpy as np
import pandas as pd
import os
import sys
msdir = 'E:\\mscore\\code_lab'; sys.path.append(msdir)
import msfilepath
import pickle
import hdf5storage
import matplotlib.pyplot as plt

endsw=False; cnt=-1
while not(endsw):
    cnt += 1
    _, _, _, endsw = msfilepath.msfilepath1(cnt)

N = cnt
print('totnal N', N)

FPS = 4.3650966869   

runlist = range(76, N)
   
# In[] 


#import sys
#msdir = 'C:\\code_lab'; sys.path.append(msdir)
#from scipy.signal import find_peaks

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
 
def mssignal_save(list1):
    newformat = [70, 71, 72, 73, 75, 76, 77, 78, 79]
    for N in list1:
        print(N, '시작합니다')
        if N not in newformat:
            path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(N)
            loadpath = path + '\\' + raw_filepath
            df = pd.read_excel(loadpath)
            ROI = df.shape[1]
            for col in range(df.shape[1]):
                if np.isnan(df.iloc[0,col]):
                    ROI = col-1
                    break
                
            print(str(N) + ' ' +raw_filepath + ' ROI ' + str(ROI-1)) # 시간축 제외하고 표기
            
            timeend = df.shape[0]
            for row in range(df.shape[0]):
                if np.isnan(df.iloc[row,0]):
                    timeend = row
                    break
             
            msraw = np.array(df.iloc[:timeend,:ROI])
            print(str(N) + ' max ' + str(np.max(np.max(msraw))) + ' min ' +  str(np.min(np.min(msraw))))
            
            while True:
                msraw, sw = errorCorrection(msraw)
                if sw == 0:
                    break
                
            # session 나눔
            phaseInfo = pd.read_excel(loadpath, sheet_name=2, header=None)
            s = 0; array2 = list()
            for ix in range(phaseInfo.shape[0]):
                for frame in range(msraw.shape[0]):
                    if abs(msraw[frame,0] -  phaseInfo.iloc[ix,0]) < 0.00001:
                        print(N,s,frame)
                        array2.append(np.array(msraw[s:frame,1:]))
                        s = frame;
        
                if ix == phaseInfo.shape[0]-1:
                     array2.append(np.array(msraw[s:,1:]))
                     
        elif N in newformat:
            path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(N)
            loadpath = path + '\\' + raw_filepath
            array0 = []; array2 =[]; k = -1
            while True:
                k += 1
                print('k', k)
                try:
                    df = pd.read_excel(loadpath, sheet_name=k, header=None)
                    array0.append(df)
                except:
                    break
            
            print(N, 'newformat으로 처리됩니다.', 'total session #', k)
                  
            for se in range(k):
                ROI = array0[se].shape[1]
                for col in range(array0[se].shape[1]):
                    if np.isnan(array0[se].iloc[0,col]):
                        ROI = col-1
                        print(N, 'NaN value로 인하여 data 수정합니다.')
                        break
                
                timeend = array0[se].shape[0]
                for row in range(array0[se].shape[0]):
                    if np.isnan(array0[se].iloc[row,0]):
                        timeend = row
                        print(N, 'NaN value로 인하여 data 수정합니다.')
                        break
                    
                array0[se] = np.array(array0[se].iloc[:timeend,:ROI])
                print(str(N) + ' max ' + str(np.max(np.max(array0[se]))) + \
                      ' min ' +  str(np.min(np.min(array0[se]))))
                
                while True:
                    array0[se], sw = errorCorrection(array0[se])
                    if sw == 0:
                        break
                    
                array2.append(np.array(array0[se][:,1:]))
            print(str(N) + ' ' +raw_filepath + ' ROI ', array2[0].shape[1])
                  
        array3 = list() # after gaussian filter
        for se in range(len(array2)):
            matrix = np.array(array2[se])
            tmp_matrix = list()
            for neuronNum in range(matrix.shape[1]):
                tmp_matrix.append(smoothListGaussian(matrix[:,neuronNum], 10))
                
            tmp_matrix = np.transpose(np.array(tmp_matrix))
            
            array3.append(tmp_matrix)
            
        array4 = list()
        for se in range(len(array3)):
            matrix = np.array(array3[se])
            matrix = np.array(list(matrix[:,:]), dtype=np.float)
            
            # In F zero 계산 
            f0_vector = list()
            for n in range(matrix.shape[1]):
                
                msmatrix = np.array(matrix[:,n])
                
                f0 = np.mean(np.sort(msmatrix)[0:int(round(msmatrix.shape[0]*0.3))])
                f0_vector.append(f0)
                
                if False:
                    plt.figure(n, figsize=(18, 9))
                    plt.title(n)
                    plt.plot(msmatrix)
                    aline = np.zeros(matrix[:,0].shape[0]); aline[:] = f0
                    plt.plot(aline)
                    print(f0, np.median(msmatrix))

            # In
            
            f0_vector = np.array(f0_vector)   
    
            f_signal = np.zeros(matrix.shape)
            for frame in range(matrix.shape[0]):
                f_signal[frame,:] = (array2[se][frame, :] - f0_vector) / f0_vector
                
            array4.append(f_signal)

        savename = path + '\\signal_save.xlsx'
        with pd.ExcelWriter(savename) as writer:  
            for se in range(len(array4)):      
                msout = pd.DataFrame(array4[se], index=None, columns=None)
                msout.to_excel(writer, sheet_name='Sheet'+str(se+1), index=False, header=False)
                
    return None

# In[]

def msMovementExtraction(list1):
    for N in list1:
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(N)
        behav_data_ms = list()
        for i in range(len(behav_data)):
            tmp = behav_data[i][0:3]
            behav_data_ms.append(tmp + '.avi.mat')
        
        for i in range(len(behav_data_ms)):
            loadpath = path + '\\' + behav_data_ms[i]
        
            df = hdf5storage.loadmat(loadpath)
            diffplot = df['msdiff_gauss']
            diffplot = np.reshape(diffplot, (diffplot.shape[1]))
        
            msmatrix = np.array(diffplot)
            msmax = np.max(msmatrix); msmin = np.min(msmatrix); diff = (msmax - msmin)/10
            
            tmpmax = -np.inf; savemax = np.nan
            for j in range(10):
                c1 = (msmatrix >= (msmin + diff * j))
                c2 = (msmatrix < (msmin + diff * (j+1)))
    #            print(np.sum(c1 * c2), j)
                if tmpmax < np.sum(c1 * c2):
                    tmpmax = np.sum(c1 * c2); savemax = j
                    
            c1 = (msmatrix >= (msmin + diff * savemax))
            c2 = (msmatrix < (msmin + diff * (savemax+1)))
            mscut = np.mean(msmatrix[(c1 * c2)])
            
            thr = mscut + 0.15
            
            # 예외 규정 
            if N == 10 and i == 0:
                thr = 1.5
            if N == 10 and i == 2:
                thr = 1.5
            if N == 10 and i == 4:
                thr = 1.5
            if N == 14 and i == 2:
                thr = mscut + 0.05
            if N == 14 and i == 3:
                thr = mscut + 0.05
            if N == 25 and i == 3:
                thr = 0.9 
            if N == 26 and i == 2:
                thr = mscut + 0.20
            if N == 25 and i == 3:
                thr = 0.5
            if N == 42 and i == 2:
                thr = 1.8
            if N == 42 and i == 3:
                thr = 1.8
            if N == 43:
                thr = 0.76
            if N == 45:
                thr = 1
            if N == 57 and i ==1:
                thr = 1.25
            if N == 44 and i ==0:
                thr = 0.8
            if N == 73 and i ==0:
                thr = 1
            if N == 76 and i ==0:
                thr = 1
                
            aline = np.zeros(diffplot.shape[0]); aline[:] = thr
            
            if True:
                plt.figure(i, figsize=(18, 9))
                ftitle = str(N) + '_' + str(i) + '_' + behav_data_ms[i] + '.png'
                plt.title(i)
                plt.plot(msmatrix)
                
                print(ftitle, diffplot.shape[0])
                
                plt.plot(aline)
                plt.axis([0, diffplot.shape[0], np.min(diffplot)-0.05, 2.5])
                
                savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\0728_behavior'
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                os.chdir(savepath)
                
                plt.savefig(ftitle)
                plt.close(i)
                
           
            savems = np.zeros(msmatrix.shape[0])
            savems[msmatrix > aline] = 1
                      
            savename = path + '\\' + 'MS_' + behav_data[i]   
            msout = pd.DataFrame(savems ,index=None, columns=None)
            msout.to_csv(savename, index=False, header=False)

# In[]

                      
mssignal_save(runlist)
msMovementExtraction(runlist)
#N, FPS, signalss, bahavss, baseindex, movement, msGroup, basess = msRun('main')


# In[] signal & behavior import
signalss = list(); bahavss = list()

# 20190903: 이부분이 시간을 많이 잡아먹는듯 한데, skip 기능을 만들어서 속도를 향상시킬 수 있을 것임.

for SE in range(N):
    print(SE, N)
    path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
#    loadpath = path + '\\events_save.xlsx'
    loadpath2 = path + '\\signal_save.xlsx'
    
    signals = list(); behavs = list() # events = list(); 
    os.chdir(path)
    
    
    for se in range(5):
        try:
    #        df = pd.read_excel(loadpath, header=None, sheet_name=se)
            df2 = pd.read_excel(loadpath2, header=None, sheet_name=se)
            df3 = np.array(pd.read_csv('MS_' + behav_data[se]))
    
    #        events.append(np.array(df))
            signals.append(np.array(df2))
            behavs.append(np.array(df3))
            
        except:
            print(SE, se, 'session 없습니다. 예외 group으로 판단, 이전 session을 복사하여 채웁니다.')
            signals.append(np.array(df2))
            behavs.append(np.array(df3))
  
#    eventss.append(events)
    signalss.append(signals)
    bahavss.append(behavs) # 변수명이 오타인데.. 이미 하도많이 사용해서 그냥 두겠음..
    
# In QC
# delta df/f0 / frame 이 thr 을 넘기는 경우 이상신호로 간주
thr = 10

def abnomarSignal(startSE):
    for SE in range(startSE, N):
        print(SE)
        signals = signalss[SE]
        for se in range(5):
            signal = np.array(signals[se])
            
            for n in range(signal.shape[1]):
                msplot = np.zeros(signal.shape[0]-1)
                for frame in range(signal.shape[0]-1):
                    msplot[frame] = np.abs(signal[frame+1,n] - signal[frame,n])
    
                    if msplot[frame] > thr:
                        print(SE, se, n, msplot[frame], frame+1)
                        signalss[SE][se][frame+1,n] = float(signal[frame,n]) # 변화가 급격한 경우 noise로 간주, 이전 intensity 값으로 대체함.
                        return 1, SE
                    
    return 0, SE

sw = 1; startSE = 0
while sw:
    sw, startSE = abnomarSignal(startSE)
                    
#                    print(signalss[SE][se][frame+1,n], signal[frame,n])

        
# In nmr factor (ROI 갯수)추정, or ROI 검사 (df/d0 0.3을 한번도 넘지 못한 ROI의 존재 유무)
for SE in range(N):
    signals = signalss[SE]  
    
    ROIsw = np.zeros(np.array(signals[0]).shape[1])
    for n in range(np.array(signals[0]).shape[1]):
        sw = 0
        for se in range(5):
            signal = np.array(signals[se])
        
            if np.max(signal[:,n]) > 0.3: # 0.3에 특별한 의미는 없고, 경험적으로 한번도 0.3을 못넘는 ROI는 발견되지 않음.
                ROIsw[n] = 1
                break
            
    if np.sum(ROIsw) != np.array(signals[0]).shape[1]:
        print("signal이 없는 ROI가 존재함")
    
# In movement 계산 
movement = np.zeros((N,5))
for SE in range(N):
    print(SE,N)
    behavs = np.array(bahavss[SE])
    for se in range(5):
        behav = np.array(behavs[se])
        movement[SE,se] = np.sum(behav)/behav.shape[0]
        
# In
from scipy.stats.stats import pearsonr 
def msbehav_syn(behav, signal): # behav syn 맞추기 
    behav = np.array(behav)
    signal = np.array(signal)
    
    behav_syn = np.zeros(signal.shape[0])
    syn = signal.shape[0]/behav.shape[0]
    for behavframe in range(behav.shape[0]):
        imagingframe = int(round(behavframe*syn))
    
        if behav[behavframe] > 0 and not imagingframe == signal.shape[0]:
            behav_syn[imagingframe] += 1
            
    return behav_syn  

# syn를 위한 상수 계산
    
# 여긴 계산은 하지만, 대부분 사용하지 않는 코드임.
# 비해비어 & 시그널의 최대 싱크를 맞춰보기만함.
# 수치를와 시각화자료를 보고, 명백히 데이터획득과정의 에러로 판단되면 수정하고 그렇지 않으면 그냥 둠.
# 명백히 달라서 수정하는것 -> fixlist

synsave = np.zeros((N,5))
SE = 6; se = 1    
for SE in range(N):
    signals = signalss[SE]
    behavs = bahavss[SE] 
    for se in range(5):
        signal = np.array(signals[se])
        meansignal = np.mean(signal,1) 
        
        behav = np.array(behavs[se])
        behav_syn = msbehav_syn(behav, signal)
                
        xaxis = list(); yaxis = list()
        if np.mean(behav) > 0.01 or (SE == 36 and se == 3):
            synlist = np.arange(-300,301,1)
            
            if (SE == 36 and se == 3) or (SE == 1 and se == 2) or (SE == 38 and se == 2) or (SE == 42 and se == 1): # 예외처리
                 synlist = np.arange(-50,50,1)
                
            for syn in synlist:
                syn = int(round(syn))
                   
                if syn >= 0:
                    singal_syn = meansignal[syn:]
                    sz = singal_syn.shape[0]
                    behav_syn2 = behav_syn[:sz]
                    
                elif syn <0:
                    singal_syn = meansignal[:syn]
                    behav_syn2 = behav_syn[-syn:]
                    
                msexcept = not((SE == 40 and se == 1) or (SE == 6 and se == 1) or (SE == 8 and se == 3) \
                               or (SE == 10 and se == 1) or (SE == 10 and se == 3) or (SE == 11 and se == 1) \
                               or (SE == 15 and se == 2) or (SE == 19 and se == 4) or (SE == 21 and se == 1) \
                               or (SE == 22 and se == 0) or (SE == 32 and se == 4) or (SE == 34 and se == 0) \
                               or (SE == 35 and se == 1) or (SE == 36 and se == 0) or (SE == 37 and se == 0) \
                               or (SE == 37 and se == 1) or (SE == 37 and se == 4) or (SE == 38 and se == 2) \
                               or (SE == 39 and se == 4) or (SE == 40 and se == 4) or (SE == 41 and se == 1) \
                               or (SE == 42 and se == 0) or (SE == 41 and se == 1) or (SE == 42 and se == 0) \
                               or (SE == 42 and se == 1))
                
                if np.sum(behav_syn2) < np.sum(behav_syn) and msexcept:
                    continue
                    
                xaxis.append(syn)
                yaxis.append(pearsonr(singal_syn, behav_syn2)[0])
                
                if np.sum(np.isnan(yaxis)) < 0:
                    print(SE,se, 'nan 있어요')
            
#            plt.plot(xaxis,yaxis)
            maxsyn = xaxis[np.argmax(yaxis)]
        else:
            maxsyn = 0
        
        synsave[SE,se] = maxsyn
        
# 예외처리
synsave[12,4] = 0
synsave[18,4] = 0
synsave[43,3] = 0 
synsave[43,4] = 0
#synsave[39,3] = 0
SE = 1; se = 1
SE = 8; se = 4

fixlist = [[1,1],[8,4]]
print('다음 session은 syn가 안맞으므로 수정합니다.')
print(fixlist)

behavss2 = list()
for SE in range(N):
    behavss2.append([])
    for se in range(5):
       
        behav_syn = msbehav_syn(bahavss[SE][se], signalss[SE][se])
        
        if [SE, se] in fixlist:
            fix = np.zeros(behav_syn.shape[0])
            s = int(synsave[SE,se])
            if s > 0:
                fix[s:] = behav_syn[:-s]
            elif s < 0:
                s = -s
                fix[:-s] = behav_syn[s:]
                
            plt.plot(np.mean(signalss[SE][se], axis=1))
            plt.plot(fix)
            
        else:
            fix = behav_syn
               
        behavss2[SE].append(fix)
    
if True:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\0709'
    print('signal, movement 시각화는', savepath, '에 저장됩니다.')
    
    os.chdir(savepath)
    
    for SE in range(N):
        print('save msplot', SE)
        signals = signalss[SE]
        behavs = behavss2[SE]
        for se in range(5):
            behav = np.array(behavs[se])
            signal = np.array(signals[se])
    
            plt.figure(SE, figsize=(18, 9))
    
            plt.subplot(411)
            for n in range(signal.shape[1]):
                msplot = signal[:,n]
                plt.plot(msplot)
                
            mstitle = 'msplot_' + str(SE) + '_' + str(se) + '.png'
            plt.title(mstitle)
                
            scalebar = np.ones(int(round(signal.shape[0]/FPS)))
            plt.subplot(412)
    #        plt.plot(scalebar)
            plt.xticks(np.arange(0, scalebar.shape[0]+1, 5.0))
                
            plt.subplot(413)
            msplot = np.median(signal,1)
            plt.plot(msplot)
            plt.plot(np.zeros(msplot.shape[0]))
            plt.xticks(np.arange(0, msplot.shape[0]+1, 50.0))
            
            plt.subplot(414)
            msplot = np.mean(signal,1)
            plt.plot(behav)
            plt.xticks(np.arange(0, behav.shape[0]+1, 500.0))

            #       
            plt.savefig(mstitle)
            plt.close(SE)

        
#    import baseestimator


try:
    savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\msbak\\Documents\\tensor\\'; os.chdir(savepath);
    except:
        savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

msdata = {
        'FPS' : FPS,
        'N' : N,
        'bahavss' : bahavss,
        'behavss2' : behavss2,
#        'baseindex' : baseindex,
#        'basess' : basess,
        'movement' : movement,
        'msGroup' : msGroup,
        'msdir' : msdir,
        'signalss' : signalss
        }

with open('mspickle.pickle', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(msdata, f, pickle.HIGHEST_PROTOCOL)
    print('mspickle.pickle 저장되었습니다.')


