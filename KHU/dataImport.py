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
highGroup =         [0,2,3,4,5,6,8,9,10,11,59] # 5%                 
# 1추가 제거
midleGroup =        [20,21,22,23,24,25,26,57] # 1%
restrictionGroup =  [27,28,29,30,43,44,45] # restriction 5%
lowGroup =          [31,32,33,35,36,37,38]  # 0.25%                 
salineGroup =       [12,13,14,15,16,17,18,19,47,48,52,53,56,58] # control
ketoGroup =         [39,40,41,42,46,49,50]
lidocaineGroup =    [51,54,55]
capsaicinGroup =    [60,61,62,64,65,82,83,104,105]
yohimbineGroup =    [63,66,67,68,69,74] 
pslGroup =          [70,71,72,73,75,76,77,78,79,80,84,85,86,87,88,93,94] 
shamGroup =         [81,89,90,91,92,97]
adenosineGroup =    [98,99,100,101,102,103,110,111,112,113,114,115]
CFAgroup =          [106,107,108,109,116,117]
highGroup2 =        [95,96]  # base / ealry / inter
chloroquineGroup =  [118,119,120,121,122,123,124,125,126,127]
itSalineGroup =     [128,129,130,134,135,138,139,140]
itClonidineGroup =  [131,132,133,136,137] # 132 3일차는 it saline으로 분류되어야함.
ipsaline_pslGroup = [141,142,143,144,145,146,147,148,149,150,152,155,156,158,159]
ipclonidineGroup =  [151,153,154,157,160,161,162,163]
gabapentinGroup =   [164,165,166,167,168,169,170,171,172,173,174,175,176,177, \
                     178,179,180,181,182,183,184,185,186, 226, 227, 228, 229]
beevenomGroup =     [187]
oxaliGroup =        [188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,220,221]
glucoseGroup =      [204,205,206,207,208,209,210,211,212,213,214,215,222,223]
PSLscsaline =       [216,217,218,219,224,225]

highGroup3 =        list(range(230,239)) + list(range(247,273))
PSLgroup_khu =      [239, 240, 241, 242, 243, 244, 245, 246]
morphineGroup =     [273, 274]

msset = [[70,72],[71,84],[75,85],[76,86],[79,88],[78,93],[80,94]]
msset2 = [[98,110],[99,111],[100,112],[101,113],[102,114],[103,115], \
          [134,135],[136,137],[128,138],[130,139],[129,140],[144,147],[145,148],[146,149], \
          [153,154],[152,155],[150,156],[151,157],[158,159],[161,160],[162,163],[167,168], \
          [169,170],[172,173],[174,175],[177,178],[179,180],[188,189],[190,191],[192,193], \
          [194,195],[196,197],[198,199],[226,227],[228,229],[239,240],[241,242],[243,244], \
          [245,246]] # baseline 독립, training 때 base를 skip 하지 않음.

nmr_list = [81,89,90,91,92,70,71,72]

for i in range(200,226,2):
    msset2.append([i, i+1])

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
msGroup['shamGroup'] = shamGroup
msGroup['adenosineGroup'] = adenosineGroup 
msGroup['highGroup2'] = highGroup2
msGroup['CFAgroup'] = CFAgroup
msGroup['chloroquineGroup'] = chloroquineGroup
msGroup['itSalineGroup'] = itSalineGroup
msGroup['itClonidineGroup'] = itClonidineGroup
msGroup['ipsaline_pslGroup'] = ipsaline_pslGroup
msGroup['ipclonidineGroup'] = ipclonidineGroup
msGroup['ipclonidineGroup'] = ipclonidineGroup
msGroup['gabapentinGroup'] = gabapentinGroup
msGroup['beevenomGroup'] = beevenomGroup
msGroup['oxaliGroup'] = oxaliGroup
msGroup['glucoseGroup'] = glucoseGroup
msGroup['PSLscsaline'] = PSLscsaline
msGroup['highGroup3'] = highGroup3
msGroup['PSLgroup_khu'] = PSLgroup_khu
msGroup['morphineGroup'] = morphineGroup

msGroup['msset'] = msset
msGroup['msset2'] = msset2


import numpy as np
import pandas as pd
import os
import sys
msdir = 'D:\\mscore\\code_lab'
sys.path.append('C:\\Users\\skklab\\Documents\\mscode')
sys.path.append('D:\\mscore\\code_lab')
import msFunction
import msfilepath
try: import pickle5 as pickle
except: import pickle
import hdf5storage
import matplotlib.pyplot as plt 

endsw=False; cnt=-1
while not(endsw):
    cnt += 1
    _, _, _, endsw = msfilepath.msfilepath1(cnt)

N = cnt; N2 = N
print('totnal N', N)

FPS = 4.3650966869
SNU_FPS = 4.3650966869
runlist = range(N)
   
#%


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
 
def ms_syn(target_signal=None, FPS=None):
    downratio = FPS / SNU_FPS
    wanted_size = int(round(target_signal.shape[0] / downratio))
    allo = np.zeros(wanted_size) * np.nan
    
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        allo[frame] = np.mean(target_signal[s:e])
        
    return allo
#%%
gfiltersw=True; skipsw = True; dfsw=True; SE = 0
 
def mssignal_save(list1=None, gfiltersw=True, skipsw = False, dfsw=True, khuoffset=None):
    
    signalss = msFunction.msarray([N])
    signalss_raw = msFunction.msarray([N])
    behavss = msFunction.msarray([N])
    roi_del_ix_save = msFunction.msarray([N]); thr = 10 # df limit
    nmr_value = np.zeros((N)) * np.nan
    
    snuformat1 = list(range(0,70)) + [74]
    snuformat2 = list(range(70,230)); snuformat2.remove(74)
    khuformat = list(range(230,N2)) 

    for SE in list1:
        print('signal preprosessing...', SE)
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
        if dfsw: 
            savepath = path + '\\singalss_behavss_withrow.pickle'
#            savepath = path + '\\singalss_behavss_offset.pickle'
#            savepath = path + '\\singalss_behavss_noneoffset.pickle'
        else: 
            savepath = path + '\\singalss_behavss_zscore.pickle'
            savepath = path + '\\singalss_behavss_zscore_biex.pickle'
#            savepath = path + '\\singalss_behavss_zscore_biex_offset.pickle'
#        savepath = path + '\\singalss_behavss_zscore_meannmr.pickle'
        
#        if gfiltersw:  savename = path + '\\signal_save.xlsx'
#        elif not(gfiltersw): savename = path + '\\signal_save_nongaussian.xlsx'
            
        if not(os.path.exists(savepath) and skipsw) or (SE in PSLgroup_khu): 
            if SE in snuformat1:
                loadpath = path + '\\' + raw_filepath
                
#                if SE in nmr_list: loadpath = loadpath[:-5] + '_m.xlsx'
                
                df = pd.read_excel(loadpath)
                ROI = df.shape[1]
                for col in range(df.shape[1]):
                    if np.isnan(df.iloc[0,col]):
                        ROI = col-1
                        break
                    
                print(str(SE) + ' ' +raw_filepath + ' ROI ' + str(ROI-1)) # 시간축 제외하고 표기
                
                timeend = df.shape[0]
                for row in range(df.shape[0]):
                    if np.isnan(df.iloc[row,0]):
                        timeend = row
                        break
                 
                msraw = np.array(df.iloc[:timeend,:ROI])
                print(str(SE) + ' max ' + str(np.max(np.max(msraw))) + ' min ' +  str(np.min(np.min(msraw))))
                
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
                            print(SE,s,frame)
                            array2.append(np.array(msraw[s:frame,1:]))
                            s = frame;
            
                    if ix == phaseInfo.shape[0]-1:
                         array2.append(np.array(msraw[s:,1:]))
                         
            elif SE in snuformat2:  
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
                
                print(SE, 'newformat으로 처리됩니다.', 'total session #', k)
                      
                for se in range(k):
                    ROI = array0[se].shape[1]
                    for col in range(array0[se].shape[1]):
                        if np.isnan(array0[se].iloc[0,col]):
                            ROI = col-1
                            print(SE, 'NaN value로 인하여 data 수정합니다.')
                            break
                    
                    timeend = array0[se].shape[0]
                    for row in range(array0[se].shape[0]):
                        if np.isnan(array0[se].iloc[row,0]):
                            timeend = row
                            print(SE, 'NaN value로 인하여 data 수정합니다.')
                            break
                        
                    array0[se] = np.array(array0[se].iloc[:timeend,:ROI])
                    print(str(SE) + ' max ' + str(np.max(np.max(array0[se]))) + \
                          ' min ' +  str(np.min(np.min(array0[se]))))
                    
                    msraw = np.array(array0[se])
                    while True:
                        msraw, sw = errorCorrection(msraw)
                        if sw == 0:
                            break
                    array0[se] = np.array(msraw)
                    array2.append(np.array(array0[se][:,1:]))
                print(str(SE) + ' ' +raw_filepath + ' ROI ', array2[0].shape[1])

### KHU format              
            elif SE in khuformat:
                loadpath = path + '\\' + raw_filepath
                array0 = []; array2 =[]; k = -1
                while True:
                    k += 1
                    print('k', k)
                    try:
                        df = pd.read_excel(loadpath, sheet_name=k, header=None)
                        array0.append(np.array(df))
                    except:
                        break
                
                print(SE, 'khu format으로 처리됩니다.', 'total session #', k)
                se = 0
                for se in range(k):
                    ROInum = array0[se].shape[1]
                    for col in range(ROInum):
                        if np.isnan(array0[se][0,col]):
                            ROInum = col-1
                            print(SE, 'NaN value로 인하여 ROInum수정합니다.')
                            break
                    
                    timeend = array0[se].shape[0]
                    for row in range(timeend):
                        if np.isnan(array0[se][row,0]):
                            timeend = row
                            print(SE, 'NaN value로 인하여 data 수정합니다.')
                            break
    
                    array0[se] = np.array(array0[se][:timeend,:ROInum])
                    array0[se] = array0[se] - khuoffset
                    array0[se][np.where(array0[se]<0)] = 0
                    
                    KHU_FPS = 1/0.191001
                    if SE in [231, 232, 233, 234]: 
                        KHU_FPS = 1/0.2291
                        print(SE, 'Galvano scanner FPS', KHU_FPS)
                    if SE > 252: KHU_FPS = 1/0.195564
                
                    savematrix = []
                    for ROI in range(0, array0[se].shape[1]):
                        savematrix.append(ms_syn(target_signal=array0[se][:,ROI], FPS=KHU_FPS))
                    array0[se] = np.transpose(np.array(savematrix))
                    
                    print(str(SE) + ' max ' + str(np.max(np.max(array0[se]))) + \
                          ' min ' +  str(np.min(np.min(array0[se]))))
                    
                    msraw = np.array(array0[se])
                    while True:
                        msraw, sw = errorCorrection(msraw)
                        if sw == 0:
                            break
                    array0[se] = np.array(msraw)
                    array2.append(np.array(array0[se][:,1:]))
                print(str(SE) + ' ' +raw_filepath + ' ROI ', array2[0].shape[1])
            
            if gfiltersw:          
                array3 = list() # after gaussian filter
                for se in range(len(array2)):
                    matrix = np.array(array2[se])
                    tmp_matrix = list()
                    for neuronNum in range(matrix.shape[1]):
                        tmp_matrix.append(smoothListGaussian(matrix[:,neuronNum], 10))
                        
                    tmp_matrix = np.transpose(np.array(tmp_matrix))
                    
                    array3.append(tmp_matrix)
            elif not(gfiltersw):
                array3 = array2
                
            # In F zero 계산
            if dfsw:
                array4 = list(); se = 0; n = 0
                for se in range(len(array3)):
                    matrix = np.array(array3[se])
                    matrix = np.array(list(matrix[:,:]), dtype=np.float)
                    
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
    
                    f0_vector = np.array(f0_vector)   
                    f_signal = np.zeros(matrix.shape)
                    for frame in range(matrix.shape[0]):
                        f_signal[frame,:] = (array3[se][frame, :] - f0_vector) / f0_vector
                    array4.append(f_signal)

                    # Zscore 계산
            if not(dfsw):
                array4 = list()
                for se in range(len(array3)):
                    matrix = np.array(array3[se])
                    matrix = np.array(list(matrix[:,:]), dtype=np.float)
#                    matrix = matrix / np.mean(matrix)
                    
                    z_matrix = np.zeros(matrix.shape) * np.nan
                    for ROI in range(matrix.shape[1]):
                        base = np.sort(matrix[:,ROI])[0:int(round(matrix.shape[1]*0.3))]
                        base_mean = np.mean(base)
                        base_sd = np.std(base, ddof=1)
                        
                        z_matrix[:,ROI] = (matrix[:,ROI] - base_mean) / base_sd
                    
                    # 양극단 제거 시작
                    biex_num = 2
                    lowix = np.argsort(np.mean(z_matrix, axis=0))[:biex_num]
                    highix = np.argsort(np.mean(z_matrix, axis=0))[::-1][:biex_num]
                    z_matrix = np.delete(z_matrix, [lowix] + [highix], axis=1)
                    # 양극단 제거 끝
                    
                    array4.append(z_matrix)
                 #
            # session 분리
#            loadpath2 = path + '\\signal_save.xlsx'
            signals = []; behavs = []; signals_raw = []
            for se in range(len(array4)):
                try:
                    df2 = array4[se]
                    df3 = np.array(pd.read_csv(path + '\\MS_' + behav_data[se]))
                    df4 = array2[se]
            
                    signals.append(np.array(df2))
                    behavs.append(np.array(df3))
                    signals_raw.append(np.array(df4))
                    
                except:  
                    print(SE, se, 'session 없습니다. 예외 group으로 판단, 이전 session을 복사하여 채웁니다.')
                    signals.append(np.array(df2))
                    behavs.append(np.array(df3))
                    signals_raw.append(np.array(df4))
                    
            msdict = {'signals': signals, 'behavs': behavs, 'signals_raw': signals_raw}
            with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                print(savepath, '저장되었습니다.')
                    
        elif os.path.exists(savepath) and skipsw or (SE in PSLgroup_khu): # 이미 있는 경우
            with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
                msdict = pickle.load(f)
            signals = msdict['signals']
            signals_raw = msdict['signals_raw']
            behavs = msdict['behavs']
                
        signalss[SE] = signals
        signalss_raw[SE] = signals_raw
        behavss[SE] = behavs
    
        # QC (in SE for)
        savepath2 = path + '\\roi_del.pickle'
        if not(os.path.isfile(savepath2)):
            rois = np.zeros(signalss[SE][0].shape[1]) 
            for se in range(len(signalss[SE])):
                wsw = True
                while wsw:
                    wsw = False
                    signal = np.array(signalss[SE][se])
                    for n in range(signal.shape[1]):
                        msplot = np.zeros(signal.shape[0]-1)
                        for frame in range(signal.shape[0]-1):
                            msplot[frame] = np.abs(signal[frame+1,n] - signal[frame,n])
            
                            if msplot[frame] > thr and rois[n] < 20:
                                wsw = True
                                rois[n] += 1
    #                            print(SE, se, n, msplot[frame], frame+1)
                                signalss[SE][se][frame+1,n] = float(signal[frame,n]) # 변화가 급격한 경우 noise로 간주, 이전 intensity 값으로 대체함.
                
                # 이상신호가 20개 이상일때 ROI 제거되도록 설계되었으나,
                # 20200818, traial간 ROI tracking을 위해서 바로 제거하지 않고, 제거 정보만 넘기는 것으로 수정하겠음.
#            for se in range(signalss[SE][0].shape[1]):
#                signal = np.array(signalss[SE][se])
#        #        signalss[SE][se] = np.delete(signal, np.where(rois==20)[0], 1)
#                if se == 0: print('ROI delete', SE, se, np.where(rois==20)[0])
            roi_del_ix = np.where(rois==20)[0]
            ROInum = np.array(signalss[SE][0]).shape[1]
            for ROI in range(ROInum):
                if not ROI in roi_del_ix:
                    passsw = False
                    for se in range(len(signalss[SE])):
                        if np.max(signalss[SE][se][:,ROI]) > 0.3:
                            passsw = True
                            break
                    if not(passsw):
                        print('SE', SE, 'ROI', ROI, "signal max가 0.3이하인 ROI가 존재함")
                        print('ROI 에서 제거후 진행')
                        tmp = list(roi_del_ix) + [ROI]
                        roi_del_ix = tmp
            
            print(str(SE), 'roi_del_ix', roi_del_ix)
            with open(savepath2, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(roi_del_ix, f, pickle.HIGHEST_PROTOCOL)
                print(savepath2, '저장되었습니다.')    
                
        else:
            with open(savepath2, 'rb') as f:  # Python 3: open(..., 'rb')
                roi_del_ix = pickle.load(f)     
        roi_del_ix_save[SE] = np.array(roi_del_ix)
        
    for SE in list1:
#        if SE == 170: import sys; sys.exit()
        mssignal = np.array(signalss[SE][0])
        roilist = list(range(mssignal.shape[1]))
        vlist = list(set(roilist) - set(roi_del_ix_save[SE]))
        se1roinum = signalss[SE][0].shape[1]
        for se in range(len(signalss[SE])):
            
            if se1roinum != signalss[SE][se].shape[1]:
                print(SE, se, 'session간 roinum 불일치')
            if se1roinum == signalss[SE][se].shape[1]:
                mssignal = np.array(signalss[SE][se])
                
                if se == 0: print(SE, signalss[SE][se].shape, signalss_raw[SE][se].shape)
                signalss[SE][se] = mssignal[:,vlist]
                signalss_raw[SE][se] = np.array(signalss_raw[SE][se])[:,vlist]
                if se == 0: print(SE, signalss[SE][se].shape, signalss_raw[SE][se].shape)

    return signalss, behavss, signalss_raw, roi_del_ix_save, nmr_value

def msMovementExtraction(list1, skipsw=False, skipfig=False):
#    movement_thr_save = np.zeros((N2,5))
    for SE in list1:
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
        behav_data_ms = list()
        for i in range(len(behav_data)):
            tmp = behav_data[i][0:3]
            behav_data_ms.append(tmp + '.avi.mat')
        
        for i in range(len(behav_data_ms)):
            loadpath = path + '\\' + behav_data_ms[i]
            savename = path + '\\' + 'MS_' + behav_data[i] 
            
            if os.path.exists(savename) and skipsw or (SE in PSLgroup_khu):
                print('이미 처리됨. skip', savename)
                continue
        
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
            if N == 57 and i == 1:
                thr = 1.25
            if N == 44 and i == 0:
                thr = 0.8
            if N == 73 and i == 0:
                thr = 1
            if N == 76 and i == 0:
                thr = 1
            if N == 83 and i == 1:
                thr = 1.1
            if N == 86 and i == 0:
                thr = 1
            if N == 87 and i == 2:
                thr = 0.93
            if N == 90 and i == 1:
                thr = 0.65
            if N == 91 and i == 0:
                thr = 0.55
            if N == 91 and i == 1:
                thr = 0.65
            if N == 97 and i == 0:
                thr = 0.53
            if N == 97 and i == 1:
                thr = 0.63
            if N == 97 and i == 2:
                thr = 0.8
            if N == 99 and i in [0,1]:
                thr = 1
            if N == 99 and i in [2]:
                thr = 1.2
            if N == 100 and i in [1]:
                thr = 0.9
            if N == 101 and i in [2]:
                thr = 1
            if N == 116 and i in [0]:
                thr = 0.9
            if N == 127 and i in [1]:
                thr = 1
            if N == 128 and i in [2]:
                thr = 1
            if N == 154 and i in [3]:
                thr = 1
                #
            if N == 223 and i in [0]:
                thr = 1.4
            if N == 224 and i in [1]:
                thr = 1.6
            if N == 224 and i in [2]:
                thr = 1.3
                   
            aline = np.zeros(diffplot.shape[0]); aline[:] = thr
#            movement_thr_save[SE,se] = thr
            ftitle = str(N) + '_' + str(i) + '_' + behav_data_ms[i] + '.png'
            if os.path.isfile(ftitle) and not(skipfig):
                plt.figure(i, figsize=(18, 9))
                
                plt.title(i)
                plt.plot(msmatrix)
                
                print(ftitle, diffplot.shape[0])
                
                plt.plot(aline)
                plt.axis([0, diffplot.shape[0], np.min(diffplot)-0.05, 2.5])
                
                savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\0728_behavior'
                if not os.path.exists(savepath):
                    os.mkdir(savepath)
                os.chdir(savepath)
                
                plt.savefig(ftitle)
                plt.close(i)

            # raw
            msmatrix[msmatrix<thr] = 0
            
            # exception
            if N == 223 and i in [3]: msmatrix[:5000] = 0
            
            savems = msmatrix

            msout = pd.DataFrame(savems ,index=None, columns=None)
            msout.to_csv(savename, index=False, header=False)
    return None
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

def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

# syn를 위한 상수 계산
def behavss2_calc(signalss, behavss):
    synsave = msFunction.msarray([N])
    SE = 6; se = 1    
    for SE in range(N):
        signals = signalss[SE]
        behavs = behavss[SE] 
        for se in range(len(signals)):
            signal = np.array(signals[se])
            meansignal = np.mean(signal,1) 
            
            behav = np.array(behavs[se])
            behav_syn = msbehav_syn(behav, signal)
                    
            xaxis = list(); yaxis = list()
            if np.mean(behav) > 0.01 or (SE == 36 and se == 3):
                synlist = np.arange(-300,300,1)
                
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
                                   or (SE == 42 and se == 1) or (SE == 220 and se == 2))
                    
                    if np.sum(behav_syn2) < np.sum(behav_syn) and msexcept: continue
     
                    if not np.sum(behav_syn2) == 0:
                        r = pearsonr(singal_syn, behav_syn2)[0]
                    elif np.sum(behav_syn2) == 0:
                        r = 0
                        
#                    print(syn, r)
                    xaxis.append(syn)
                    yaxis.append(r)
                    
                    if np.sum(np.isnan(yaxis)) < 0:
                        print(SE,se, 'nan 있어요')
                
    #            plt.plot(xaxis,yaxis)
                maxsyn = xaxis[np.argmax(yaxis)]
            else:
                maxsyn = 0
            
            synsave[SE].append(maxsyn)
            
    # 예외처리
    synsave[12][4] = 0
    synsave[18][4] = 0
    synsave[43][3] = 0 
    synsave[43][4] = 0
    #synsave[39,3] = 0
    #SE = 1; se = 1
    #SE = 8; se = 4
    
    fixlist = [[1,1],[8,4],[220,2],[220,3]]
    print('다음 session은 syn가 안맞으므로 수정합니다.')
    print(fixlist)

#%
    behavss2 = list()
    for SE in range(N):
        behavss2.append([])
        for se in range(len(signalss[SE])):
            msbehav = np.array(behavss[SE][se])
            behav_syn = downsampling(msbehav, signalss[SE][se].shape[0])
            
            if [SE, se] in fixlist:
                fix = np.zeros(behav_syn.shape[0])
                s = int(synsave[SE][se])
                if s > 0:
                    fix[s:] = behav_syn[:-s]
                elif s < 0:
                    s = -s
                    fix[:-s] = behav_syn[s:]
                
                plt.figure()
                plt.title('synfix ' + str(SE) + '_' + str(se))
                plt.plot(np.mean(signalss[SE][se], axis=1))
                plt.plot(fix)
                
            else:
                fix = behav_syn
                   
            behavss2[SE].append(fix)
    return behavss2
    
def visualizaiton_save(runlist):
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\0709'
    print('signal, movement 시각화는', savepath, '에 저장됩니다.')
    
    os.chdir(savepath)
    
    for SE in runlist:
        print('save msplot', SE)
        signals = signalss[SE]
        behavs = behavss2[SE]
        for se in range(len(signals)):
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
def dict_save(savepath):
    msdata = {
            'FPS' : FPS,
            'N' : N,
            'bahavss' : behavss, # behavior 원본 b'a'havss 오탈자인데 그대로 유지하겠음
            'behavss2' : behavss2, # behavior frame fix
            'msGroup' : msGroup,
            'msdir' : msdir,
            'signalss' : signalss,
            'signalss_raw' : signalss_raw,
            'roi_del_ix_save' : roi_del_ix_save,
            'nmr_value' : nmr_value
            }
    
    with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(msdata, f, pickle.HIGHEST_PROTOCOL)
        print(savepath, '저장되었습니다.')

#%%

runlist = range(0, N)
#runlist = highGroup + highGroup3 + midleGroup + salineGroup

msMovementExtraction(runlist, skipsw=False, skipfig=True) 

signalss, behavss, signalss_raw, roi_del_ix_save, nmr_value = mssignal_save(list1=runlist, \
                                                   gfiltersw=True, skipsw=False, dfsw=True, khuoffset = 0)

behavss2 = behavss2_calc(signalss, behavss) # signal과 함께 syn 맞춤

visualizaiton_save(runlist =  range(273, N))
savepath = 'D:\\mscore\\syncbackup\\google_syn\\mspickle.pickle'; dict_save(savepath)
savepath = 'C:\\mass_save\\PSLpain\\mspickle.pickle'; dict_save(savepath)
#savepath = 'D:\\mscore\\syncbackup\\google_syn\\mspickle_raw.pickle'; dict_save(savepath)

import sys; sys.exit()


#%%























 









