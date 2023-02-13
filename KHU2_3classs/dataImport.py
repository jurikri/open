# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:15:05 2019

@author: user
"""

"""
N값, Group 수정
N값 자동화함. Group 지정만, 
"""
MAXSE = 40
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
pslGroup =          [70,71, 73,75,76,77,78,79,80, 87 ,93,94] 
shamGroup =         [81,89,90,91,92,97]
adenosineGroup =    [98,99,100,101,102,103,110,111,112,113,114,115]
CFAgroup =          [106,107,108,109,116,117]
highGroup2 =        [95,96]  # base / ealry / inter
chloroquineGroup =  [118,119,120,121,122,123,124,125,126,127]
itSalineGroup =     [128,129,130,134,135,138,139,140]
itClonidineGroup =  [131,132,133,136,137] # 132 3일차는 it saline으로 분류되어야함.
ipsaline_pslGroup = [141,142,143,144,145,146, 150,152, 158 ]
ipclonidineGroup =  [151,153, 161,162]
gabapentinGroup =   [164,165,166,167,168,170,171,172,173,174,175,176,177, \
                     178,179,180,181,182,183,184,185,186, 226 ]
beevenomGroup =     [187]
oxaliGroup =        [188, 190, 192, 194, 196, 198, 200, 202, 220]
glucoseGroup =      [204, 206, 208, 210, 212, 214, 222]
PSLscsaline =       [216,217,218,219,224,225]

# highGroup3 =        list(range(230,239)) + list(range(247,273))
highGroup3 =        list(range(247,268)) + [269, 272]; highGroup3.remove(259)
KHU_saline =        [249,255,263,264,268,270,271]

PSLgroup_khu =      [239, 241, 243, 245]
morphineGroup =     [273, 274, 275, 276, 277, 294, 295, 296, 297, 298, 299, 300, 301]
KHUsham =           list(range(302, 312))
KHU_CFA =           list(range(312, 325))
KHU_PSL_magnolin =  list(range(332,339))

PDpain =            list(range(278, 286))
PDnonpain =         list(range(286, 294))
PDmorphine =        list(range(325,332)) + [339,340,341]

PD_ldopa =          list(range(351,358))
highGroup3_late =   list(range(342, 351))

oxali_BV =          [358, 359, 363, 364, 365, 370]
oxali_sal =         [360, 361, 362, 366, 367, 368, 369]



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
msGroup['highGroup3_late'] = highGroup3_late
msGroup['KHU_saline'] = KHU_saline

msGroup['PSLgroup_khu'] = PSLgroup_khu
msGroup['morphineGroup'] = morphineGroup
msGroup['KHUsham'] = KHUsham
msGroup['KHU_CFA'] = KHU_CFA

msGroup['PDpain'] = PDpain
msGroup['PDnonpain'] = PDnonpain
msGroup['PDmorphine'] = PDmorphine

msGroup['PD_ldopa'] = PD_ldopa
msGroup['KHU_PSL_magnolin'] = KHU_PSL_magnolin

msGroup['oxali_BV'] = oxali_BV
msGroup['oxali_sal'] = oxali_sal

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
from tqdm import tqdm
import scipy

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

def smoothListGaussian(array1, window):  
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
    snuformat1 = list(range(0,70)) + [74]
    snuformat2 = list(range(70,230)); snuformat2.remove(74)
    khuformat = list(range(230,N2)) 

    for SE in list1:
        print('signal preprosessing...', SE)
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
        if path is None: print(SE, '삭제된 SE, skip'); continue
        savepath = path + '\\singalss_behavss_withrow.pickle'
  
        if not(os.path.exists(savepath) and skipsw): 
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
                    if k == 0:
                        df = pd.read_excel(loadpath, sheet_name=k, header=None)
                        array0.append(np.array(df))
                    elif k != 0:
                        try:
                            df = pd.read_excel(loadpath, sheet_name=k, header=None)
                            array0.append(np.array(df))
                        except:
                            break
                    
                print(SE, 'khu format으로 처리됩니다.', 'total session #', k)
                se = 0
                for se in range(k):
                    ROInum = array0[se].shape[1]
                    print(ROInum)
                    for col in range(ROInum):
                        if np.isnan(array0[se][0,col]):
                            ROInum = col-1
                            print(SE, se, col, 'NaN value로 인하여 ROInum수정합니다.')
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
            elif not(gfiltersw): array3 = array2
                
            # In F zero 계산
            if dfsw:
                array4 = []; se = 0; n = 0
                for se in range(len(array3)):
                    matrix = np.array(array3[se])
                    matrix = np.array(list(matrix[:,:]), dtype=np.float)

                    f0_vector = []
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
                    df2 = array4[se] # signalss
                    df4 = array2[se] # signalss_raw
                    
                    # behavior
                    if behav_data[se] == 'empty':
                        df3 = [[], []]
                    else:
                        loadpath = path + '\\' + 'MS_' + behav_data[se]  + '.pickle'
                        with open(loadpath, 'rb') as f:  # Python 3: open(..., 'wb')
                            msdict = pickle.load(f)
                        msmatrix = msdict['msmatrix']
                        thr = msdict['thr']
                        df3 = [msmatrix, thr]
                        # print(SE, se, 'movement data 없음')
                    
                    signals.append(np.array(df2))
                    behavs.append(df3)
                    signals_raw.append(np.array(df4))
                    
                except:
                    print('없애는중')
                    print(SE, se, 'session 없습니다.')
                    import sys; sys.exit()
                    signals.append([])
                    behavs.append([])
                    signals_raw.append([])
                    
            # QC (in SE for)
            dfthr = 10 # df limit
            rois = np.zeros(signals[0].shape[1]) 
            for se in range(len(signals)):
                wsw = True
                while wsw:
                    wsw = False
                    signal = np.array(signals[se])
                    for n in range(signal.shape[1]):
                        msplot = np.zeros(signal.shape[0]-1)
                        for frame in range(signal.shape[0]-1):
                            msplot[frame] = np.abs(signal[frame+1,n] - signal[frame,n])
                            if msplot[frame] > dfthr and rois[n] < 20:
                                wsw = True
                                rois[n] += 1
                                signals[se][frame+1,n] = float(signal[frame,n]) # 변화가 급격한 경우 noise로 간주, 이전 intensity 값으로 대체함.
            roi_del_ix = np.where(rois==20)[0]
            ROInum = np.array(signals[0]).shape[1]
            for ROI in range(ROInum):
                if not ROI in roi_del_ix:
                    passsw = False
                    for se in range(len(signals)):
                        if np.max(signals[se][:,ROI]) > 0.3:
                            passsw = True
                            break
                    if not(passsw):
                        print('SE', SE, 'ROI', ROI, "signal max가 0.3이하인 ROI가 존재함")
                        print('ROI 에서 제거후 진행')
                        tmp = list(roi_del_ix) + [ROI]
                        roi_del_ix = tmp
                        
            roisave = []
            for msse in range(len(signals_raw)):
                roisave.append(signals_raw[msse].shape[1])
            if len(list(set(roisave))) > 1: print('roi matching error', SE, se); print(roisave); import sys; sys.exit()
                    
            msdict = {'signals': signals, 'behavs': behavs, 'signals_raw': signals_raw, 'roi_del_ix': roi_del_ix}
            with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                print(savepath, '저장되었습니다.')
    return None
#%%

def mssignal_save_merge():
    signalss = msFunction.msarray([N])
    signalss_raw = msFunction.msarray([N])
    behavss = msFunction.msarray([N])
    roi_del_ix_save = msFunction.msarray([N]);
    
    for SE in range(N):
        print(SE)
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
        if path is None: continue
    
        savepath = path + '\\singalss_behavss_withrow.pickle'
        
        with open(savepath, 'rb') as f:  # Python 3: open(..., 'rb')
            msdict = pickle.load(f)
        signals = msdict['signals']
        signals_raw = msdict['signals_raw']
        behavs = msdict['behavs']
        roi_del_ix = msdict['roi_del_ix']
            
        signalss[SE] = signals
        signalss_raw[SE] = signals_raw
        behavss[SE] = behavs
        roi_del_ix_save[SE] = roi_del_ix

    # roidel 적용
    # for SE in range(N):
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

    return signalss, behavss, signalss_raw, roi_del_ix_save
#%%
def msMovementExtraction(list1, skipsw=False):
#    movement_thr_save = np.zeros((N2,5))
    for SE in list1:
        path, behav_data, raw_filepath, _ = msfilepath.msfilepath1(SE)
        if path is None: continue

        for i in range(len(behav_data)):
            se = i
            savename = path + '\\' + 'MS_' + behav_data[i]  + '.pickle'
            if behav_data[i] == 'empty':
                # print('empty', behav_data[i], i)
                continue

            loadpath = path + '\\' + behav_data[i]
            if behav_data[i][-4:] == '.csv': loadpath = path + '\\' + behav_data[i][0:3] + '.avi.mat'

            if os.path.exists(savename) and skipsw: print('이미 처리됨. skip', savename); continue
            
            if not(os.path.exists(loadpath)): print('파일없음', loadpath)
            else:
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
                
                # 20221102 THR 계산방식 변경
                if SE > 357:
                    vix = np.argsort(msmatrix)[:int(len(msmatrix)*0.1)]
                    baseline = np.median(msmatrix[vix])
                    std = np.std(msmatrix[vix])
                    # thr = baseline + (std * 5)
                    thr = baseline + 0.1

                # 예외 규정 
                N = SE
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
                if N == 223 and i in [0]:
                    thr = 1.4
                if N == 224 and i in [1]:
                    thr = 1.6
                if N == 224 and i in [2]:
                    thr = 1.3
                       
                if SE == 222 and i in [5,6]: msmatrix[:5000] = 0
                if SE >= 239 and SE < 298: thr = 0.15
                if SE >= 298 and i >= 6: thr = 0.8
                if SE >= 302 and SE < 317: thr = 0.7
                if SE == 317: thr = 0.65
                if SE >= 318 and SE < 358: thr = 0.6
                
                if [SE, i] in [[322, 8], [322, 8], [322, 9], [322, 10], [322, 11]]: thr = 0.9
                if [SE, i] in [[321, 9], [323, 8], [323, 9], [323, 10], [323, 11], [324, 10], [324, 11]]: thr = 0.7
                if [SE, i] in [[319, 8], [319, 9], [319, 10], [319, 11], [320, 11], [321, 8], [321, 9], [321, 10], [321, 11]]: thr = 0.8
                if [SE, i] in [[320, 8], [320, 9]]: thr = 0.65
                if [SE, i] in [[324, 2], [324, 3], [324, 5], [324, 6]]: thr = 0.5
                if SE in [328] and se in [4,5,6,7,8]: thr = 0.55
                if [SE, se] in [[329, 11], [330, 5], [330, 8]]: thr = 0.55
                if SE in [331] and se in [4,5,6,7,8,9,10,11,12]: thr = 0.51
                if SE in [340] and se in [1,2,3]: thr = 0.5
                if SE in [340] and se in [10,11,12,13]: thr = 0.45
                if SE in [341] and se in [0,1,2,3]: thr = 0.5
                if SE in [340] and se in [14]: thr = 0.47
   
                if SE in [153] and se in [7]: thr = 1.0
                if SE in [146] and se in [1]: thr = 0.9
                if SE in [162] and se in [3]: thr = 1.2
                if SE in [204] and se in [3]: thr = 0.85
                
                if SE > 341 and SE < 358:
                  mix = np.argsort(msmatrix)[:int(len(msmatrix)*0.05)]
                  thr = np.median(msmatrix[mix]) * 1.05
                  if False:
                      plt.plot(msmatrix)
                      plt.plot(np.ones(len(msmatrix))*thr)
 
    
                if SE in [354] and se in [4]: thr = 0.8
                if SE in [354] and se in [9]: thr = 0.83
                if SE in [357] and se in [0,1,2,4,6]: thr = 0.9
    
                if SE in [358] and se in [6,7]: thr = 0.85
                if SE in [359] and se in [4, 6,7]: thr = 0.85
                if SE in [361] and se in [1,2,3]: thr = 0.78
                if SE in [363] and se in [6,7]: thr = 0.87
                if SE in [364] and se in [0,1]: thr = 0.7
                if SE in [364] and se in [6,7]: thr = 0.85
                if SE in [365] and se in [0,1,2,3]: thr = 0.7
                if SE in [365] and se in [7]: thr = 0.85
                if SE in [366] and se in [0,1,2]: thr = 0.7
                if SE in [366] and se in [7]: thr = 0.85
                if SE in [367] and se in [1,2]: thr = 0.7
                if SE in [367] and se in [6,7]: thr = 0.85
                if SE in [370] and se in [0]: thr = 0.85
                if SE in [370] and se in [1]: thr = 0.82
                
                if SE in [358] and se in [1]: thr = 0.87
                if SE in [360] and se in [2]: thr = 0.75
                
                # if [SE, i] in [[222, 3], [223, 2], [189, 0]]:
                #     msmatrix, thr = [], []
                    
                msdict = {'msmatrix': msmatrix, 'thr': thr}
                with open(savename, 'wb') as f:  # Python 3: open(..., 'wb')
                    pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
                    # print(savename, '저장되었습니다.')
                
                # msout = pd.DataFrame(savems ,index=None, columns=None)
                # msout.to_csv(savename, index=False, header=False)
    return None
#%%
from scipy.stats.stats import pearsonr 

def downsampling(msssignal, wanted_size):
    downratio = msssignal.shape[0]/wanted_size
    downsignal = np.zeros(wanted_size)
    downsignal[:] = np.nan
    for frame in range(wanted_size):
        s = int(round(frame*downratio))
        e = int(round(frame*downratio+downratio))
        downsignal[frame] = np.mean(msssignal[s:e])
        
    return np.array(downsignal)

#%% syn를 위한 상수 계산
def behavior_synfix(signals=None, behavs=None, SE=None, se=None, behavss=None):
    fixlist = [[1,1],[8,4],[220,4],[220,5],[220,6],[220,7]]
    if [SE, se] in fixlist:
        print('fix sync', SE, se)
        
        if len(behavs) > 0:
            meansignal = np.mean(signals, axis=1)
            xaxis = []; yaxis = []
            behavior_binary = behavs > behavss[SE][se][1]
    
            if np.mean(behavior_binary) > 0.01 or (SE == 36 and se == 3):
                synlist = np.arange(-300,300,1)
                
                if (SE == 36 and se == 3) or (SE == 1 and se == 2) or (SE == 38 and se == 2) or (SE == 42 and se == 1): # 예외처리
                     synlist = np.arange(-50,50,1)
                    
                for syn in synlist:
                    syn = int(round(syn))
                       
                    if syn >= 0:
                        singal_syn = meansignal[syn:]
                        sz = singal_syn.shape[0]
                        behav_syn2 = behavs[:sz]
                        
                    elif syn <0:
                        singal_syn = meansignal[:syn]
                        behav_syn2 = behavs[-syn:]
                        
                    msexcept = not((SE == 40 and se == 1) or (SE == 6 and se == 1) or (SE == 8 and se == 3) \
                                   or (SE == 10 and se == 1) or (SE == 10 and se == 3) or (SE == 11 and se == 1) \
                                   or (SE == 15 and se == 2) or (SE == 19 and se == 4) or (SE == 21 and se == 1) \
                                   or (SE == 22 and se == 0) or (SE == 32 and se == 4) or (SE == 34 and se == 0) \
                                   or (SE == 35 and se == 1) or (SE == 36 and se == 0) or (SE == 37 and se == 0) \
                                   or (SE == 37 and se == 1) or (SE == 37 and se == 4) or (SE == 38 and se == 2) \
                                   or (SE == 39 and se == 4) or (SE == 40 and se == 4) or (SE == 41 and se == 1) \
                                   or (SE == 42 and se == 0) or (SE == 41 and se == 1) or (SE == 42 and se == 0) \
                                   or (SE == 42 and se == 1) or (SE == 220 and se == 2))
                    
                    # if np.sum(behav_syn2) < np.sum(behavs) and msexcept: continue
     
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
            else: maxsyn = 0
        else: maxsyn = 0
    
        # 예외처리
        if SE == 12 and se == 4: maxsyn = 0
        if SE == 18 and se == 4: maxsyn = 0
        if SE == 43 and se == 3: maxsyn = 0
        if SE == 43 and se == 4: maxsyn = 0
            
        fix = np.zeros(behavs.shape[0])
        s = int(maxsyn)
        if s > 0:
            fix[s:] = behavs[:-s]
        elif s < 0:
            s = -s
            fix[:-s] = behavs[s:]
        
        if True:
            plt.figure()
            plt.title('synfix ' + str(SE) + '_' + str(se))
            plt.plot(np.mean(signals, axis=1))
            plt.plot(fix)
        behavs = fix
        
    return behavs
    #%%
def visualizaiton_save(runlist, signalss=None, movement_syn=None, behavss=None, dpi=100):
    savepath = 'D:\\mscore\\syncbackup\\paindecoder\\save\\msplot\\0709'
    print('signal, movement 시각화는', savepath, '에 저장됩니다.')
    os.chdir(savepath)
    
    for SE in runlist:
        print('save msplot', SE)

        for se in range(MAXSE):
            if len(signalss[SE][se]) > 0:
                behav = np.array(movement_syn[SE][se])
                signal = np.array(signalss[SE][se])
        
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
                msplot = np.mean(signal,1)
                plt.plot(msplot)
                # plt.plot(np.zeros(msplot.shape[0]))
                plt.xticks(np.arange(0, msplot.shape[0]+1, 50.0))
                
                if not(len(behav) == 0 or np.isnan(np.mean(behav))):
                    plt.subplot(414)
                    msplot = np.mean(signal,1)
                    plt.plot(behav)
                    plt.plot(np.ones(len(behav))*behavss[SE][se][1])
                    plt.xticks(np.arange(0, behav.shape[0]+1, 500.0))        
    
                #       
                plt.savefig(mstitle, dpi=dpi)
                plt.close(SE)

#%%

import sys; sys.exit()
#%%

wantedlist = PD_ldopa
# wantedlist  = [370]
elist = []

for run in wantedlist:
    try:
        runlist = [run]
        msMovementExtraction(runlist, skipsw=False)
        mssignal_save(list1=runlist, gfiltersw=True, skipsw=False, khuoffset=0)
    except:
        elist.append(run)
print()        
print(elist)

#%%

signalss, behavss, signalss_raw, roi_del_ix_save = mssignal_save_merge()

#%
from sklearn.linear_model import LinearRegression
baseratio = 0.3
signalss2 = msFunction.msarray([N,MAXSE])
for SE in range(N):
    msplot = []
    for se in range(len(signalss_raw[SE])):
        tmp = np.array(signalss_raw[SE][se])
        allo = np.zeros(tmp.shape) * np.nan
        for ROI in range(tmp.shape[1]):
            vix = np.argsort(tmp[:,ROI])[:int(round(len(tmp[:,ROI])*baseratio))]
            base = tmp[:,ROI][vix]
            m = np.median(base)
            s = np.std(base)
            
            allo[:, ROI] = (tmp[:,ROI] - m) / s
            
            df = np.mean(allo[:, ROI])
            raw = np.mean(tmp[:,ROI])
            msplot.append([raw, df])     
        signalss2[SE][se] = allo
        
        if np.inf == np.mean([df, raw]): import sys;sys.exit()
        
        if False:
            msplot = np.array(msplot)
            line_fitter = LinearRegression()
            X = msplot[:,0]; X = np.reshape(X, (X.shape[0], 1))
            line_fitter.fit(X, msplot[:,1])
            m = line_fitter.coef_
            b = line_fitter.intercept_
    
            plt.scatter(msplot[:,0], msplot[:,1], alpha = 0.5)
            xaxis = np.linspace(np.min(msplot[:,0]),np.max(msplot[:,0]),10)
            plt.plot(xaxis, xaxis*m + b, c='orange')
            print('slope', m)
 
msplot = []
for SE in range(N):
    for se in range(len(signalss_raw[SE])):
        if SE == 328 and se == 18: continue
        raw = np.mean(signalss_raw[SE][se])
        df = np.nanstd(signalss2[SE][se])
        msplot.append([raw, df])
        if np.mean([df, raw]) == np.inf: print(SE, se ); import sys;sys.exit()
        if np.isnan(np.mean([df, raw])): print(SE, se ); import sys;sys.exit()
            
msplot = np.array(msplot)
line_fitter = LinearRegression()
X = msplot[:,0]; X = np.reshape(X, (X.shape[0], 1))
line_fitter.fit(X, msplot[:,1])
m = line_fitter.coef_
b = line_fitter.intercept_

plt.scatter(msplot[:,0], msplot[:,1], alpha = 0.2)
xaxis = np.linspace(np.min(msplot[:,0]),np.max(msplot[:,0]),10)
plt.plot(xaxis, xaxis*m + b, c='orange')
print('slope', m)

# print('PD move skip 중. check')
movement_syn = msFunction.msarray([N,MAXSE])
for SE in range(N):
    for se in range(len(signalss_raw[SE])):
        behav_tmp = behavss[SE][se][0]
        if len(behav_tmp) > 0:
            mov_tmp = msFunction.downsampling(behav_tmp, signalss2[SE][se].shape[0])[0,:]
            signals=signalss_raw[SE][se]; behavs=mov_tmp
            movement_syn[SE][se] = behavior_synfix(signals=signals, behavs=behavs, SE=SE, se=se, behavss=behavss)
            if np.isnan(np.mean(movement_syn[SE][se])): movement_syn[SE][se] = []


visualizaiton_save(runlist = wantedlist, signalss=signalss2, movement_syn=movement_syn, behavss=behavss)

#%% corr matrix

inter_corr = np.zeros((N, MAXSE)) * np.nan
for SE in tqdm(range(N)):
    for se in range(len(signalss_raw[SE])):
        xdata = np.array(signalss2[SE][se])
        roiNum = xdata.shape[1]
        rmatrix = np.zeros((roiNum,roiNum)) * np.nan
        for ROI in range(roiNum):
            for ROI2 in range(ROI+1, roiNum):
                rmatrix[ROI,ROI2] = scipy.stats.pearsonr(xdata[:,ROI], xdata[:,ROI2])[0]
        inter_corr[SE,se] = np.nanmean(rmatrix)
            
signalss_df = msFunction.msarray([N, MAXSE])
for SE in tqdm(range(N)):
    for se in range(len(signalss_raw[SE])):
        if len(signalss_raw[SE][se]) > 0:
            tmp = signalss_raw[SE][se]
            mssave = np.zeros(tmp.shape) * np.nan
            roiNum = tmp.shape[1]
            for ROI in range(roiNum):
                vix = np.argsort(tmp[:,ROI])[:int(round(tmp.shape[0]*0.3))]
                m = np.median(tmp[vix,ROI])
                mssave[:,ROI] = (tmp[:,ROI] - m) / m
            
            signalss_df[SE][se] = mssave

savepath = 'C:\\SynologyDrive\\2p_data\\' + 'mspickle.pickle'
def dict_save(savepath):
    msdata = {
            'FPS' : FPS,
            'N' : N,
            'behavss' : behavss, # behavior 원본 b'a'havss 오탈자인데 그대로 유지하겠음
            'movement_syn' : movement_syn, # behavior frame fix
            'msGroup' : msGroup,
            'msdir' : msdir,
            'signalss' : signalss,
            'signalss2' : signalss2,
            'signalss_raw' : signalss_raw,
            'roi_del_ix_save' : roi_del_ix_save,
            'inter_corr' : inter_corr,
            'signalss_df' : signalss_df,
            # 'nmr_value' : nmr_value
            }
    
    with open(savepath, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(msdata, f, pickle.HIGHEST_PROTOCOL)
        print(savepath, '저장되었습니다.')
dict_save(savepath)

#%%

#%%






















