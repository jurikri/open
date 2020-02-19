# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:54:23 2020

@author: msbak
"""
import hdf5storage # python -m pip install hdf5storage
import numpy as np

filepath = 'E:\\ksh_perkinje\\suite2p\\test1\\suite2p\\plane0\\'
filename = 'MLSpike_data.mat'
matfile = hdf5storage.loadmat(filepath + filename)

signal_fit = matfile['fit']
spikeix = matfile['spikest']
FPS = 1/float(matfile['dt'])
roiindex = np.array(matfile['ixsave'][0], dtype=int) # index임 int로 변환

# MLspike로 spike 위치 정보와 spike 위치에서의 signal 값들로 dendrite별 frequency, amplitude 정보

ms_frequency = np.zeros(roiindex.shape[0]); ms_frequency[:] = np.nan # events / second
ms_amplitude = np.zeros(roiindex.shape[0]); ms_amplitude[:] = np.nan # mean amplitude
time_s = signal_fit[0,0].shape[0]/FPS

ROInum = signal_fit.shape[1]
spike_amplitude_eachROI = []
for ROI in range(ROInum):
    spikeix2 = np.array(spikeix[0,ROI][0]*FPS, dtype=int)
    spike_amplitude_eachROI.append(np.array(signal_fit[0,ROI])[spikeix2]) # for save
   
for ROI in range(ROInum):    
    ms_amplitude[roiindex[ROI]] = np.mean(spike_amplitude_eachROI[ROI])
    ms_frequency[roiindex[ROI]] = spike_amplitude_eachROI[ROI].shape[0]/FPS
    
    







































