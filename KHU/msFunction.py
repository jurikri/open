# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 15:19:13 2020

@author: MSBak
"""

#%%

import numpy as np

def nanex(array1):
    array1 = np.array(array1)
    array1 = array1[np.isnan(array1)==0]
    return array1

def msarray(listlike, nonesw=False):
    out = []
    [out.append([]) for u in range(listlike[0])]

    if nonesw: dlevel = len(listlike)
    
    if nonesw and dlevel==1:
        for i in range(listlike[0]):
            out[i] = None
    
    if len(listlike) > 1:
        for i in range(listlike[0]):
            [out[i].append([]) for u in range(listlike[1])]
            
            if nonesw and dlevel==2:
                for j in range(listlike[1]):
                    out[i][j] = None
            
            if len(listlike) > 2:
                for j in range(listlike[1]):
                    [out[i][j].append([]) for u in range(listlike[2])]
                    
                    if nonesw and dlevel==3:
                        for k in range(listlike[2]):
                            out[i][j][k] = None
                    
                    if len(listlike) > 3:
                        for k in range(listlike[2]):
                            [out[i][j][k].append([]) for u in range(listlike[3])]
                        
                            if nonesw and dlevel==4:
                                for l in range(listlike[3]):
                                    out[i][j][k][l] = None
                            
    return out




    






















