# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:19:17 2022

@author: MSBak
"""


import os
import shutil

#filename = 'test.txt'
#src = '/home/banana/' 
#dir = '/home/banana/txt/' 
#shutil.move(src + filename, dir + filename)

#%% # 하위 폴더에서 상위폴더로 꺼내기
dir_path = 'D:\\hn\\s220224_2'

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        try:
            if '.avi' in file:
                file_path = os.path.join(root, file)
                print(file_path)
                # import sys; sys.exit()
                
                if True:
                    shutil.move(file_path, dir_path + '\\' + file)
        except: pass

#%% mat file을 상위폴더 이름형식으로 바꾸기 ex) 폴더명_1.avi.mat 

dir_path = 'A:\\data_tmp\\rename\\'

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if '.avi.mat' in file:
            updir = os.path.basename(root)
            
            se = 0
            while True:
                refiename = updir + '_' + str(se) + '.avi.mat'
                if not(os.path.isfile(root + '\\' + refiename)): break
                se += 1
            
            original = root + '\\' + file
            copy = root + '\\' + refiename
            
            shutil.copy2(original, copy)
            print(original, '>>', copy) 
            
















