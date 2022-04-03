# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 16:19:17 2022

@author: MSBak
"""

# 하위 폴더에서 상위폴더로 꺼내기
import os
import shutil

#filename = 'test.txt'
#src = '/home/banana/' 
#dir = '/home/banana/txt/' 
#shutil.move(src + filename, dir + filename)

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