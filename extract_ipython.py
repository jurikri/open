import os
try:
    savepath = 'C:\\Users\\user\\Google 드라이브\\BMS Google drive\\희라쌤\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\msbak\\Documents\\tensor\\'; os.chdir(savepath);
    except:
        savepath = ''; # os.chdir(savepath); 
print('savepath', savepath)

RESULT_SAVE_PATH = './result/'
if not os.path.exists(RESULT_SAVE_PATH):
  os.mkdir(RESULT_SAVE_PATH)

import zipfile
filename = '0819_test_2'
fantasy_zip = zipfile.ZipFile('./result/' + filename + '.zip')
fantasy_zip.extractall('./result/'  + filename)
 
fantasy_zip.close()
