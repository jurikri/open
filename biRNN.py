# -*- coding: utf-8 -*-
# msbak, 2019. 09. 02.

# library import
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import random

from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras

#from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
import os
#from keras.callbacks import ModelCheckpoint

# set pathway
try:
    savepath = 'C:\\Users\\user\\Google 드라이브\\BMS Google drive\\희라쌤\\save\\tensorData\\'; os.chdir(savepath)
except:
    try:
        savepath = 'C:\\Users\\msbak\\Documents\\tensor\\'; os.chdir(savepath);
    except:
        savepath = ''; # os.chdir(savepath);
print('savepath', savepath)

# check the save pathway
try:
    df2 = [['SE', 'se', '%']]
    df2.append([1, 1, 1])
    csvfile = open('mscsvtest.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in range(len(df2)):
        csvwriter.writerow(df2[row])
    
    csvfile.close()
except:
    print('저장경로가 유효하지 않습니다.')

# var import
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']
behavss2 = msdata_load['behavss2']
movement = msdata_load['movement']
msGroup = msdata_load['msGroup']
msdir = msdata_load['msdir']
signalss = msdata_load['signalss']

highGroup = msGroup['highGroup']
midleGroup = msGroup['midleGroup']
lowGroup = msGroup['lowGroup']
salineGroup = msGroup['salineGroup']
restrictionGroup = msGroup['restrictionGroup']
ketoGroup = msGroup['ketoGroup']
lidocaineGroup = msGroup['lidocaineGroup']
capsaicinGroup = msGroup['capsaicinGroup'] 
yohimbineGroup = msGroup['yohimbineGroup']

# preprocessing setup

# preprecessing 사용자정의함수 선언
def preprocessing(endpoint=False , mannualsw=False):
    # mannual setting
    SE = 0; se = 1
    signalss_semi = []
    for SE in range(N):
        signalss_semi.append([])
        for se in range(5):
            signal = np.array(signalss[SE][se])
            s = 0
            
            if not(endpoint):
                e = signal.shape[0] 
            elif endpoint:
                e = endpoint # 497 # 첫 497만  쓴다.
             
            mstmp = signal[s:e,:]
            signalss_semi[SE].append(mstmp)
            
    return signalss_semi

def array_recover(X_like):
    X_like_toarray = []; X_like = np.array(X_like)
    for input_dim in range(msunit):
        tmp = np.zeros((X_like.shape[0],X_like[0,input_dim].shape[0]))
        for row in range(X_like.shape[0]):
            tmp[row,:] = X_like[row,input_dim]
    
        X_like_toarray.append(tmp)
        
        X_like_toarray[input_dim] =  \
        np.reshape(X_like_toarray[input_dim], (X_like_toarray[input_dim].shape[0],X_like_toarray[input_dim].shape[1],1))
    
    return X_like_toarray

# data 생성
SE = 0; se = 1; label = 1
def dataGeneration(SE, se, label, roiNum=None, bins=10):    
    X = []; Y = []; Z = []

    if label == 0:
        label = [1, 0] # nonpain
    elif label == 1:
        label = [0, 1] # pain
 
    if not(roiNum==None):
        s = roiNum; e = roiNum+1
    elif roiNum==None:
        s = 0; e = signalss[SE][se].shape[1]
        
    signal_full = np.array(signalss_497[SE][se])
    signal_full_roi = np.mean(signal_full[:,s:e], axis=1)
    
    lastsave = np.zeros(msunit, dtype=int)
    for frame in range(0, full_sequence - np.min(sequenceSize) + 1, bins):   
        X_tmp = []; [X_tmp.append([]) for k in range(msunit)] 
            
        for unit in range(msunit):
            if frame < full_sequence - sequenceSize[unit] + 1:
                X_tmp[unit] = (signal_full_roi[frame : frame + sequenceSize[unit]])
                lastsave[unit] = frame
                
            else:
                X_tmp[unit] = (signal_full_roi[lastsave[unit] : lastsave[unit] + sequenceSize[unit]])
#                print(frame, unit, lastsave[unit])

        X.append(X_tmp)
        Y.append(label)
        Z.append([SE,se])

    return X, Y, Z

# 최소길이 찾기
msvalue = []
for SE in range(N):
    for se in range(5):
        signal = np.array(signalss[SE][se])
        msvalue.append(signal.shape[0])

full_sequence = np.min(msvalue)
print('full_sequence', full_sequence, 'frames')

signalss_497 = preprocessing(endpoint=int(full_sequence))

msunit = 6 # input으로 들어갈 시계열 길이 및 갯수를 정함. full_sequence기준으로 1/n, 2/n ... n/n , n/n

sequenceSize = np.zeros(msunit) # 각 시계열 길이들을 array에 저장
for i in range(msunit):
    sequenceSize[i] = int(full_sequence/6*(i+1))
sequenceSize = sequenceSize.astype(np.int)

print('full_sequence', full_sequence)
print('sequenceSize', sequenceSize)
        
# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# ############# #############

# training set에 사용 될 group을 설정합니다.
mouselist = []
mouselist += msGroup['highGroup']
mouselist += msGroup['ketoGroup']
mouselist += msGroup['midleGroup']
mouselist += msGroup['salineGroup']
mouselist += msGroup['yohimbineGroup'] # 20190903: yohimbineGroup group , tarining set에 추가 
mouselist += [msGroup['lidocaineGroup'][0]] # etc set의 test를 위하여 모든 training data를 사용함.
etc = msGroup['lidocaineGroup'][0]
mouselist.sort()

# 학습할 set 결정, 따로 조작하지 않을 땐 mouselist로 설정하면 됨.
wanted = mouselist #highGroup + midleGroup + [etc] # 작동할것을 여기에 넣어 
mannual = [] # 절대 아무것도 넣지마 
print('wanted', wanted)
for i in wanted:
    try:
        mannual.append(np.where(np.array(mouselist)==i)[0][0])
    except:
        print(i, 'is excluded')
print('etc ix', np.where(np.array(mouselist)== etc)[0])
# 구지 mannual을 두고 다시 indexing 하는 이유는, 인지하기 편하기 때문임. 딱히 안써도 됨

###############
# hyperparameters #############

# learning intensity
epochs = 50 # epoch 종료를 결정할 최소 단위.
lr = 2e-3 # learning rate

n_hidden = 8 # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = 8 # fully conneted laye node 갯수 # 8
# 1부터 2배수로 test 결과 8이 performance가 충분한 최소 단위임.

# regularization
l2_rate = 0.2 # regularization 상수
dropout_rate = 0.10 # dropout late

testsw = False  # test 하지 않고 model만 저장함. # cloud 사용량을 줄이기 위한 전략.. 
trainingsw = True # training 하려면 True 
statelist = ['exp'] # ['exp', 'con']  # random shuffled control 사용 유무
validation_sw = True # 시각화목적으로만 test set을 validset으로 배치함.

acc_thr = 0.95 # 0.93 -> 0.94
batch_size = 20000

###############

# constant
maxepoch = 5000
n_in =  1 # number of features
n_out = 2 # number of class
classratio = 1 # class under sampling ratio

# project name
# settingID =  '0903_test/' # 이 폴더에 저장됨
# seed = 2

project_list = []
project_list.append(['0903_seeding_1/', 1]) # proejct name, seed
project_list.append(['0903_seeding_2/', 2]) 
# project_list.append(['0903_seeding_3/', 3]) 
# project_list.append(['0903_seeding_4/', 4])

q = project_list[0]
for q in project_list:
    settingID = q[0]; seed = q[1]
    print('settingID', settingID, 'seed', seed)

    # set the pathway2
    RESULT_SAVE_PATH = './result/'
    if not os.path.exists(RESULT_SAVE_PATH):
        os.mkdir(RESULT_SAVE_PATH)

    RESULT_SAVE_PATH = './result/' + settingID
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

    # save_hyper_parameters 기록남기기
    save_hyper_parameters = []
    save_hyper_parameters.append(['settingID', settingID])
    save_hyper_parameters.append(['epochs', epochs])
    save_hyper_parameters.append(['lr', lr])
    save_hyper_parameters.append(['n_hidden', n_hidden])
    save_hyper_parameters.append(['layer_1', layer_1])
    save_hyper_parameters.append(['l2_rate', l2_rate])
    save_hyper_parameters.append(['dropout_rate', dropout_rate])
    save_hyper_parameters.append(['acc_thr', acc_thr])
    save_hyper_parameters.append(['batch_size', batch_size])
    save_hyper_parameters.append(['seed', seed])
    save_hyper_parameters.append(['classratio', classratio])
    
    savename4 = RESULT_SAVE_PATH + 'model/' + '00_model_save_hyper_parameters.csv'
    csvfile = open(savename4, 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in range(len(save_hyper_parameters)):
        csvwriter.writerow(save_hyper_parameters[row])
    csvfile.close()


    # preprocessing 시작
    # 각 class의 data 입력준비
    X_save = []; Y_save = []; Z_save = [];
    X_save2 = []; Y_save2 = []; Z_save2 = [];
    for classnum in range(n_out):
        X_save.append([])
        Y_save.append([])
        Z_save.append([])
        
        X_save2.append([])
        Y_save2.append([])
        Z_save2.append([])


    # 각 class의 data 입력조건설정
    painGroup = highGroup + midleGroup + ketoGroup + yohimbineGroup 

    for SE in range(N):
        for se in range(5):     
            # nonpain
            c1 =  SE in painGroup and se in [0,2] # baseline, interphase
            if SE in salineGroup or c1:
                msclass = 0 # nonpain
                X, Y, Z = dataGeneration(SE, se, label = msclass) 
                X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z

            if SE in painGroup and se == 1 and SE not in [1, 26]: # 1, 26은 특별히 제외함. 
                msclass = 1 # pain
                X, Y, Z = dataGeneration(SE, se, label = msclass)
                X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
    #            
    #        if (SE == 60 and se == 0) or (SE == 61 and se == 2): # capsacine 특이 케이스 
    #            msclass = 0 # nonpain
    #            X, Y, Z = dataGeneration(SE, se, label = msclass)
    #            X_save[msclass] += X; Y_save[msclass] += Y; Z_save[msclass] += Z
                
    # class 별로 sample수 맞추기, 최소 갯수 기준으로 넘치는 class는 random sampling 한다. 
    # In[]

    mslenlist = []
    for i in range(n_out):
        mslenlist.append(len(Y_save[i]))
    sampleNum = np.min(mslenlist)
    print('sampleNum', sampleNum)

    # class간에 data 갯수의 비율을 맞추기 위한 함수.
    # class 0인 nonpain이 갯수가 더 많으므로, 필수 요소 추가후 남은 숫자 만큼 랜덤하게 뽑음
    # 이 함수에서 두 class 모두 shuffled됨. 하지만 X, Y, Z가 동일 index로 shuffle 되기 때문에 구조는 유지됨

    def ms_sampling(sampleNum, datasetX, datasetY, datasetZ, msclass):
        if msclass == 0:
            essentialIndex = []
            for j in range(len(datasetZ)):
                if datasetZ[j] in essentialList:
                    essentialIndex.append(j)
                    
            ixlist = list(range(len(datasetZ)))
            for k in essentialIndex:
                ixlist.remove(k)
                
            random.seed(seed)
            ixlist = random.sample(ixlist, int(sampleNum* classratio) - len(essentialIndex))
            ixlist = ixlist + essentialIndex
            
            datasetX = np.array(datasetX)[ixlist]
            datasetY = np.array(datasetY)[ixlist]
            datasetZ = np.array(datasetZ)[ixlist]
            
        elif msclass == 1:
            ixlist = range(len(datasetX)); ixlist = random.sample(ixlist, int(sampleNum))
            datasetX = np.array(datasetX)[ixlist]
            datasetY = np.array(datasetY)[ixlist]
            datasetZ = np.array(datasetZ)[ixlist]
            
        return datasetX, datasetY, datasetZ

    # essentialList: 반드시 포함해야 하는 nonpian session
    essentialList = [[3,0], [8,0], [14,1], [15,0], [47,1], [47,3], [48,1], \
        [48,3], [52,0], [53,1], [53,3], [53,4], [58,1], [58,3], [67,0]]



    for i in range(n_out):
        print('class', str(i), 'sampling 이전', np.array(X_save[i]).shape[0])
        X_save2[i], Y_save2[i], Z_save2[i] = \
        ms_sampling(sampleNum, datasetX = X_save[i], datasetY = Y_save[i], datasetZ = Z_save[i], msclass = i)
        print('class', str(i),'sampling 이후', np.array(X_save2[i]).shape[0])

    sw = 0
    for y in essentialList:
        if not y in Z_save[0]:
            sw = 1
            print(y, 'essentialList 누락')
            
    if sw == 0:
        print('essentialList 모두 확인됨')

    X = np.array(X_save2[0]); Y = np.array(Y_save2[0]); Z = np.array(Z_save2[0])
    for i in range(1,n_out):
        X = np.concatenate((X,X_save2[i]), axis = 0)
        Y = np.concatenate((Y,Y_save2[i]), axis = 0)
        Z = np.concatenate((Z,Z_save2[i]), axis = 0)

    X = array_recover(X)
    Y = np.array(Y); Y = np.reshape(Y, (Y.shape[0], n_out))
    indexer = np.array(Z)

    # control: label을 sessiom만 유지하면서 무작위로 섞음
    Y_control = np.array(Y)
    for SE in range(N):
        for se in range(5):
            cbn = [SE, se]
            
            identical_ix = np.where(np.sum(indexer==cbn, axis=1)==2)[0]
            if identical_ix.shape[0] != 0:
                dice = random.choice([[0,1],[1,0]])
                Y_control[identical_ix] = dice
                
    # cross validation을 위해, training / test set split            

    X_training = []; [X_training.append([]) for i in range(msunit)] # input은 msunit만큼 병렬구조임으로 list도 여러개 만듦
    Y_training_list = []
    Y_training_control_list = []
    

    Y_training = np.array(Y); Y_training_control = np.array(Y_control)# 여기서 뺸다

    # mouselist는 training set에 사용된 list임.
    # training set에 사용된 mouse의 마릿수 만큼 test set을 따로 만듦

    for test_i in range(len(mouselist)):
        
        delist = np.where(indexer[:,0]==mouselist[test_i])[0] # index는 각 data의 [SE, se]를 저장하고 있음
        for unit in range(msunit): # input은 msunit 만큼 병렬구조임. for loop으로 각자 계산함
            X_training[unit].append(np.delete(X[unit], delist, 0))
    
        Y_training_list.append(np.delete(Y_training, delist, 0))
        Y_training_control_list.append(np.delete(Y_training_control, delist, 0))

    msc1 = len(X_training[0])
    msc2 = len(Y_training_list)
    msc3 = len(Y_training_control_list)
    print(msc1, msc2, msc3, 'data, lable, label_suffled set 개수 입니다. 셋은 서로 동일해야 합니다.')
    if not(msc1 == msc2 and msc2 == msc3):
        print('set 개수 불일치, 확인요망')
        
    # 정보유출 유무 test
    if False:    
        unitNum = 1; mouseNum = 0
        delist = np.where(indexer[:,0]==mouseNum)[0]
        
        testdata = np.array(X_training[unitNum][mouseNum]) # unit 번째 병렬구조 input에서 mouseNum의 training set
        for row in range(testdata.shape[0]):
            for dataNum in range(X[unitNum][delist].shape[0]):
                indentical_score = np.sum(testdata[row,:,:] == X[unitNum][delist][dataNum]) # mouseNum번 쥐의 raw data중 dataNum번째 시계열 data
                if not indentical_score == 0:
                    print(row, indentical_score)
        
                
        # 위와 동일한 구조에서 검사대상인 set을 현재 쥐가 속하지 않은 set으로 바꾸면
        # 중복 data가 검출됨. 즉 positive control.
        unitNum = 1; mouseNum = 0
        delist = np.where(indexer[:,0]==mouseNum)[0]
        
        testdata = np.array(X_training[unitNum][mouseNum+1]) # unit 번째 병렬구조 input에서 mouseNum의 training set
        for row in range(testdata.shape[0]):
            for dataNum in range(X[unitNum][delist].shape[0]):
                indentical_score = np.sum(testdata[row,:,:] == X[unitNum][delist][dataNum]) # mouseNum번 쥐의 raw data중 dataNum번째 시계열 data
                if not indentical_score == 0:
                    print(row, indentical_score)
                    
    # 정보유출을 사전차단하기 위해 set이 아닌 raw data 변수 자체를 삭제한다.

    inputsize = np.zeros(msunit, dtype=int) 
    for unit in range(msunit):
        inputsize[unit] = X[unit].shape[1] # size 정보는 계속사용하므로, 따로 남겨놓는다.
                                
    del(X); del(Y); del(Z)
    del(Y_training); del(Y_training_control); del(Y_control)
    for j in range(n_out):
        i = 0
        del(X_save[i]); del(Y_save[i]); del(Z_save[i])
        del(X_save2[i]); del(Y_save2[i]); del(Z_save2[i])
    

    # model setup
    def keras_setup():
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        
        dt = datetime.now()
        idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

        #init = initializers.glorot_normal(seed=None)
        
        try:
            model.reset_states()
            print('올라와있는 model이 있었기 때문에, 초기화 하였습니다.')
        except:
            pass 
            # print('reset할 기존 model 없음')
        
        init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용
        
        input1 = []; [input1.append([]) for i in range(msunit)] # 최초 input layer
        input2 = []; [input2.append([]) for i in range(msunit)] # input1을 받아서 끝까지 이어지는 변수
        
        for unit in range(msunit):
            input1[unit] = keras.layers.Input(shape=(inputsize[unit], n_in)) # 각 병렬 layer shape에 따라 input 받음
            input2[unit] = Bidirectional(LSTM(n_hidden))(input1[unit]) # biRNN -> 시계열에서 단일 value로 나감
            input2[unit] = Dense(layer_1, kernel_initializer = init, activation='relu')(input2[unit]) # fully conneted layers, relu
            input2[unit] = Dropout(dropout_rate)(input2[unit]) # dropout
        
        added = keras.layers.Add()(input2) # 병렬구조를 여기서 모두 합침
        merge_1 = Dense(layer_1, kernel_initializer = init, activation='relu')(added) # fully conneted layers, relu
        merge_2 = Dropout(dropout_rate)(merge_1) # dropout
        merge_2 = Dense(n_out, kernel_initializer = init, activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
        merge_3 = Dense(n_out, input_dim=n_out, kernel_regularizer=regularizers.l2(l2_rate))(merge_2) # regularization
        merge_4 = Activation('softmax')(merge_3) # activation as softmax function
        
        model = keras.models.Model(inputs=input1, outputs = merge_4) # input output 선언
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
        
        #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
        return model, idcode

    model, idcode = keras_setup()

    if False: # 시각화 
        # 20190903, VS code로 옮긴뒤로 에러나는 중, 해결필요
        print(model.summary())
        from keras.utils import plot_model
        plot_model(model, to_file='./model.png')


    print('acc_thr', acc_thr, '여기까지 학습합니다.')
    print('maxepoch', maxepoch)

    state = 'exp'
    sett = 0; ix = 0 # for test
    for state in statelist:
        for ix, sett in enumerate(mannual):
            # training 구문입니다.
            exist_model = False; recent_model = False

            # training된 model이 있는지 검사
            if state == 'exp':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
        #        print('exp')
            elif state == 'con':
                final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final_control.h5'
        #        print('con')

            print('final_weightsave', final_weightsave)

            try:
                model.load_weights(final_weightsave) 
                exist_model = True
                print('exist_model', exist_model)
            except:
                exist_model = False
                print('exist_model', exist_model, 'load 안됨')

            # 없다면, 2시간 이내에 training이 시작되었는지 검사
            if not(exist_model) and trainingsw:
                if state == 'exp':
                    loadname = RESULT_SAVE_PATH + 'tmp/' + str([mouselist[sett]]) + '_log.csv'
                elif state == 'con':
                    loadname = RESULT_SAVE_PATH + 'tmp/' + str([mouselist[sett]]) + '_log_control.csv'

                try:
                    mscsv = []       
                    f = open(loadname, 'r', encoding='utf-8')
                    rdr = csv.reader(f)
                    for line in rdr:
                        mscsv.append(line)
                    f.close()    
                    mscsv = np.array(mscsv)

                    dt = datetime.now()
                    idcode = dt.year * 10**4 + dt.month * 10**(4-2) + dt.day * 10**(4-4) + dt.hour * 10**(4-6)

                    sameday = int(idcode) == int(float(mscsv[0][0]))
                    hour_diff = ((idcode - int(idcode)) - (float(mscsv[0][0]) - int(float(mscsv[0][0])))) * 100
                    if sameday:
                        print('mouse #', [mouselist[sett]], '은', hour_diff, '시간전에 학습을 시작했습니다.')
                        if hour_diff < 2.0:
                            recent_model = True
                        elif hour_diff >= 2.0:
                            recent_model = False        
                except:
                    recent_model = False

                # control은 추가로, exp plot이 되어있는지 확인
                if state == 'con':
                    try:
                        loadname2 = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_' + 'exp' + '_trainingSet_result.csv'
                        f = open(loadname2, 'r', encoding='utf-8')
                        f.close()
                    except:
                        print(mouselist[sett], 'exp pair 없음, control 진행을 멈춥니다.')
                        recent_model = True
                # 학습된 모델도 없고, 최근에 진행중인것도 없으니 학습 시작합니다.    
                if not(recent_model):
                    print('mouse #', [mouselist[sett]], '학습된', state, 'model 없음. 새로시작합니다.')
                    model, idcode = keras_setup() # 시작과 함께 weight reset 됩니다.

                    df2 = [idcode]
                    csvfile = open(loadname, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(df2)         
                    csvfile.close()
                    print('학습시작시간을 기록합니다.', df2)        
                    print('mouse #', [mouselist[sett]])
                    print('sample distributions.. ', np.round(np.mean(Y_training_list[sett], axis = 0), 4))

                    # validation set을 사용할경우 준비합니다.
                    if validation_sw and state == 'exp': # control은 validation을 볼 필요가없다.
                        totalROI = signalss[mouselist[sett]][0].shape[1]#; painIndex = 1
                        X_all = []; [X_all.append([]) for i in range(msunit)]
                        for se in range(3):
                            label = 0
                            if mouselist[sett] in painGroup and se == 1:
                                label = 1

                            for ROI in range(totalROI):
                                unknown_data, Y, Z = dataGeneration(mouselist[sett], se, label=label, roiNum = ROI)
                                Z = np.array(Z); tmpROI = np.zeros((Z.shape[0],1)); tmpROI[:,0] = ROI
                                Z = np.concatenate((Z, tmpROI), axis = 1)    

                                unknown_data_toarray = array_recover(unknown_data)

                                if se == 0 and ROI == 0:
                                    for k in range(msunit):
                                        X_all[k] = np.array(unknown_data_toarray[k])    
                                    Z_all = np.array(Z); Y_all = np.array(Y)

                                elif not(se == 0 and ROI == 0):
                                    for k in range(msunit):
                                        X_all[k] = np.concatenate((X_all[k],unknown_data_toarray[k]), axis=0); 
                                    Z_all = np.concatenate((Z_all,Z), axis=0); Y_all = np.concatenate((Y_all, np.array(Y)), axis=0)

                            valid = tuple([X_all, Y_all])

                    # training set을 준비합니다.    
                    shuffleix = list(range(X_training[0][sett].shape[0]))
                    np.random.shuffle(shuffleix) 

                    tr_y_shuffle = Y_training_list[sett][shuffleix]
                    tr_y_shuffle_control = Y_training_control_list[sett][shuffleix]

                    tr_x = []
                    for unit in range(msunit):
                        tr_x.append(X_training[unit][sett][shuffleix])


                    # 특정 training acc를 만족할때까지 epoch를 100단위로 지속합니다.
                    current_acc = -np.inf; cnt = -1
                    while current_acc < acc_thr: # 0.93: # 목표 최대 정확도, epoch limit
                        print('stop 조건을 표시합니다')
                        print('current_acc', current_acc, current_acc < acc_thr)

                        if cnt > maxepoch/epochs:
                            model, idcode = keras_setup()
                            current_acc = -np.inf; cnt = -1
                            print('model reset 후 처음부터 다시 학습합니다.')

                        cnt += 1; print('cnt', cnt, 'current_acc', current_acc)

                        if state == 'exp':
                            current_weightsave = RESULT_SAVE_PATH + 'tmp/'+ str(idcode) + '_' + str(mouselist[sett]) + '_my_model_weights.h5'
                        elif state == 'con':
                            current_weightsave = RESULT_SAVE_PATH + 'tmp/'+ str(idcode) + '_' + str(mouselist[sett]) + '_my_model_weights_control.h5'

                        try:
                            model.load_weights(current_weightsave)
                            print('mouse #', [mouselist[sett]], cnt, '번째 이어서 학습합니다.')

                        except:
                            print('학습 진행중인 model 없음. 새로 시작합니다')

                        # control 전용, control_epochs 구하기
                        if state == 'con':
                            mscsv = []
                            f = open(loadname2, 'r', encoding='utf-8')
                            rdr = csv.reader(f)
                            for line in rdr:
                                mscsv.append(line)
                            f.close()    
                            mscsv = np.array(mscsv)
                            control_epochs = mscsv.shape[1]

                        if validation_sw and state == 'exp':
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs, validation_data = valid)
                        elif not(validation_sw) and state == 'exp': 
                            hist = model.fit(tr_x, tr_y_shuffle, batch_size = batch_size, epochs = epochs) #, validation_data = valid)
                        elif state == 'con':
                            hist = model.fit(tr_x, tr_y_shuffle_control, batch_size = batch_size, epochs = control_epochs)

                        model.save_weights(current_weightsave)

                        if cnt == 0:
                            hist_save_loss = np.array(hist.history['loss'])
                            hist_save_acc = np.array(hist.history['acc'])

                            if validation_sw and state == 'exp':
                                hist_save_val_loss = np.array(hist.history['val_loss'])
                                hist_save_val_acc = np.array(hist.history['val_acc'])

                        elif cnt > 0:
                            hist_save_loss = np.concatenate((hist_save_loss, np.array(hist.history['loss'])), axis = 0)
                            hist_save_acc = np.concatenate((hist_save_acc, np.array(hist.history['acc'])), axis = 0)

                            if validation_sw and state == 'exp':
                                hist_save_val_loss = np.concatenate((hist_save_val_loss, np.array(hist.history['val_loss'])), axis = 0)
                                hist_save_val_acc = np.concatenate((hist_save_val_acc, np.array(hist.history['val_acc'])), axis = 0)
                        
                        # 종료조건: 
                        current_acc = np.min(hist_save_acc[-10:]) 
                        
                        if state == 'con':
                            current_acc = np.inf

                        if cnt > 7 and current_acc < 0.6:
                            # 700 epochs 후에도 학습이 안되고 있다면 초기화
                            print('고장남.. 초기화')
                            cnt = np.inf

                    # 학습 model 최종 저장
                    model.save_weights(final_weightsave)   
                    print('mouse #', [mouselist[sett]], 'traning 종료, final model을 저장합니다.')

                    # hist 저장      
                    plt.figure();
                    mouseNum = mouselist[sett]
                    plt.plot(hist_save_loss, label= '# ' + str(mouseNum) + ' loss')
                    plt.plot(hist_save_acc, label= '# ' + str(mouseNum) + ' acc')
                    plt.legend()
                    plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_trainingSet_result.png')

                    savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_trainingSet_result.csv'
                    csvfile = open(savename, 'w', newline='')
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(hist_save_acc)
                    csvwriter.writerow(hist_save_loss)
                    csvfile.close()

                    if validation_sw and state == 'exp':
                        plt.figure();
                        mouseNum = mouselist[sett]
                        plt.plot(hist_save_val_loss, label= '# ' + str(mouseNum) + ' loss')
                        plt.plot(hist_save_val_acc, label= '# ' + str(mouseNum) + ' acc')
                        plt.legend()
                        plt.savefig(RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_validationSet_result.png')

                        savename = RESULT_SAVE_PATH + 'model/' + str(mouseNum) + '_' + state + '_validationSet_result.csv'
                        csvfile = open(savename, 'w', newline='')
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(hist_save_val_acc)
                        csvwriter.writerow(hist_save_val_loss)
                        csvfile.close()

            ####### test 구문 입니다. ##########            
            if testsw:
                testlist = []
                if not(etc == mouselist[sett]):
                    testlist = [mouselist[sett]]
                    print('mouse #', [mouselist[sett]], 'set 유무를 판단합니다.')
                elif etc == mouselist[sett]:
                    print('etc group set 유무를 판단합니다.')

                    # training set에 속하지 않은 모든쥐 찾기
                    grouped_total_list = []
                    keylist = list(msGroup.keys())
                    for k in range(len(keylist)):
                        grouped_total_list += msGroup[keylist[k]]
                    for k in range(N):
                        if not (k in mouselist) and k in grouped_total_list:
                            testlist.append(k)
                    testlist.append(etc)

                if state == 'exp':
                    final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final.h5'
                elif state == 'con':
                    final_weightsave = RESULT_SAVE_PATH + 'model/' + str(mouselist[sett]) + '_my_model_weights_final_control.h5'

                trained_fortest = False
                print(final_weightsave)
                try:
                    model.load_weights(final_weightsave)
                    trained_fortest =  True
                    print('trained_fortest', trained_fortest)
                except:
                    trained_fortest = False
                    print('trained_fortest', trained_fortest)

                for test_mouseNum in testlist:
                    print('mouse #', test_mouseNum, '에 대한 기존 test 유무를 확인합니다.')
                    #    test 되어있는지 확인.

                    if state == 'exp':
                        savename = RESULT_SAVE_PATH + 'exp_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
                    elif state == 'con':
                        savename = RESULT_SAVE_PATH + 'control_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'

                    tested = False
                    print(savename)
                    try:
                        csvfile = open(savename, 'r', newline='')
                        tested = True
                        print('tested', tested)
                    except:
                        tested = False
                        print('tested', tested)

                    if not(tested) and trained_fortest: 
                        print('mouse #', test_mouseNum, 'test 진행')
                        totalROI = signalss[test_mouseNum][0].shape[1]; painIndex = 1
                        X_all = []; [X_all.append([]) for i in range(msunit)]

                        for se in range(5):
                            for ROI in range(totalROI):
                                unknown_data, Y, Z = dataGeneration(test_mouseNum, se, label=1, roiNum = ROI)
                                Z = np.array(Z); tmpROI = np.zeros((Z.shape[0],1)); tmpROI[:,0] = ROI
                                Z = np.concatenate((Z, tmpROI), axis = 1)    

                                unknown_data_toarray = array_recover(unknown_data)

                                if se == 0 and ROI == 0:
                                    for k in range(msunit):
                                        X_all[k] = np.array(unknown_data_toarray[k])    
                                    Z_all = np.array(Z); Y_all = np.array(Y)

                                elif not(se == 0 and ROI == 0):
                                    for k in range(msunit):
                                        X_all[k] = np.concatenate((X_all[k],unknown_data_toarray[k]), axis=0); 
                                    Z_all = np.concatenate((Z_all,Z), axis=0); Y_all = np.concatenate((Y_all, np.array(Y)), axis=0)

                        prediction = model.predict(X_all)

                        df1 = np.concatenate((Z_all,prediction), axis=1)
                        df2 = [['SE', 'se', 'nonpain', 'pain']]; se = 0 # 최종결과 (acc) 저장용

                        # [SE, se, ROI, nonpain, pain]
                        for se in range(5):
                            predicted_pain = np.mean(df1[:,painIndex+3][np.where(df1[:,1]==se)[0]] > 0.5)
                            mspredict = [1-predicted_pain, predicted_pain] # 전통을 중시...

                            df2.append([[test_mouseNum], se] + mspredict)

                        for d in range(len(df2)):
                            print(df2[d])

                        # 최종평가를 위한 저장 
                        # acc_experiment 저장
                        if state == 'exp':
                            savename = RESULT_SAVE_PATH + 'exp/' + 'biRNN_acc_' + str(test_mouseNum)  + '.csv'
                        elif state == 'con':
                            savename = RESULT_SAVE_PATH + 'control/' + 'biRNN_acc_' + str(test_mouseNum)  + '.csv'

                        csvfile = open(savename, 'w', newline='')
                        csvwriter = csv.writer(csvfile)
                        for row in range(len(df2)):
                            csvwriter.writerow(df2[row])
                        csvfile.close()

                        # raw 저장
                        if state == 'exp':
                            savename = RESULT_SAVE_PATH + 'exp_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'
                        elif state == 'con':
                            savename = RESULT_SAVE_PATH + 'control_raw/' + 'biRNN_raw_' + str(test_mouseNum) + '.csv'

                        csvfile = open(savename, 'w', newline='')
                        csvwriter = csv.writer(csvfile)
                        for row in range(len(df1)):
                            csvwriter.writerow(df1[row])
                        csvfile.close()

    

