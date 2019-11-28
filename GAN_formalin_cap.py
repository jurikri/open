# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:52:09 2019

@author: msbak
"""

from keras.optimizers import Adam
from keras.models import Sequential
#from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.layers import Input
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Reshape
#from keras.datasets import mnist
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
# 기본경로 설정
savepath = 'E:\\mscore\\syncbackup\\paindecoder\\save\\tensorData\\'
os.chdir(savepath)
print('savepath', savepath)

import pickle
with open('mspickle.pickle', 'rb') as f:  # Python 3: open(..., 'rb')
    msdata_load = pickle.load(f)
    
FPS = msdata_load['FPS']
N = msdata_load['N']
bahavss = msdata_load['bahavss']   # 움직임 정보
behavss2 = msdata_load['behavss2'] # 투포톤과 syn 맞춰진 버전 
movement = msdata_load['movement'] # 움직인정보를 평균내서 N x 5 matrix에 저장
msGroup = msdata_load['msGroup'] # 그룹정보
msdir = msdata_load['msdir'] # 기타 코드가 저장된 외부저장장치 경로
signalss = msdata_load['signalss'] # 투포톤 이미징데이터 -> 시계열

highGroup = msGroup['highGroup']    # 5% formalin
midleGroup = msGroup['midleGroup']  # 1% formalin
lowGroup = msGroup['lowGroup']      # 0.25% formalin
salineGroup = msGroup['salineGroup']    # saline control
restrictionGroup = msGroup['restrictionGroup']  # 5% formalin + restriciton
ketoGroup = msGroup['ketoGroup'] # 5% formalin + keto 100
lidocaineGroup = msGroup['lidocaineGroup'] # 5% formalin + lidocaine
capsaicinGroup = msGroup['capsaicinGroup'] # capsaicin
yohimbineGroup = msGroup['yohimbineGroup'] # 5% formalin + yohimbine
pslGroup = msGroup['pslGroup'] # partial sciatic nerve injury model
shamGroup = msGroup['shamGroup']

grouped_total_list = []
keylist = list(msGroup.keys())
for k in range(len(keylist)):
    grouped_total_list += msGroup[keylist[k]]

bins = 10 # 최소 time frame 간격   
noise_seed = 500

####
project_list = []
project_list.append('1128_GAN_cap')

for q in project_list:
    settingID = q
    
    RESULT_SAVE_PATH1 = './GAN/'
    if not os.path.exists(RESULT_SAVE_PATH1):
        os.mkdir(RESULT_SAVE_PATH1)
    
    RESULT_SAVE_PATH2 = './GAN/' + settingID + '//'
    if not os.path.exists(RESULT_SAVE_PATH2):
        os.mkdir(RESULT_SAVE_PATH2)

    RESULT_SAVE_PATH3 = './GAN/' + settingID + '//model_training//'
    if not os.path.exists(RESULT_SAVE_PATH3):
        os.mkdir(RESULT_SAVE_PATH3)
        
    RESULT_SAVE_PATH4 = './GAN/' + settingID + '//GANdata//'
    if not os.path.exists(RESULT_SAVE_PATH4):
        os.mkdir(RESULT_SAVE_PATH4)
        

def build_generator():
    noise_shape = (noise_seed,)

    model = Sequential()

    model.add(Dense(256, input_shape=noise_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024*2))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(sequence_length)) #, activation='tanh'))
    model.add(Reshape((sequence_length,1)))

    model.summary()

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

def build_discriminator():

#    img_shape = (img_rows, img_cols, channels)

    model = Sequential()

#    model.add(Dense(sequence_length, input_shape=(sequence_length)))
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(sequence_length,1)))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dense(int(sequence_length/2)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(int(sequence_length/4)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape= (sequence_length,1))
    validity = model(img)

    return Model(img, validity)

def save_imgs(epoch):
    showing_num = 10
    noise = np.random.normal(0, 1, (showing_num, noise_seed))
    gen_imgs = generator.predict(noise)
    
    current_signal = np.array(gen_imgs[0,:,:])
    current_signal2 = np.reshape(current_signal, (current_signal.shape[1], current_signal.shape[0]))
#    plt.figure()
#    plt.plot(current_signal2[0,:])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(showing_num, 1)
    cnt = 0
    for i in range(showing_num):
        current_signal = np.array(gen_imgs[cnt,:,:])
        current_signal2 = np.reshape(current_signal, (current_signal.shape[1], current_signal.shape[0]))
        axs[i].plot(current_signal2[0,:])
        axs[i].axis('off')
        cnt += 1
            
    fig.savefig(RESULT_SAVE_PATH3 + 'sett_for' + str(sett) + '_' + str(epoch) + '.png')
    plt.close()
    
    return None

# In[]

full_sequence = 497

# cv
mouselist = [0]
for sett in mouselist:
    excludelist = [sett]
    # 평균내어 model data 저장
    gan_model = []
    
    for SE in range(N):
        if not SE in excludelist: 
            for se in [1]: # pain, early session only 
                c1 = SE in highGroup + midleGroup + yohimbineGroup + ketoGroup and se in [1]
                c2 = SE in capsaicinGroup and se in [1]
                if c1:
                    mssignal = np.mean(signalss[SE][se], axis=1)
                    msbins = np.arange(0, mssignal.shape[0]-full_sequence+1, bins)
                    
                    for u in msbins:
                        gan_model.append(mssignal[u:u+full_sequence])
    
    print('excludelist', excludelist, 'model #', len(gan_model))
    
    ####
    # In[]
    full_sequence = 497
    sequence_length = full_sequence
#    optimizer = Adam(lr = 0.001, 0.0002, 0.5)
    
    # Build and compile the discriminator
    discriminator = build_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.000001, beta_1=0.0002, beta_2=0.5), metrics=['accuracy'])
    
    generator = build_generator()
    generator.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.05, beta_1=0.0002, beta_2=0.5))
    
    z = Input(shape=(noise_seed,))
    img = generator(z)
    
    discriminator.trainable = False
    valid = discriminator(img)
    
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer = Adam(lr=0.001, beta_1=0.0002, beta_2=0.5))
    
    # In[]
    
    # train function call
    epochs=5000; batch_size=len(gan_model); save_interval=50 # arg
    
    X_train = np.array(gan_model)
    X_train2 = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    half_batch = int(batch_size / 2)
    
    epoch = 0
    axiss = []; [axiss.append([]) for i in range(2)]
    accsave = []; outcnt = 0
    epoch2 = 0 # 시작용
    # In[]
#    epoch2 = 1298
    for epoch in range(epoch2, epochs):
      # In[]  
        # ---------------------
        #  Train Discriminator
        # ---------------------
         
        # Select a random half batch of images
        idx = np.random.randint(0, X_train2.shape[0], half_batch) # low high size
        X_train3 = X_train2[idx]
    
        noise = np.random.normal(0, 1, (half_batch, noise_seed))
        
        # Generate a half batch of new images
        gen_imgs = generator.predict(noise)

        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(X_train3, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) # 평균 
        
        noise = np.random.normal(0, 1, (batch_size, noise_seed))
        valid_y = np.array([1] * batch_size)
        g_loss = combined.train_on_batch(noise, valid_y)
    
        # Plot the progress
    #    print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
        accsave.append(d_loss_real[1])
        # If at save interval => save generated image samples
    #    if epoch % save_interval == 0:
        
        print(epoch, 'accracy for real samples', d_loss_real[1] * 100 ,'%')
    
        if epoch % save_interval == 0:
            save_imgs(epoch)
            
            axiss[0].append(epoch)
            axiss[1].append(d_loss_real[1] * 100)
 # In[]   
        if epoch > 100 and np.mean(accsave[-100:]) < 0.6:
            break
                
#        if outcnt > 2:
#            break
 # In[] 
    epoch2 = 1754
    if False: # save
     # GAN model 저장, 불러오기는 아직 없음. 
        weightsave_generator = RESULT_SAVE_PATH2 + str(epoch) + '_GAN_model_generator.h5'
        generator.save_weights(weightsave_generator)
        
        weightsave_discriminator = RESULT_SAVE_PATH2 + str(epoch) + '_GAN_model_discriminator.h5'
        discriminator.save_weights(weightsave_discriminator)
        
        weightsave_combined = RESULT_SAVE_PATH2 + str(epoch) + '_GAN_model_combined.h5'
        combined.save_weights(weightsave_combined)
        
    if False: # load
        weightsave_generator = RESULT_SAVE_PATH2 + str(epoch2) + '_GAN_model_generator.h5'
        generator.load_weights(weightsave_generator)
        
        weightsave_discriminator = RESULT_SAVE_PATH2 + str(epoch2) + '_GAN_model_discriminator.h5'
        discriminator.load_weights(weightsave_discriminator)
        
        weightsave_combined = RESULT_SAVE_PATH2 + str(epoch2) + '_GAN_model_combined.h5'
        combined.load_weights(weightsave_combined)
    
    ## Model 생성후, GAN data 생성, 저장
    generation_num = 100000
    minibatch = 100
    
    igni = True
    for batchi in range(int(generation_num/minibatch)):
        print(batchi, '/', int(generation_num/minibatch))
        noise = np.random.normal(0, 1, (minibatch, noise_seed))
        gen_imgs = generator.predict(noise)
        if igni:
            gen_imags_save = gen_imgs
            igni = False
        elif not igni:
            gen_imags_save = np.concatenate((gen_imags_save, gen_imgs), axis=0)
        
        
    sett = 0
        
    csvfile = open(RESULT_SAVE_PATH4 + 'GANdata_sett_for_' + str(sett) + '.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile)
    for row in range(gen_imags_save.shape[0]):
        csvwriter.writerow(gen_imags_save[row,:,0])
    csvfile.close()
















































