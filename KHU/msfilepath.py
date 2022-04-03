# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 20:58:38 2019

@author: user
"""
def msfilepath1(session):

    mainpath = 'D:\\mscore\\syncbackup\\paindecoder\\data\\'

    endsw = False
    path = None
    behav_data = None
    raw_filepath = None
    
    if session == 0:
        path = mainpath + 's1113_F_before5min_awake'
        behav_data = list()
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        raw_filepath = 's1113_Formalin_before5m_awake.xlsx'
        
    elif session == 1:
        path = mainpath + 'S0501_1'
        behav_data = list()
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        raw_filepath = 's0501_1_Formalin_awake.xlsx'
        
    elif session == 2:
        path = mainpath + 'S0611_2'
        behav_data = list()
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        raw_filepath = 's0611_2_Formalin_awake.xlsx'
        
    elif session == 3:
        path = mainpath + 's1114_F_before5min_awake'
        behav_data = list()
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('008000.csv')
        raw_filepath = 's1114_Formalin_before5m_awake.xlsx'
        
    elif session == 4:
        path = mainpath + 's1123_1_before5min_awake'
        behav_data = list()
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        raw_filepath = 's1123_1_Formalin_before5min_awake.xlsx'
        
    elif session == 5:
        path = mainpath + 's1207_2_before5min_awake'
        behav_data = list()
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('008000.csv')
        raw_filepath = 's1207_2_Formalin_before5min_awake.xlsx'
        
    elif session == 6:
        path = mainpath + 's1002_1'
        behav_data = list()
        raw_filepath = 's1002_1_Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('009000.csv')
        
    elif session == 7:
        path = mainpath + 's1002_2'
        behav_data = list()
        raw_filepath = 's1002_2_Fomaln_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 8:
        path = mainpath + 's1222_1_before5min_awake'
        behav_data = list()
        raw_filepath = 's1222_1_Formalin_before5min_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 9:
        path = mainpath + 'S0622_2'
        behav_data = list()
        raw_filepath = 's0622_2_Formalin_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('010000.csv')
        
    elif session == 10:
        path = mainpath + 's0702_modify'
        behav_data = list()
        raw_filepath = 's0702_Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('010000.csv')
         
    elif session == 11:
        path = mainpath + 's0508_1'
        behav_data = list()
        raw_filepath = 's0508_1_Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 12:
        path = mainpath + 's0704'
        behav_data = list()
        raw_filepath = 'S0704_Saline_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('010000.csv')
        
    elif session == 13:
        path = mainpath + 's0622_1'
        behav_data = list()
        raw_filepath = 's0622_1_Saline_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('010000.csv')
        
    elif session == 14:
        path = mainpath + 'S0615_saline'
        behav_data = list()
        raw_filepath = 's0615_Saline_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 15:
        path = mainpath + 's0514_2'
        behav_data = list()
        raw_filepath = 's0514_2_Saline_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 16:
        path = mainpath + 's1005_2_saline'
        behav_data = list()
        raw_filepath = 's1005_2_Saline_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 17:
        path = mainpath + 's1005_1_saline'
        behav_data = list()
        raw_filepath = 's1005_1_Saline_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('009000.csv')
        
    elif session == 18:
        path = mainpath + 's0104_5m_saline'
        behav_data = list()
        raw_filepath = 's0104_Saline_before5m_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 19:
        path = mainpath + 's1011_2_saline'
        behav_data = list()
        raw_filepath = 's1011_2_Saline_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('008000.csv')
        
    elif session == 20:
        path = mainpath + 's0803_1_1%'
        behav_data = list()
        raw_filepath = 'S_0803_1_Formalin1%_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
        
    elif session == 21:
        path = mainpath + 's0803_2_1%'
        behav_data = list()
        raw_filepath = 's0803_2_Formalin1_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 22:
        path = mainpath + 's0816_1%'
        behav_data = list()
        raw_filepath = 's0816_Formalin1%_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 23:
        path = mainpath + 's1012_1%'
        behav_data = list()
        raw_filepath = 's1012_Formalin1_awake(1).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('008000.csv')
        
    elif session == 24:
        path = mainpath + 's0829_1%'
        behav_data = list()
        raw_filepath = 's0829_Formalin 1%_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 25:
        path = mainpath + 's0404_3_1%'
        behav_data = list()
        raw_filepath = 's0404_3_1% Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 26:
        path = mainpath + 's0405_1_1%'
        behav_data = list()
        raw_filepath = 's0405_1_1% Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('007000.csv')
        
    elif session == 27:
        path = mainpath + 's0129_restricted'
        behav_data = list()
        raw_filepath = 's0129_restricted.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        
    elif session == 28:
        path = mainpath + 's0130_1_restricted'
        behav_data = list()
        raw_filepath = 's0130_1_restricted.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        behav_data.append('001000.csv')
        
    elif session == 29:
        path = mainpath + 's0419_2_Restricted'
        behav_data = list()
        raw_filepath = 's0419_2_Restricted.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv') # 8번 누락인데, 서로 안움직인것으로 비슷하여 7번으로 대체함 .
        
    elif session == 30:
        path = mainpath + 's0419_1_retricted'
        behav_data = list()
        raw_filepath = 's0419_1_retricted.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 31: ##
        path = mainpath + 's0711_Formalin25_awake'
        behav_data = list()
        raw_filepath = 's0711_Formalin25_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 32:
        path = mainpath + 's0718_1_Formalin25_awake'
        behav_data = list()
        raw_filepath = 's0718_1_Formalin25_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 33:
        path = mainpath + 'S0823_1_Formalin0.25_awake'
        behav_data = list()
        raw_filepath = 'S0823_1_Formalin0.25_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 34:
        path = mainpath + 's0823_2_Formalin25_awake'
        behav_data = list()
        raw_filepath = 's0823_2_Formalin25_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 35:
        path = mainpath + 's0906_2_Formalin25_awake'
        behav_data = list()
        raw_filepath = 's0906_2_Formalin25_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 36:
        path = mainpath + 's0405_2_0.25%_Formalin'
        behav_data = list()
        raw_filepath = 's0405_2_0.25%_Formalin.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 37:
        path = mainpath + 's0411_0.25%_Formalin'
        behav_data = list()
        raw_filepath = 's0411_0.25%_Formalin.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 38:
        path = mainpath + 's0412_0.25%_Formalin'
        behav_data = list()
        raw_filepath = 's0412_0.25%_Formalin.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 39:
        path = mainpath + 's0109_1_ketoprofen100_before5min'
        behav_data = list()
        raw_filepath = 's0109_1_ketoprofen100_before5min.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        
    elif session == 40:
        path = mainpath + 's0109_2_ketoprofen100_before5min'
        behav_data = list()
        raw_filepath = 's0109_2_ketoprofen100_before5min.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('007000.csv')
        
    elif session == 41:
        path = mainpath + 's0110_1_Keto100_before5m_awake'
        behav_data = list()
        raw_filepath = 's0110_1_Keto100_before5m_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('007000.csv')
        
    elif session == 42:
        path = mainpath + 's0117_2_keto100_before5min'
        behav_data = list()
        raw_filepath = 's0117_2_keto100_before5min.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        
    elif session == 43:
        path = mainpath + 's0516_2_Restricted'
        behav_data = list()
        raw_filepath = 's0516_2_Restricted.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 44:
        path = mainpath + 's0517_1_Restricted'
        behav_data = list()
        raw_filepath = 's0517_1_Restricted.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        
    elif session == 45:
        path = mainpath + 's0517_2_Restricted'
        behav_data = list()
        raw_filepath = 's0517_2_Restricted.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 46:
        path = mainpath + 's0214_1_Keto 100'
        behav_data = list()
        raw_filepath = 's0214_1_Keto 100.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('010000.csv')
        
    elif session == 47:
        path = mainpath + 's0620_1_Saline_movement'
        behav_data = list()
        raw_filepath = 's0620_1_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 48:
        path = mainpath + 's0620_2_Saline_movement'
        behav_data = list()
        raw_filepath = 's0620_2_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')

    elif session == 49:
        path = mainpath + 's0214_2_Keto 100'
        behav_data = list()
        raw_filepath = 's0214_2_Keto 100.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('008000.csv')
        
    elif session == 50:
        path = mainpath + 's0215_1_Keto 100'
        behav_data = list()
        raw_filepath = 's0215_1_Keto 100.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('007000.csv')
        behav_data.append('010000.csv')
        
    elif session == 51:
        path = mainpath + 's0321_lidocaine'
        behav_data = list()
        raw_filepath = 's0321_lidocaine.xlsx'
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('008000.csv')
        behav_data.append('011000.csv')
        
    elif session == 52:
        path = mainpath + 's0627_2_Saline_movement'
        behav_data = list()
        raw_filepath = 's0627_2_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 53:
        path = mainpath + 's0628_1_Saline_movement'
        behav_data = list()
        raw_filepath = 's0628_1_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 54: 
        path = mainpath + 's0424_2_lidocaine'
        behav_data = list()
        raw_filepath = 's0424_2_lidocaine.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('009000.csv')
        behav_data.append('011000.csv')
        
    elif session == 55: 
        path = mainpath + 's0425_1_lidocaine'
        behav_data = list()
        raw_filepath = 's0425_1_lidocaine.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 56: 
        path = mainpath + 's0703_2_Saline_movement'
        behav_data = list()
        raw_filepath = 's0703_2_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 57: 
        path = mainpath + 's0404_2_1% Formalin_awake'
        behav_data = list()
        raw_filepath = 's0404_2_1% Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 58: 
        path = mainpath + 's0705_1_Saline_movement'
        behav_data = list()
        raw_filepath = 's0705_1_Saline_movement.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 59: 
        path = mainpath + 's0904_Formalin_awake'
        behav_data = list()
        raw_filepath = 's0904_Formalin_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('009000.csv')
        
    elif session == 60: 
        path = mainpath + 's0710_1_Cap_awake'
        behav_data = list()
        raw_filepath = 's0710_1_Cap_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 61: 
        path = mainpath + 's0718_2_Cap_awake'
        behav_data = list()
        raw_filepath = 's0718_2_Cap_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 62: 
        path = mainpath + 's0718_1_Cap_awake'
        behav_data = list()
        raw_filepath = 's0718_1_Cap_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 63: 
        path = mainpath + 's0726_3_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0726_3_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 64: 
        path = mainpath + 's0726_1_Cap_awake'
        behav_data = list()
        raw_filepath = 's0726_1_Cap_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 65: 
        path = mainpath + 's0731_2_Cap_awake'
        behav_data = list()
        raw_filepath = 's0731_2_Cap_awake.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        
    elif session == 66: 
        path = mainpath + 's0731_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0731_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')

    elif session == 67: 
        path = mainpath + 's0808_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0808_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 68: 
        path = mainpath + 's0809_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0809_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 69: 
        path = mainpath + 's0814_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0814_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        
    elif session == 69: 
        path = mainpath + 's0814_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0814_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        
    elif session == 69: 
        path = mainpath + 's0814_1_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0814_1_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('006000.csv')
        
    elif session == 70: 
        path = mainpath + 's0829_PSL'
        behav_data = list()
        raw_filepath = 's0829_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 71: 
        path = mainpath + 's0903_PSL'
        behav_data = list()
        raw_filepath = 's0903_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 72: pass
        
    elif session == 73: 
        path = mainpath + 's0906_1_PSL'
        behav_data = list()
        raw_filepath = 's0906_1_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        
    elif session == 74: 
        path = mainpath + 's0814_2_Yohimbine_Formalin'
        behav_data = list()
        raw_filepath = 's0814_2_Yohimbine_Formalin.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 75: 
        path = mainpath + 's0906_2_PSL'
        behav_data = list()
        raw_filepath = 's0906_2_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    ## 20191022
    
    elif session == 76: 
        path = mainpath + 's0909_1_PSL'
        behav_data = list()
        raw_filepath = 's0909_1_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 77: 
        path = mainpath + 's0910_PSL'
        behav_data = list()
        raw_filepath = 's0910_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        
    elif session == 78: 
        path = mainpath + 's0918_1_PSL'
        behav_data = list()
        raw_filepath = 's0918_1_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        
    elif session == 79: 
        path = mainpath + 's0918_2_PSL'
        behav_data = list()
        raw_filepath = 's0918_2_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        
    elif session == 80: 
        path = mainpath + 's0919_PSL'
        behav_data = list()
        raw_filepath = 's0919_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        
    elif session == 81: 
        path = mainpath + 's0926_Sham'
        behav_data = list()
        raw_filepath = 's0926_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 82: 
        path = mainpath + 's0802_1_Cap'
        behav_data = list()
        raw_filepath = 's0802_1_Cap.xlsx'
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('004000.csv')
        behav_data.append('004000.csv')
        
    elif session == 83: 
        path = mainpath + 's0802_2_Cap'
        behav_data = list()
        raw_filepath = 's0802_2_Cap.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 84: pass
        
    elif session == 85: pass
        
    elif session == 86: pass
        
    elif session == 87: pass
        
    elif session == 88: pass
        
    elif session == 89: 
        path = mainpath + 's1011_1_Sham'
        behav_data = list()
        raw_filepath = 's1011_1_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
        ##
        
    elif session == 90: 
        path = mainpath + 's1029_1_Sham'
        behav_data = list()
        raw_filepath = 's1029_1_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 91: 
        path = mainpath + 's1029_2_Sham'
        behav_data = list()
        raw_filepath = 's1029_2_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 92: 
        path = mainpath + 's1011_2_Sham'
        behav_data = list()
        raw_filepath = 's1011_2_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 93: 
        path = mainpath + 's0918_1_PSL_2'
        behav_data = list()
        raw_filepath = 's0918_1_PSL_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 94: 
        path = mainpath + 's0919_PSL_2'
        behav_data = list()
        raw_filepath = 's0919_PSL_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 95: 
        path = mainpath + 's1220_2_Formalin_before5min_awake'
        behav_data = list()
        raw_filepath = 's1220_2_Formalin_before5min_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 96: 
        path = mainpath + 's1226_1_Formalin_before5min_awake'
        behav_data = list()
        raw_filepath = 's1226_1_Formalin_before5min_awake.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 97: 
        path = mainpath + 's1105_Sham'
        behav_data = list()
        raw_filepath = 's1105_Sham.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 98: 
        path = mainpath + 's1112_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1112_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 99: 
        path = mainpath + 's1114_1_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1114_1_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
        ##
        
    elif session == 100: 
        path = mainpath + 's1114_2_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1114_2_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 101: 
        path = mainpath + 's1115_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1115_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 102: 
        path = mainpath + 's1119_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1119_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 103: 
        path = mainpath + 's1126_PSL_Adenosine'
        behav_data = list()
        raw_filepath = 's1126_PSL_Adenosine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 104: 
        path = mainpath + 's1204_1_Cap'
        behav_data = list()
        raw_filepath = 's1204_1_Cap.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 105: 
        path = mainpath + 's1204_2_Cap'
        behav_data = list()
        raw_filepath = 's1204_2_Cap.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        ##
    elif session == 106: 
        path = mainpath + 's1211_CFA'
        behav_data = list()
        raw_filepath = 's1211_CFA.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 107: 
        path = mainpath + 's1212_1_CFA'
        behav_data = list()
        raw_filepath = 's1212_1_CFA.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 108: 
        path = mainpath + 's1212_2_CFA'
        behav_data = list()
        raw_filepath = 's1212_2_CFA.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 109: 
        path = mainpath + 's1218_CFA'
        behav_data = list()
        raw_filepath = 's1218_CFA.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
        ##
        
    elif session == 110: 
        path = mainpath + 's1112_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1112_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 111: 
        path = mainpath + 's1114_1_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1114_1_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 112: 
        path = mainpath + 's1114_2_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1114_2_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 113: 
        path = mainpath + 's1115_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1115_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 114: 
        path = mainpath + 's1119_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1119_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 115: 
        path = mainpath + 's1126_PSL_Adenosine_2'
        behav_data = list()
        raw_filepath = 's1126_PSL_Adenosine_2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 116: 
        path = mainpath + 's0107_Cfa'
        behav_data = list()
        raw_filepath = 's0107_Cfa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 117: 
        path = mainpath + 's0109_Cfa'
        behav_data = list()
        raw_filepath = 's0109_Cfa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv') ##
        
    elif session == 118: 
        path = mainpath + 's0115_Chloroquine'
        behav_data = list()
        raw_filepath = 's0115_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')        
        
    elif session == 119: 
        path = mainpath + 's0116_Chloroquine'
        behav_data = list()
        raw_filepath = 's0116_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')        
        
    elif session == 120: 
        path = mainpath + 's0117_Chloroquine'
        behav_data = list()
        raw_filepath = 's0117_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

    elif session == 121: 
        path = mainpath + 's0122_Chloroquine'
        behav_data = list()
        raw_filepath = 's0122_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')        
        
    elif session == 122: 
        path = mainpath + 's0130_Chloroquine'
        behav_data = list()
        raw_filepath = 's0130_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')        
        
    elif session == 123: 
        path = mainpath + 's0130_2_Chloroquine'
        behav_data = list()
        raw_filepath = 's0130_2_Chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 124: 
        path = mainpath + 's0205_1_chloroquine'
        behav_data = list()
        raw_filepath = 's0205_1_chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 125: 
        path = mainpath + 's0205_2_chloroquine'
        behav_data = list()
        raw_filepath = 's0205_2_chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 126: 
        path = mainpath + 's0206_1_chloroquine'
        behav_data = list()
        raw_filepath = 's0206_1_chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 127: 
        path = mainpath + 's0206_2_chloroquine'
        behav_data = list()
        raw_filepath = 's0206_2_chloroquine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
       
        ### 20200324
        
    elif session == 128: 
        path = mainpath + 's0214_1_PSL_itSaline'
        behav_data = list()
        raw_filepath = 's0214_1_PSL_itSaline.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 129: 
        path = mainpath + 's0214_2_PSL_itSaline'
        behav_data = list()
        raw_filepath = 's0214_2_PSL_itSaline.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 130: 
        path = mainpath + 's0228_PSL_itSaline'
        behav_data = list()
        raw_filepath = 's0228_PSL_itSaline.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 131: 
        path = mainpath + 's0215_2_PSL_clonidine'
        behav_data = list()
        raw_filepath = 's0215_2_PSL_clonidine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 132: 
        path = mainpath + 's0215_PSL_clonidine_saline'
        behav_data = list()
        raw_filepath = 's0215_PSL_clonidine_saline.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 133: 
        path = mainpath + 's0218_PSL_clonidine'
        behav_data = list()
        raw_filepath = 's0218_PSL_clonidine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        ##
    elif session == 134: 
        path = mainpath + 's0304_PSL_i.t.Saline'
        behav_data = list()
        raw_filepath = 's0304_PSL_i.t.Saline.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

    elif session == 135: 
        path = mainpath + 's0304_PSL_i.t.Saline_t2'
        behav_data = list()
        raw_filepath = 's0304_PSL_i.t.Saline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

    elif session == 136: 
        path = mainpath + 's0305_PSL_i.t.clonidine'
        behav_data = list()
        raw_filepath = 's0305_PSL_i.t.clonidine.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

    elif session == 137: 
        path = mainpath + 's0305_PSL_clonidine_t2'
        behav_data = list()
        raw_filepath = 's0305_PSL_clonidine_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')       
        
    elif session == 138: 
        path = mainpath + 's0214_1_PSL_i.t.Saline_t2'
        behav_data = list()
        raw_filepath = 's0214_1_PSL_i.t.Saline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

    elif session == 139: 
        path = mainpath + 's0228_PSL_i.t.Saline_t2'
        behav_data = list()
        raw_filepath = 's0228_PSL_i.t.Saline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 140: 
        path = mainpath + 's0214_2_PSL_Saline_t2'
        behav_data = list()
        raw_filepath = 's0214_2_PSL_Saline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        ##
    elif session == 141: 
        path = mainpath + 's0312_PSL_Saline(i.p.)'
        behav_data = list()
        raw_filepath = 's0312_PSL_Saline(i.p.).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')    
        
    elif session == 142: 
        path = mainpath + 's0313_1_PSL_Saline(i.p.)'
        behav_data = list()
        raw_filepath = 's0313_1_PSL_Saline(i.p.).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')       
        
    elif session == 143: 
        path = mainpath + 's0313_2_PSL_Saline(i.p.)'
        behav_data = list()
        raw_filepath = 's0313_2_PSL_Saline(i.p.).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')  
        
    elif session == 144: 
        path = mainpath + 's0318_1_PSL_Saline(ip)'
        behav_data = list()
        raw_filepath = 's0318_1_PSL_Saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 145: 
        path = mainpath + 's0319_1_PSL_Saline(ip)'
        behav_data = list()
        raw_filepath = 's0319_1_PSL_Saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 146: 
        path = mainpath + 's0319_2_PSL_Saline(ip)'
        behav_data = list()
        raw_filepath = 's0319_2_PSL_Saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv') 
        behav_data.append('006000.csv') 
        
    elif session == 147: pass
        
    elif session == 148: pass
        
    elif session == 149: pass

    elif session == 150: 
        path = mainpath + 's0324_1_PSL_Saline(ip)'
        behav_data = list()
        raw_filepath = 's0324_1_PSL_Saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
        ###
    elif session == 151: 
        path = mainpath + 's0324_2_PSL_Clonidine(ip)'
        behav_data = list()
        raw_filepath = 's0324_2_PSL_Clonidine(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 152: 
        path = mainpath + 's0326_1_PSL_saline(ip)'
        behav_data = list()
        raw_filepath = 's0326_1_PSL_saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 153: 
        path = mainpath + 's0326_2_PSL_clonidine(ip)'
        behav_data = list()
        raw_filepath = 's0326_2_PSL_clonidine(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        ##
    elif session == 154: pass
        
    elif session == 155: pass
        
    elif session == 156: pass
        
    elif session == 157: pass
        
    elif session == 158: 
        path = mainpath + 's0402_PSL_saline(ip)'
        behav_data = list()
        raw_filepath = 's0402_PSL_saline(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv') 
        
    elif session == 159: pass
        
    elif session == 160: pass
        
    elif session == 161: 
        path = mainpath + 's0401_2_PSL_clonidine(ip)'
        behav_data = list()
        raw_filepath = 's0401_2_PSL_clonidine(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 162: 
        path = mainpath + 's0401_1_PSL_Clonidine(ip)'
        behav_data = list()
        raw_filepath = 's0401_1_PSL_Clonidine(ip).xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        
    elif session == 163: pass
        
    elif session == 164: 
        path = mainpath + 's0407_1_PSL_GB VX'
        behav_data = list()
        raw_filepath = 's0407_1_PSL_GB VX.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 165: 
        path = mainpath + 's0407_2_PSL_GB VX'
        behav_data = list()
        raw_filepath = 's0407_2_PSL_GB VX.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('004000.csv') 
        
    elif session == 166: 
        path = mainpath + 's0409_PSL_GB VX'
        behav_data = list()
        raw_filepath = 's0409_PSL_GB VX.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')  
        
    elif session == 167: 
        path = mainpath + 's0416_PSL_lido_GB VX'
        behav_data = list()
        raw_filepath = 's0416_PSL_lido_GB VX.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv')
        
    elif session == 168: pass
        
        # path = mainpath + 's0416_PSL_lido_GB VX_t2'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')
        
    elif session == 169: pass # merged with 164
        # path = mainpath + 's0407_1_PSL_GB VX_lidoaine'
        # behav_data = list()
        # raw_filepath = 's0407_1_PSL_GB VX_lidoaine.xlsx'   
        
    elif session == 170: pass
        # path = mainpath + 's0409_PSL_GB VX_lidocaine'
        # behav_data = list()
        # raw_filepath = 's0409_PSL_GB VX_lidocaine.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv') 
        
    elif session == 171: pass
        # path = mainpath + 's0416_PSL_GB VX_lidocaine'
        # behav_data = list()
        # raw_filepath = 's0416_PSL_GB VX_lidocaine.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv') 
        
    elif session == 172: 
        path = mainpath + 's0429_1_PSL_D2_GB VX_lidocaine_t1'
        behav_data = list()
        raw_filepath = 's0429_1_PSL_D2_GB VX_lidocaine_t1.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv')
        
    elif session == 173: pass
        # path = mainpath + 's0429_1_PSL_D2_GB VX_lidocaine_t2'
        # behav_data = list()
        # raw_filepath = 's0429_1_PSL_D2_GB VX_lidocaine_t2.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv')      
        
    elif session == 174: 
        path = mainpath + 's0429_2_PSL_D3_GB VX_lidocaine_t1'
        behav_data = list()
        raw_filepath = 's0429_2_PSL_D3_GB VX_lidocaine_t1.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv') 
        
    elif session == 175: pass
        # path = mainpath + 's0429_2_PSL_D3_GB VX_lidocaine_t2'
        # behav_data = list()
        # raw_filepath = 's0429_2_PSL_D3_GB VX_lidocaine_t2.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv') 
        
        ##
    elif session == 176: pass
        # path = mainpath + 's0429_1_PSL_D10_GB VX_lidocaine'
        # behav_data = list()
        # raw_filepath = 's0429_1_PSL_D10_GB VX_lidocaine.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        ##
    elif session == 177: 
        path = mainpath + 's0514_1_PSL_GB VX_D3'
        behav_data = list()
        raw_filepath = 's0514_1_PSL_GB VX_D3.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')  
        
    elif session == 178: pass
        # path = mainpath + 's0514_1_PSL_GB VX_D3_t2'
        # behav_data = list()
        # raw_filepath = 's0514_1_PSL_GB VX_D3_t2.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')
        
    elif session == 179: 
        path = mainpath + 's0514_2_PSL_GB VX_D3'
        behav_data = list()
        raw_filepath = 's0514_2_PSL_GB VX_D3.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        behav_data.append('011000.csv')
        behav_data.append('012000.csv')
 
        ##
    elif session == 180: pass
        # path = mainpath + 's0514_2_PSL_GB VX_D3_t2'
        # behav_data = list()
        # raw_filepath = 's0514_2_PSL_GB VX_D3_t2.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('003000.csv')

    elif session == 181: 
        path = mainpath + 's0515_1_PSL_GB VX_D3'
        behav_data = list()
        raw_filepath = 's0515_1_PSL_GB VX_D3.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv') 
        
    elif session == 182: 
        path = mainpath + 's0515_2_PSL_GB VX_D3'
        behav_data = list()
        raw_filepath = 's0515_2_PSL_GB VX_D3.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 183: pass
        # path = mainpath + 's0514_1_PSL_GB VX_D10'
        # behav_data = list()
        # raw_filepath = 's0514_1_PSL_GB VX_D10.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        
    elif session == 184: pass
        # path = mainpath + 's0514_2_PSL_GB VX_D10'
        # behav_data = list()
        # raw_filepath = 's0514_2_PSL_GB VX_D10.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('002000.csv')
        
    elif session == 185: pass
        # path = mainpath + 's0514_2_PSL_GB VX_D20'
        # behav_data = list()
        # raw_filepath = 's0514_2_PSL_GB VX_D20.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv')
        
    elif session == 186: pass
        # path = mainpath + 's0515_1_PSL_GB VX_D20'
        # behav_data = list()
        # raw_filepath = 's0515_1_PSL_GB VX_D20.xlsx'
        # behav_data.append('001000.csv')
        # behav_data.append('002000.csv')
        # behav_data.append('003000.csv')
        # behav_data.append('004000.csv')
        # behav_data.append('004000.csv')
        
    elif session == 187: 
        path = mainpath + 's0429_1_PSL_D13_BV'
        behav_data = list()
        raw_filepath = 's0429_1_PSL_D13_BV.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        ## 20200825
    elif session == 188: 
        path = mainpath + 's0716_Oxa'
        behav_data = list()
        raw_filepath = 's0716_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('empty')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 189:  pass

    elif session == 190: 
        path = mainpath + 's0717_Oxa'
        behav_data = list()
        raw_filepath = 's0717_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        
    elif session == 191: pass

    elif session == 192: 
        path = mainpath + 's0721_1_Oxa'
        behav_data = list()
        raw_filepath = 's0721_1_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 193:  pass

    elif session == 194: 
        path = mainpath + 's0721_2_Oxa'
        behav_data = list()
        raw_filepath = 's0721_2_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 195: pass

        
    elif session == 196: 
        path = mainpath + 's0722_2_Oxa'
        behav_data = list()
        raw_filepath = 's0722_2_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 197: pass
        
    elif session == 198: 
        path = mainpath + 's0723_Oxa'
        behav_data = list()
        raw_filepath = 's0723_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 199: pass

        ##
    elif session == 200: 
        path = mainpath + 's0731_1_Oxa'
        behav_data = list()
        raw_filepath = 's0731_1_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 201: pass
        
    elif session == 202: 
        path = mainpath + 's0828_1_Oxa'
        behav_data = list()
        raw_filepath = 's0828_1_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 203: pass

    elif session == 204: 
        path = mainpath + 's0728_Glucose'
        behav_data = list()
        raw_filepath = 's0728_Glucose.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        
    elif session == 205: pass

    elif session == 206: 
        path = mainpath + 's0827_1_Glucose'
        behav_data = list()
        raw_filepath = 's0827_1_Glucose.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 207: pass     

    elif session == 208: 
        path = mainpath + 's0828_2_Glu'
        behav_data = list()
        raw_filepath = 's0828_2_Glu.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')

    elif session == 209: pass
        
    elif session == 210: 
        path = mainpath + 's0901_1_Glu'
        behav_data = list()
        raw_filepath = 's0901_1_Glu.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')

    elif session == 211: pass
        
    elif session == 212: 
        path = mainpath + 's0901_2_Glu'
        behav_data = list()
        raw_filepath = 's0901_2_Glu.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 213: pass

    elif session == 214: 
        path = mainpath + 's0806_Glu'
        behav_data = list()
        raw_filepath = 's0806_Glu.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 215: pass
        
    elif session == 216: 
        path = mainpath + 's0902_2_PSL_scSaline_t1'
        behav_data = list()
        raw_filepath = 's0902_2_PSL_scSaline_t1.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv') 
        
    elif session == 217: 
        path = mainpath + 's0902_2_PSL_scSaline_t2'
        behav_data = list()
        raw_filepath = 's0902_2_PSL_scSaline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv') 
        
    elif session == 218: 
        path = mainpath + 's0908_1_PSL_scSaline_t1'
        behav_data = list()
        raw_filepath = 's0908_1_PSL_scSaline_t1.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv') 
        
    elif session == 219: 
        path = mainpath + 's0908_1_PSL_scSaline_t2'
        behav_data = list()
        raw_filepath = 's0908_1_PSL_scSaline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')

##
    elif session == 220: 
        path = mainpath + 's0917_1_Oxa'
        behav_data = list()
        raw_filepath = 's0917_1_Oxa.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 221: pass
        
    elif session == 222: 
        path = mainpath + 's0917_2_Glu'
        behav_data = list()
        raw_filepath = 's0917_2_Glu.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        
    elif session == 223: pass
        
    elif session == 224: 
        path = mainpath + 's0918_2_PSL_Saline_t1'
        behav_data = list()
        raw_filepath = 's0918_2_PSL_Saline_t1.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv') 
        
    elif session == 225: 
        path = mainpath + 's0918_2_PSL_Saline_t2'
        behav_data = list()
        raw_filepath = 's0918_2_PSL_Saline_t2.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        behav_data.append('003000.csv')
        
    elif session == 226: 
        path = mainpath + 's1029_PSL'
        behav_data = list()
        raw_filepath = 's1029_PSL.xlsx'
        behav_data.append('001000.csv')
        behav_data.append('002000.csv')
        behav_data.append('003000.csv')
        behav_data.append('004000.csv')
        behav_data.append('005000.csv')
        behav_data.append('006000.csv')
        behav_data.append('007000.csv')
        behav_data.append('008000.csv')
        behav_data.append('009000.csv')
        behav_data.append('010000.csv') 
        behav_data.append('011000.csv')
        behav_data.append('012000.csv')
        behav_data.append('013000.csv')
        behav_data.append('014000.csv')
        
    elif session == 227: pass
    elif session == 228: pass
    elif session == 229: pass
        
    elif session == 230: 
        path = mainpath + 's201201_1_5%F'
        behav_data = list()
        raw_filepath = 's201201_1_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        ##
    elif session == 231: 
        path = mainpath + 's201127_1_5%F'
        behav_data = list()
        raw_filepath = 's201127_1_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 232: 
        path = mainpath + 's201127_2_5%F'
        behav_data = list()
        raw_filepath = 's201127_2_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 233: 
        path = mainpath + 's201130_1_5%F'
        behav_data = list()
        raw_filepath = 's201130_1_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 234: 
        path = mainpath + 's201130_2_5%F'
        behav_data = list()
        raw_filepath = 's201130_2_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 235: 
        path = mainpath + 's201201_2_5%F'
        behav_data = list()
        raw_filepath = 's201201_2_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 236: 
        path = mainpath + 's201202_5%F'
        behav_data = list()
        raw_filepath = 's201202_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty') 
        
    elif session == 237: 
        path = mainpath + 's201203_1_5%F'
        behav_data = list()
        raw_filepath = 's201203_1_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 238: 
        path = mainpath + 's201203_2_5%F'
        behav_data = list()
        raw_filepath = 's201203_2_5%F.xlsx'
        for se in range(2):        
            behav_data.append('empty')
        
    elif session == 239: 
        path = mainpath + 's210112_1_PSL'
        behav_data = list()
        raw_filepath = 's210112_1_PSL.xlsx'
        behav_data.append('[SHANA]s210112_1_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210112_1_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210112_1_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210112_1_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210112_1_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210112_1_PSL_5.avi.mat')
 
    elif session == 240: pass
        # path = mainpath + 's210112_1_PSL_t2'
        # behav_data = list()
        # raw_filepath = 's210112_1_PSL_t2.xlsx'
        
        
    elif session == 241: 
        path = mainpath + 's210112_2_PSL'
        behav_data = list()
        raw_filepath = 's210112_2_PSL.xlsx'
        behav_data.append('[SHANA]s210112_2_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210112_2_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210112_2_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210112_2_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210112_2_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210112_2_PSL_5.avi.mat')
        
    elif session == 242:  pass
        # path = mainpath + 's210112_2_PSL_t2'
        # behav_data = list()
        # raw_filepath = 's210112_2_PSL_t2.xlsx'

        
    elif session == 243: 
        path = mainpath + 's210115_PSL'
        behav_data = list()
        raw_filepath = 's210115_PSL.xlsx'
        behav_data.append('[SHANA]s210115_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210115_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210115_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210115_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210115_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210115_PSL_5.avi.mat')
 
    elif session == 244: pass
        # path = mainpath + 's210115_PSL_t2'
        # behav_data = list()
        # raw_filepath = 's210115_PSL_t2.xlsx'
        
    elif session == 245: 
        path = mainpath + 's210128_PSL'
        behav_data = list()
        raw_filepath = 's210128_PSL.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('[SHANA]s210128_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210128_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210128_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210128_PSL_5.avi.mat')
 
    elif session == 246: pass
        # path = mainpath + 's210128_PSL_t2'
        # behav_data = list()
        # raw_filepath = 's210128_PSL_t2.xlsx'
        # behav_data.append('empty')

    elif session == 247: 
        path = mainpath + 's210324_1_SF'
        behav_data = list()
        raw_filepath = 's210324_1_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
  
    elif session == 248: 
        path = mainpath + 's210324_2_SF'
        behav_data = list()
        raw_filepath = 's210324_2_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 249: 
        path = mainpath + 's210326_S'
        behav_data = list()
        raw_filepath = 's210326_S.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')

    elif session == 250: 
        path = mainpath + 's210330_SF'
        behav_data = list()
        raw_filepath = 's210330_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 251: 
        path = mainpath + 's210331_SF'
        behav_data = list()
        raw_filepath = 's210331_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 252: 
        path = mainpath + 's210408_F'
        behav_data = list()
        raw_filepath = 's210408_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 253: 
        path = mainpath + 's210408_1_H_F'
        behav_data = list()
        raw_filepath = 's210408_1_H_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 254: 
        path = mainpath + 's210408_2_H_F'
        behav_data = list()
        raw_filepath = 's210408_2_H_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')

    elif session == 255: 
        path = mainpath + 's210409_H_S'
        behav_data = list()
        raw_filepath = 's210409_H_S.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        
    elif session == 256: 
        path = mainpath + 's210413_1_H_F'
        behav_data = list()
        raw_filepath = 's210413_1_H_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 257: 
        path = mainpath + 's210414_1_H_SF'
        behav_data = list()
        raw_filepath = 's210414_1_H_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 258: 
        path = mainpath + 's210414_2_H_SF'
        behav_data = list()
        raw_filepath = 's210414_2_H_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 259: 
        path = mainpath + 's210415_1_H_SF'
        behav_data = list()
        raw_filepath = 's210415_1_H_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')

    elif session == 260: 
        path = mainpath + 's210415_2_F'
        behav_data = list()
        raw_filepath = 's210415_2_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 261: 
        path = mainpath + 's210416_F'
        behav_data = list()
        raw_filepath = 's210416_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 262: 
        path = mainpath + 's210420_1_H_SF'
        behav_data = list()
        raw_filepath = 's210420_1_H_SF.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        behav_data.append(name + 'B3.avi.mat')
        behav_data.append(name + 'B4.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 263: 
        path = mainpath + 's210420_1_S'
        behav_data = list()
        raw_filepath = 's210420_1_S.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')
        
    elif session == 264: 
        path = mainpath + 's210420_2_S'
        behav_data = list()
        raw_filepath = 's210420_2_S.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'S1.avi.mat')

    elif session == 265: 
        path = mainpath + 's210422_1_F'
        behav_data = list()
        raw_filepath = 's210422_1_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 266: 
        path = mainpath + 's210422_2_F'
        behav_data = list()
        raw_filepath = 's210422_2_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
        
    elif session == 267: 
        path = mainpath + 's210427_1_F'
        behav_data = list()
        raw_filepath = 's210427_1_F.xlsx'
        name = '[SHANA]' + raw_filepath[:-5] + '_'
        behav_data.append(name + 'B1.avi.mat')
        behav_data.append(name + 'B2.avi.mat')
        behav_data.append(name + 'F1.avi.mat')
                                            ##
    elif session == 268: 
        path = mainpath + 's210415_2_H_S'
        behav_data = list()
        raw_filepath = 's210415_2_H_S.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        
    elif session == 269: 
        path = mainpath + 's210427_2_F'
        behav_data = list()
        raw_filepath = 's210427_2_F.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')

    elif session == 270: 
        path = mainpath + 's210427_H_S'
        behav_data = list()
        raw_filepath = 's210427_H_S.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        
    elif session == 271: 
        path = mainpath + 's210428_1_H_S'
        behav_data = list()
        raw_filepath = 's210428_1_H_S.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        
    elif session == 272: 
        path = mainpath + 's210428_F'
        behav_data = list()
        raw_filepath = 's210428_F.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        
    elif session == 273: 
        path = mainpath + 's210506_H_PSL_M10mg'
        behav_data = list()
        raw_filepath = 's210506_H_PSL_M10mg.xlsx'
        behav_data.append('[SHANA]s210506_H_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_5.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_6.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_7.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_8.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_9.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_10.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_11.avi.mat')
        behav_data.append('[SHANA]s210506_H_PSL_12.avi.mat')
 
    elif session == 274: 
        path = mainpath + 's210507_H_PSL_M10mg'
        behav_data = list()
        raw_filepath = 's210507_H_PSL_M10mg.xlsx'
        behav_data.append('[SHANA]s210507_H_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_5.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_6.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_7.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_8.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_9.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_10.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_11.avi.mat')
        behav_data.append('[SHANA]s210507_H_PSL_12.avi.mat')
#
    elif session == 275: 
        path = mainpath + 's210510_H_PSL_M10'
        behav_data = list()
        raw_filepath = 's210510_H_PSL_M10.xlsx'
        behav_data.append('[SHANA]s210510_H_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_5.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_6.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_7.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_8.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_9.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_10.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_11.avi.mat')
        behav_data.append('[SHANA]s210510_H_PSL_12.avi.mat')

    elif session == 276: 
        path = mainpath + 's210511_H_PSL'
        behav_data = list()
        raw_filepath = 's210511_H_PSL.xlsx'
        behav_data.append('[SHANA]s210511_H_PSL_0.avi.mat')
        behav_data.append('[SHANA]s210511_H_PSL_1.avi.mat')
        behav_data.append('[SHANA]s210511_H_PSL_2.avi.mat')
        behav_data.append('[SHANA]s210511_H_PSL_3.avi.mat')
        behav_data.append('[SHANA]s210511_H_PSL_4.avi.mat')
        behav_data.append('[SHANA]s210511_H_PSL_5.avi.mat')

    elif session == 277: 
        path = mainpath + 's210518_1_H_PSL_M10'
        behav_data = list()
        raw_filepath = 's210518_1_H_PSL_M10.xlsx'
        behav_data.append('s210518_1_H_PSL_0.avi.mat')
        behav_data.append('s210518_1_H_PSL_1.avi.mat')
        behav_data.append('s210518_1_H_PSL_2.avi.mat')
        behav_data.append('s210518_1_H_PSL_3.avi.mat')
        behav_data.append('s210518_1_H_PSL_4.avi.mat')
        behav_data.append('s210518_1_H_PSL_5.avi.mat')
        behav_data.append('s210518_1_H_PSL_6.avi.mat')
        behav_data.append('s210518_1_H_PSL_7.avi.mat')
        behav_data.append('s210518_1_H_PSL_8.avi.mat')
        behav_data.append('s210518_1_H_PSL_9.avi.mat')
        behav_data.append('s210518_1_H_PSL_10.avi.mat')
        behav_data.append('s210518_1_H_PSL_11.avi.mat')
        
    elif session == 278: 
        path = mainpath + 's201229 MPTP_5.13Hz_512x512'
        behav_data = list()
        raw_filepath = 's201229 MPTP_5.13Hz_512x512.xlsx'
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        
    elif session == 279: 
        path = mainpath + 's210202 MPTP_5.13Hz_512x512'
        behav_data = list()
        raw_filepath = 's210202 MPTP_5.13Hz_512x512.xlsx'
        behav_data.append('[SHANA]s210202_behav_0.avi.mat')
        behav_data.append('[SHANA]s210202_behav_1.avi.mat')
        behav_data.append('[SHANA]s210202_behav_2.avi.mat')
        behav_data.append('[SHANA]s210202_behav_3.avi.mat')
        behav_data.append('[SHANA]s210202_behav_4.avi.mat')
        behav_data.append('[SHANA]s210202_behav_5.avi.mat')
        behav_data.append('[SHANA]s210202_behav_6.avi.mat')
        behav_data.append('[SHANA]s210202_behav_7.avi.mat')
        behav_data.append('[SHANA]s210202_behav_8.avi.mat')
        behav_data.append('[SHANA]s210202_behav_9.avi.mat')
        
    elif session == 280: 
        path = mainpath + 's210203 MPTP_5.13Hz_512x512'
        raw_filepath = 's210203 MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210203_behav_' + str(se) + '.avi.mat')

    elif session == 281: 
        path = mainpath + 's210216 MPTP_5.13Hz_512x512'
        raw_filepath = 's210216 MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(6):        
            behav_data.append('[SHANA]s210216_behav_' + str(se) + '.avi.mat')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
        behav_data.append('empty')
            
    elif session == 282: 
        path = mainpath + 's210225_MPTP_5.13Hz_512x512'
        raw_filepath = 's210225_MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210225_behav_' + str(se) + '.avi.mat')
            
    elif session == 283: 
        path = mainpath + 's210226_MPTP_5.13Hz_512x512'
        raw_filepath = 's210226_MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210226_behav_' + str(se) + '.avi.mat')
            
    elif session == 284: 
        path = mainpath + 's210302_MPTP_5.13Hz_512x512'
        raw_filepath = 's210302_MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210302_behav_' + str(se) + '.avi.mat')

    elif session == 285: 
        path = mainpath + 's210405_MPTP_5.13Hz_512x512'
        raw_filepath = 's210405_MPTP_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):
            if se == 4: behav_data.append('empty')
            else: behav_data.append('[SHANA]s210405_behav_' + str(se) + '.avi.mat')
            
    elif session == 286: 
        path = mainpath + 's210308_1_Saline_5.13Hz_512x512'
        raw_filepath = 's210308_1_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210308_1_behav_' + str(se) + '.avi.mat')
            
    elif session == 287: 
        path = mainpath + 's210308_3_Saline_5.13Hz_512x512'
        raw_filepath = 's210308_3_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210308_3_behav_' + str(se) + '.avi.mat')
            
    elif session == 288: 
        path = mainpath + 's210325_1_Saline_5.13Hz_512x512'
        raw_filepath = 's210325_1_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):
            if se == 4: behav_data.append('empty')
            else: behav_data.append('[SHANA]s210325_1_behav_' + str(se) + '.avi.mat')
            #
    elif session == 289: 
        path = mainpath + 's210325_2_Saline_5.13Hz_512x512'
        raw_filepath = 's210325_2_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):  
            behav_data.append('[SHANA]s210325_2_behav_' + str(se) + '.avi.mat')
            
    elif session == 290: 
        path = mainpath + 's210329_Saline_5.13Hz_512x512'
        raw_filepath = 's210329_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210329_behav_' + str(se) + '.avi.mat')
            
    elif session == 291: 
        path = mainpath + 's210330_Saline_5.13Hz_512x512'
        raw_filepath = 's210330_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210330_behav_' + str(se) + '.avi.mat')
            
    elif session == 292: 
        path = mainpath + 's210331_Saline_5.13Hz_512x512'
        raw_filepath = 's210331_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210331_behav_' + str(se) + '.avi.mat')
            
    elif session == 293: 
        path = mainpath + 's210401_Saline_5.13Hz_512x512'
        raw_filepath = 's210401_Saline_5.13Hz_512x512.xlsx'
        behav_data = []
        for se in range(10):        
            behav_data.append('[SHANA]s210401_behav_' + str(se) + '.avi.mat')
            
    elif session == 294: 
        path = mainpath + 's210518_2_H_PSL'
        raw_filepath = 's210518_2_H_PSL.xlsx'
        behav_data = []
        for se in range(6):        
            behav_data.append('[SHANA]s210518_2_H_PSL_' + str(se) + '.avi.mat')

    elif session == 295: 
        path = mainpath + 's210520_1_PSL'
        raw_filepath = 's210520_1_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210520_1_PSL_' + str(se) + '.avi.mat')
            
    elif session == 296: 
        path = mainpath + 's210524_1_H_PSL'
        raw_filepath = 's210524_1_H_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210524_1_H_PSL_' + str(se) + '.avi.mat')
            
    elif session == 297: 
        path = mainpath + 's210528_PSL'
        raw_filepath = 's210528_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210528_PSL_' + str(se) + '.avi.mat')
            
    elif session == 298: 
        path = mainpath + 's210602_1_PSL'
        raw_filepath = 's210602_1_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210602_1_PSL_' + str(se) + '.avi.mat')
            
    elif session == 299: 
        path = mainpath + 's210603_1_H_PSL'
        raw_filepath = 's210603_1_H_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210603_1_H_PSL_' + str(se) + '.avi.mat')
            
    elif session == 300: 
        path = mainpath + 's210603_PSL'
        raw_filepath = 's210603_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210603_PSL_' + str(se) + '.avi.mat')
            
    elif session == 301: 
        path = mainpath + 's210604_2_PSL'
        raw_filepath = 's210604_2_PSL.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('[SHANA]s210604_2_PSL_' + str(se) + '.avi.mat')
            
    elif session == 302: 
        path = mainpath + 's210614_2_Sham'
        raw_filepath = 's210614_2_Sham.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('empty')
            
    elif session == 303: 
        path = mainpath + 's210618_1_Sham_M10'
        raw_filepath = 's210618_1_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210618_1_Sham_M10_' + str(se) + '.avi.mat')
            
    elif session == 304: 
        path = mainpath + 's210614_1_Sham_M10'
        raw_filepath = 's210614_1_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210614_1_Sham_M10_' + str(se) + '.avi.mat')
            
    elif session == 305: 
        path = mainpath + 's210618_2_Sham_M10'
        raw_filepath = 's210618_2_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210618_2_Sham_M10_' + str(se) + '.avi.mat')

    elif session == 306: 
        path = mainpath + 's210625_1_Sham_M10'
        raw_filepath = 's210625_1_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210625_1_Sham_M10_' + str(se) + '.avi.mat')
            
    elif session == 307: 
        path = mainpath + 's210625_2_Sham_M10'
        raw_filepath = 's210625_2_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210625_2_Sham_M10_' + str(se) + '.avi.mat') 
##
    elif session == 308: 
        path = mainpath + 's210628_1_Sham_M10'
        raw_filepath = 's210628_1_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):
            if se in range(6):
                behav_data.append('s210628_1_Sham_M10_' + str(se) + '.avi.mat')
            if se in range(6, 12):
                behav_data.append('empty')

    elif session == 309: 
        path = mainpath + 's210628_2_Sham_M10'
        raw_filepath = 's210628_2_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):
            behav_data.append('s210628_2_Sham_M10_' + str(se) + '.avi.mat')
            
    elif session == 310: 
        path = mainpath + 's210629_1_Sham_M10'
        raw_filepath = 's210629_1_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):
            if se in [6,7]: behav_data.append('empty')
            else: behav_data.append('s210629_1_Sham_M10_' + str(se) + '.avi.mat')

    elif session == 311: 
        path = mainpath + 's210630_Sham_M10'
        raw_filepath = 's210630_Sham_M10.xlsx'
        behav_data = []
        for se in range(12):        
            behav_data.append('s210630_Sham_M10_' + str(se) + '.avi.mat')
           #
    elif session == 312: 
        path = mainpath + 's210712_1_CFA_K100'
        raw_filepath = 's210712_1_CFA_K100.xlsx'
        behav_data = []
        for se in range(11):
            behav_data.append('s210712_1_CFA_K100_' + str(se) + '.avi.mat')

    elif session == 313: 
        path = mainpath + 's210715_1_CFA_K100'
        raw_filepath = 's210715_1_CFA_K100.xlsx'
        behav_data = []
        for se in range(11):
            behav_data.append('s210715_1_CFA_K100_' + str(se) + '.avi.mat')
            
    elif session == 314: 
        filename = 's210716_CFA_K100'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(10):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')

    elif session == 315: 
        filename = 's210719_CFA_K100'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(10):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 316: 
        filename = 's210721_1_CFA_K100'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(10):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 317: 
        filename = 's210721_2_CFA_K100'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(10):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 318: 
        filename = 's210722_CFA_K100'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(10):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 319: 
        filename = 's210804_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 320: 
        filename = 's210805_1_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 321: 
        filename = 's210805_2_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 322: 
        filename = 's210810_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 323: 
        filename = 's210811_1_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 324: 
        filename = 's210811_2_CFA_K50'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 325: 
        filename = '211028_1_blue_MPTP_D3m_D7s'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(16):
            if se==4: behav_data.append('empty')
            else: behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 326: 
        filename = '211028_1_red_MPTP_D3m_D7s'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(16):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 327: 
        filename = '211028_2_blue_MPTP_D3s_D4m_D7m'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(22):
            behav_data.append(filename + '_' + str(se) + '.mat')
            ##
    elif session == 328: 
        filename = '211029_MPTP_D3s_D4m_D7m'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(22):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 329: 
        filename = '211102_MPTP_D3s_D7m_D8s'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(22):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 330: 
        filename = '211103_MPTP_D3s_D7m_D8s'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(22):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 331: 
        filename = '211105_MPTP_D3m_D7s_D8m'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(22):
            behav_data.append(filename + '_' + str(se) + '.mat')
            #
    elif session == 332: 
        filename = 's211027_1_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 333: 
        filename = 's211102_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 334: 
        filename = 's211103_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 335: 
        filename = 's211104_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')

    elif session == 336: 
        filename = 's211110_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 337: 
        filename = 's211111R_1_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 338: 
        filename = 's211116_2_PSL_Mag30'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(14):
            behav_data.append(filename + '_' + str(se) + '.mat')
            
    elif session == 339: 
        filename = '211028_2_red_D3m_D7s'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(16):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 340: 
        filename = '211110_2_D4s_D7m'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(16):
            if se==4: behav_data.append('empty')
            else: behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 341: 
        filename = '211112_D4s_D7m'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(16):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
    ##
    elif session == 342: 
        filename = 's220215_2_F'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 343: 
        filename = 's220217_1_F'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(9):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 344: 
        filename = 's220217_2_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
    
    elif session == 345: 
        filename = 's220217_3_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 346: 
        filename = 's220217_4_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 347: 
        filename = 's220221_2_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            if se == 4: continue
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 348: 
        filename = 's220221_4_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 349: 
        filename = 's220224_1_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
    elif session == 350: 
        filename = 's220224_2_F5'
        path = mainpath + filename
        raw_filepath = filename + '.xlsx'
        behav_data = []
        for se in range(8):
            behav_data.append(filename + '_' + str(se) + '.avi.mat')
            
     
    else:
        path = None; behav_data = None; raw_filepath = None
        endsw = True

    return path, behav_data, raw_filepath, endsw


































































