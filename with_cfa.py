import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
import pickle
import random; seed = 0
from datetime import date

from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)
# In
code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13', header=0)[0] 

# 종목코드가 6자리이기 때문에 6자리를 맞춰주기 위해 설정해줌 
code_df.종목코드 = code_df.종목코드.map('{:06d}'.format) 
# 우리가 필요한 것은 회사명과 종목코드이기 때문에 필요없는 column들은 제외해준다. 
code_df = code_df[['회사명', '종목코드']] 
# 한글로된 컬럼명을 영어로 바꿔준다. 
code_df = code_df.rename(columns={'회사명': 'name', '종목코드': 'code'}) 

# In
def code_to_name(mscode):
    return code_df.iloc[:,0][np.where(code_df.iloc[:,1] == mscode)[0][0]]

def get_url(item_name, code_df): 
    code = code_df.query("name=='{}'".format(item_name))['code'].iloc[0]
    url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
#    print("요청 URL = {}".format(url)) 
    return url, code

def to_integer(dt_time):
    return int(10000*dt_time.year + 100*dt_time.month + dt_time.day)

def date_conv(msdata):
    return to_integer(datetime.datetime.strptime(msdata, "%Y.%m.%d").date())

def date_conv2(msdata):
    return to_integer(datetime.datetime.strptime(msdata, "%Y-%m-%d").date())

def date_conv3(msdata): # 통합버전
    if type(msdata) != str and type(msdata) != 'datetime.date':
        return np.nan
    
    try:
        msout = to_integer(datetime.datetime.strptime(msdata, "%Y-%m-%d").date())
    except:
        try:
            msout = to_integer(datetime.datetime.strptime(msdata, "%Y.%m.%d").date())
        except:
            try:
                msout = to_integer(datetime.datetime.strptime(msdata, "%Y,%m,%d").date())
            except:
                msout = 10000*msdata.year + 100*msdata.month + msdata.day
            
    return msout

def pratio_calc(ms1, ms2): # original, moving avg
    cnt=[]
    for i in range(ms1.shape[0]):
        cnt.append(ms1[i]/ms2[i])

    pratio = np.mean(cnt)
    return pratio

def them_calc(date, value, s, e, them_name, figuresw=False):
    if not(np.max(date) > e and np.min(date) < s):
#        print(item_name, them_name, 'data missing')
        return np.nan
    
    ms_graph_value = value
    ms_graph_date = date
    
    ix = ((ms_graph_date > s) * (ms_graph_date < e)) 
    ixx = np.where(ix==1)[0]
    
    if figuresw:
        plt.figure()
        plt.plot(ms_graph_value[ix], label='beta = 1')
        plt.legend()
        
    for beta in [0.8]:
        moving_avg = np.zeros(ms_graph_value.shape[0]+1); moving_avg[1:] = np.nan
        for i in range(1, ms_graph_value.shape[0]):
            moving_avg[i] = ((beta*moving_avg[i-1]) + ((1-beta)*ms_graph_value[i]))
    
        if figuresw:
            plt.plot(moving_avg[ixx], label='beta = ' + str(beta))
    
    
    if figuresw:
        msdate = ms_graph_date[ix] % 10000
        msyear = ms_graph_date[ix] // 10000
        miny = int(np.min(msyear)); maxy = int(np.max(msyear))
        msmax = np.max(ms_graph_value)
        for y in range(miny, maxy+1):
            ix2 = np.where(((msyear == y) * (msdate < 200)) == True)[0]
            if not(ix2.shape[0] == 0):
                plt.fill_between([int(np.min(ix2)),int(np.max(ix2))], 0, msmax, color='moccasin', alpha=0.5)
        
        ms_xticks = np.array(range(0, ms_graph_date[ix].shape[0], 10))
        plt.xticks(ms_xticks, np.array(ms_graph_date[ix][ms_xticks], dtype=int), rotation='vertical')
    
    ms1 = ms_graph_value[ix]
    ms2 = moving_avg[ixx]
    
    pratio = pratio_calc(ms1, ms2)

    if figuresw:
        plt.title(them_name + '_' + str(pratio))
    
    return pratio

# In
import_method = 3
# 1 for updae
# 2 for re-start
# 3 for just load

path1 = 'E:\\mscore\\code_lab\\stockmarket\\'
path2 = 'D:\\mscode\\stockmarket\\'
path3 = 'C:\\test\\stockmarket\\'

if os.path.isdir(path1):
    mainpath = path1 
elif os.path.isdir(path2):
    mainpath = path2
elif os.path.isdir(path3):
    mainpath = path3
    
print('mainpath', mainpath)

loadpath = mainpath + 'stock_law_pickle'
isfile1 = os.path.isfile(loadpath)
if import_method == 1: # 기존것 load후 update
    with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
        msdata_raw = pickle.load(f)
        
    mssignalss = msdata_raw['mssignalss']
    msindex = msdata_raw['msindex']
    
    # update
    removed = []
    for ms_event in range(0, code_df.shape[0]):
        exist_data = np.array(mssignalss[ms_event])
        recent_date = np.max(exist_data[:,0])
        #
        try:
            item_name = code_to_name(msindex[ms_event])
        except:
            print(msindex[ms_event], '존재하지 않습니다. 상폐?')
            removed.append(msindex[ms_event])
            continue
        
        url, code = get_url(item_name, code_df)
        
        df = pd.DataFrame(); msid1 = None
        for page in range(1, 1000): 
            if page % 10 == 0:
                print(item_name, page)
            
            pg_url = '{url}&page={page}'.format(url=url, page=page)
            
            msdata = pd.read_html(pg_url, header=0)[0]
            current_date = list(msdata.iloc[:,0])
            for h in range(len(current_date)):
                current_date[h] = date_conv3(current_date[h])
            
            if recent_date > np.nanmax(current_date):
                print(item_name, page, 'stop')
                break
            msdata.iloc[:,0] = current_date
            df = df.append(msdata, ignore_index=True) 
            
            msid = np.nanmean(msdata.iloc[:,1:5])
            if msid1 == msid:
                print(item_name, page, 'stop')
                break
            msid1 = msid
 
        df2_tmp = df.dropna()
        df2_tmp = pd.concat([df2_tmp.iloc[:,0], df2_tmp.iloc[:,1], df2_tmp.iloc[:,6]], axis=1) 
        # 날짜, 종가(close), 거래량만 사용 
        df2_tmp = np.array(df2_tmp)
        ix = np.argsort(df2_tmp[:,0])
        df2_sorted = df2_tmp[ix]
        
        # 크롤링한 data df2_sorted를 exist_data에 추가 후 mssignalss에 다시 저장
        for row in range(df2_sorted.shape[0]):
            crow = df2_sorted[row,:]    
            if crow[0] > exist_data[-1,0] and crow[0] != date_conv3(date.today()):
                exist_data = np.concatenate((exist_data, np.reshape(crow, (1,3))), axis=0)
                # 당일 데이터는 왜 있는겨? 아직 장중인데 
                
        mssignalss[ms_event] = np.array(exist_data)
    
    mssignalss2 = []; msindex2 =[]
    for i in range(len(msindex)):
        if msindex[i] in removed:
            print(msindex[i], '상장폐지? 제외합니다')
        else:
            mssignalss2.append(mssignalss[i])
            msindex2.append(msindex[i])
    
    msdict = {'mssignalss' : mssignalss2, 'msindex' : msindex2}
    picklesavename = mainpath + 'stock_law_pickle'
    with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(picklesavename, '저장되었습니다.')
                
elif import_method == 2: # 새롭게 만듦
    # In data from csv file
    mssignalss = []; [mssignalss.append([]) for u in range(code_df.shape[0])]
    mssave_them_score = []
    msindex = []; [msindex.append([]) for u in range(code_df.shape[0])]
    mspath = mainpath + 'stock_CSV\\'
    for u in range(0, code_df.shape[0]):
        # In
        
        item_name = code_df.iloc[u,0]
        item_name = '브이티지엠피'
        
        print(item_name, 'started')
        url, code = get_url(item_name, code_df)
        # df1
        loadpath = mspath + code + '.KS.csv'
        
        if os.path.isfile(loadpath):
            df1 = pd.read_csv(loadpath)
            for i in range(df1.shape[0]):
                try:
                    df1.iloc[i,0] = date_conv2(df1.iloc[i,0])
                except:
                    df1.iloc[i,0] = date_conv(df1.iloc[i,0])
                
            df1 = pd.concat([df1.iloc[:,0], df1.iloc[:,4], df1.iloc[:,6]], axis=1) # 날짜, 종가(close), 거래량만 사용 
            df1 = np.array(df1)
            # df2
            lastdate = np.max(df1[:,0])
            print(lastdate)
            
            # 날짜, 종가(close), 거래량만 사용 
            
            page = 0; whilesw = True
            while whilesw:
                page += 1
                pg_url = '{url}&page={page}'.format(url=url, page=page)
                df2_tmp = pd.read_html(pg_url, header=0)[0]
                df2_tmp = df2_tmp.dropna()
                
                for i in range(df2_tmp.shape[0]-1, -1, -1):
                    df2_tmp.iloc[i,0] = date_conv(df2_tmp.iloc[i,0])
                    if df2_tmp.iloc[i,0] > lastdate:
                        msadd = np.reshape(np.array([df2_tmp.iloc[i,0], df2_tmp.iloc[i,1], df2_tmp.iloc[i,6]]), (1,3))
                        df1 = np.append(df1,msadd,axis=0)
                    else:
                        print(item_name, lastdate, 'integrated...')
                        whilesw = False
                    
            ix = np.argsort(df1[:,0])
            df1_sorted = df1[ix]
            print(item_name, 'missing datas...', np.sum(np.isnan(df1_sorted[:,1])))
            
            df1_sorted = df1_sorted[np.isnan(df1_sorted[:,1])==0]
            
        else:
            print(item_name, 'old data 없음')
        
            df = pd.DataFrame() 
            # 1페이지에서 20페이지의 데이터만 가져오기 
            msid1 = None
            for page in range(1, 1000): 
                if page % 10 == 0:
                    print(item_name, page)
                
                pg_url = '{url}&page={page}'.format(url=url, page=page)
                
                msdata = pd.read_html(pg_url, header=0)[0]
                df = df.append(msdata, ignore_index=True) 
                msid = np.nanmean(msdata.iloc[:,1:5])
                if msid1 == msid:
                    print(item_name, page, 'stop')
                    break
                msid1 = msid
                
                # df.dropna()를 이용해 결측값 있는 행 제거 
            df2_tmp = df.dropna()
            df2_tmp = pd.concat([df2_tmp.iloc[:,0], df2_tmp.iloc[:,1], df2_tmp.iloc[:,6]], axis=1) 
            # 날짜, 종가(close), 거래량만 사용 
            for i in range(df2_tmp.shape[0]):
                df2_tmp.iloc[i,0] = date_conv(df2_tmp.iloc[i,0])
            
            df2_tmp = np.array(df2_tmp)
            ix = np.argsort(df2_tmp[:,0])
            df2_sorted = df2_tmp[ix]
            print(item_name, 'missing datas...', np.sum(np.isnan(df2_sorted[:,1])))
            
            df2_sorted = df2_sorted[np.isnan(df2_sorted[:,1])==0]
            df1_sorted = df2_sorted
            
            mssignalss[u] = df1_sorted # save
            msindex[u] = code # code가 변하는듯 하니, 여기 따로 저장해준다.
        
        if False: # 시각화
            plt.figure()
            plt.plot(df1_sorted[:,1])
            mstitle = item_name + ' missing datas ' + str(np.sum(np.isnan(df1_sorted[:,1]))) + '__ date '\
             + str(np.min(df1_sorted[:,0])) + ' to ' + str(np.max(df1_sorted[:,0]))
            
            plt.title(mstitle)
            print(np.min(df1_sorted[:,0]), 'to', np.max(df1_sorted[:,0]))
            
            msdate = df1_sorted[:,0] % 10000
            msyear = df1_sorted[:,0] // 10000
            miny = int(np.min(msyear)); maxy = int(np.max(msyear))
            msmax = np.max(df1_sorted[:,1])
            for y in range(miny, maxy+1):
                ix2 = np.where(((msyear == y) * (msdate < 200)) == True)[0]
                if not(ix2.shape[0] == 0):
                    plt.fill_between([int(np.min(ix2)),int(np.max(ix2))], 0, msmax, color='moccasin', alpha=0.5)
            
            ms_xticks = np.array(range(0, df1_sorted[:,0].shape[0], 200))
            plt.xticks(ms_xticks, np.array(df1_sorted[:,0][ms_xticks], dtype=int), rotation='vertical')
            
            savepath = mainpath + 'msplot\\'
            
            plt.savefig(savepath + mstitle + '.png', dpi=1000)
            plt.close()
               
    msdict = {'mssignalss' : mssignalss, 'msindex' : msindex}

    picklesavename = 'E:\\mscore\\code_lab\\stockmarket\\' + 'stock_law_pickle'
    with open(picklesavename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(msdict, f, pickle.HIGHEST_PROTOCOL)
        print(picklesavename, '저장되었습니다.')
        
elif import_method == 3: # 기존것 load후 update
    with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
        msdata_raw = pickle.load(f)
        
    mssignalss = msdata_raw['mssignalss']
    msindex = msdata_raw['msindex']

# In[]
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle    
    
overwrite = False
path1 = 'E:\\mscore\\code_lab\\stockmarket\\'
path2 = 'D:\\mscode\\stockmarket\\'
path3 = 'C:\\test\\stockmarket\\'

if os.path.isdir(path1):
    mainpath = path1 
elif os.path.isdir(path2):
    mainpath = path2
elif os.path.isdir(path3):
    mainpath = path3
    
print('mainpath', mainpath)

loadpath = mainpath + 'stock_law_pickle'
isfile1 = os.path.isfile(loadpath)
if isfile1 and not(overwrite): 
    with open(loadpath, 'rb') as f:  # Python 3: open(..., 'rb')
        msdata_raw = pickle.load(f)
        
    mssignalss = msdata_raw['mssignalss']
    msindex = msdata_raw['msindex']
    
startday = 20121201 # moving avg 구해야되니깐 한달전 부터 가져오자

X=[]; Z=[]
for i in range(len(mssignalss)):
    if np.min(mssignalss[i][:,0]) < startday:
        sat = np.min(np.where(mssignalss[i][:,0] > 20100101)[0]) # 2010년부터 시작   
        X.append(mssignalss[i][sat:,:])
        Z.append(msindex[i])

print('조건에 맞는 종목', len(X))

# 특정종목 시각화
if False:
    plt.figure()
    print(code_to_name('005880'))
    plt.plot(np.array(X[np.where(np.array(Z) == '005880')[0][0]])[:,1])
    
    
    plt.figure()
    print(code_to_name('005930'))
    plt.plot(np.array(X[np.where(np.array(Z) == '005930')[0][0]])[:,1])


# missing data, 양쪽 값의 평균으로 채움 # 날짜 for 할수있어야함
# missing이 연속으로 일어날경우, 일단 확인후, 채우거나 제외

lastday = str(int(np.max(mssignalss[0][:,0]))) 
dt_index = pd.date_range(start=str(startday), end=lastday)
dt_list = np.array(dt_index.strftime("%Y%m%d").tolist(), dtype=int)

# 휴일 체크.. holidays는 확인용 변수임
exceptlist = [20150814, 20171002, 20170922] # 뭔날인지는 모르는데,, 단체로 없음
holidays = []; dt_list2 = []
for ms_date in dt_list: # 지정한 모든 날짜
    sw = False
    for ms_event in range(len(X)): # 종목  
        if ms_date in X[ms_event][:,0] and not(ms_date in exceptlist):
            sw = True
        if sw:
            break
    if not(sw):
        holidays.append(ms_date)
    elif sw:
        dt_list2.append(ms_date)
print('holidays', len(holidays))

# In[]
# 개별 종목에서 비어있는 칸 채우기
# 얼마나 비어있는지 저장해서 추후 처리함
cnt = 0; 
X_padding_save = []; [X_padding_save.append([]) for u in range(len(X))]
padding_ix_save = []; [padding_ix_save.append([]) for u in range(len(X))]
for ms_event in range(len(X)): # 종목
    save_sw = True
    xdate = X[ms_event][:,0]
    xsignal = X[ms_event][:,1]
    X_padding = []; padding_ix = []
    for i in range(len(dt_list2)): # 지정한 모든 날짜 (휴일제외)
        ms_date = dt_list2[i]
        if ms_date in xdate:
            mssignal = xsignal[np.where(xdate==ms_date)[0][0]]
        if not (ms_date in xdate):
#            print(ms_event, ms_date, mssignal)
            
            try:
                before_day = xsignal[np.where(xdate==dt_list2[i-1])[0][0]]
                after_day  = xsignal[np.where(xdate==dt_list2[i+1])[0][0]]
                mssignal = np.mean([before_day, after_day])
            except:
                try:
                    before_day = xsignal[np.where(xdate==dt_list2[i-1])[0][0]]
                    mssignal = np.mean([before_day, after_day])
                except:
                    cnt += 1
                    print('cnt', cnt)
                    save_sw = False
                    break

            padding_ix.append(i)
        X_padding.append([mssignal, ms_date])
        
    if save_sw:
        X_padding_save[ms_event] = np.array(X_padding) 
        # skip이 있는 구문이기 때문에 preallocation을 해야만 index가 유지됨.
        # 해당 code index는 위 session에 Z에 유지되고, 아래에서 동일처리됨
        padding_ix_save[ms_event] = padding_ix

print('동일체크', len(X_padding_save))
 
pd_save = []; pd_save = np.zeros(len(padding_ix_save))
for i in range(len(padding_ix_save)):
    if len(padding_ix_save[i]) > 0:
        pd_save[i] = np.sum(np.array(padding_ix_save[i]) > -1)

# missing이 5days 이상인거 짤라버림.
eix = np.where(pd_save>5)[0]
print('mssing > 5d 는 제외합니다. 제외되는 종목수', eix.shape[0])
#print(np.where(np.array(Z) == '005930')[0][0])
# In[] 너무 많이 오른 (상한가초과) 주식 제외, 액면분할 주식 스케일링
split_stock_list = []
except_datas = [];
for i in range(len(X_padding_save)):
    if len(X_padding_save[i]) > 0:
        mssignal = np.array(X_padding_save[i][:,0]) # signal for 0
        
        diff = np.zeros(mssignal.shape[0])
        diff[:-1] = mssignal[1:]
                
        diff_ratio = (diff/mssignal)
        dmax = np.max(diff_ratio[:-1])
        dmin = np.min(diff_ratio[:-1])

        if dmax > 1.31: # 상한가가 1.3임
            except_datas.append(Z[i])
        elif dmin < 1/2:
            split_stock_list.append(Z[i])         
print(len(except_datas))
            
# 눈으로보고... 걸르자........
#i = -1
## In[]
#i += 1
#code = split_stock_list[i]          
#plt.figure()
#print(split_stock_list[i], code_to_name(code))
#plt.plot(np.array(X[np.where(np.array(Z) == code)[0][0]])[:,1])
#A = np.array(X[np.where(np.array(Z) == code)[0][0]])
        
except_datas2 = ['074600', '047560', '126870', '050120', '099340', '014470', '099320', \
                '056190', '139670', '002070', '073070', '114570', '007540', '005740', \
                '122350', '009620', '097780', '131390', '064800', '090710', '010240', \
                '016880', '001000', '000700', '137940', '053060', '060560', '005880']
cnt = 0
for i in range(len(split_stock_list)): 
    if not split_stock_list[i] in except_datas2:
        code = split_stock_list[i]          
#        plt.figure()
#        print(split_stock_list[i], code_to_name(code))
#        plt.plot(np.array(X_padding_save[np.where(np.array(Z) == code)[0][0]])[:,0])
        
        zix = np.where(np.array(Z) == code)[0][0]
        mssignal = np.array(X_padding_save[zix][:,0]) # signal for 0        
        diff = np.zeros(mssignal.shape[0])
        diff[:-1] = mssignal[1:]        
        diff_ratio = (diff[:-1]/mssignal[:-1])
        dmax = np.max(diff_ratio[:-1])
        dmin = np.min(diff_ratio[:-1])
        
#        plt.figure()
#        plt.plot(diff_ratio)
        
        splitix = list(np.where(diff_ratio<0.5)[0])
        splitix.append(mssignal.shape[0]-1)
        splitix2 = [-1]; splitix2 = splitix2 + splitix
        
        ms_merge = np.zeros(mssignal.shape[0])
        for j in range(len(splitix2)-1):
#            plt.figure()
#            plt.plot(mssignal[splitix2[j]+1:splitix2[j+1]+1])
            
            tmp_signal = np.array(mssignal[splitix2[j]+1:splitix2[j+1]+1])
            if j > 0:
                tmp_signal = tmp_signal * (mssignal[splitix2[j]] / mssignal[splitix2[j]+1])
            ms_merge[splitix2[j]+1:splitix2[j+1]+1] = tmp_signal
#            if j
        cnt += 1
        if cnt < 5 and False:
            plt.figure()
            plt.plot(X_padding_save[zix][:,0])
            plt.figure()
            plt.plot(ms_merge)
        
        X_padding_save[zix][:,0] = ms_merge

# In[]     
X2=[]; Z2=[]
for i in range(len(X_padding_save)):
#    print(len(X_padding_save[i]))
    if not i in list(eix) and len(X_padding_save[i]) > 0:
        if not Z[i] in except_datas + except_datas2:
            # 제외 list에 있거나 data가 없는 경우 빼 버림.
            # index 변경됨 
            X2.append(X_padding_save[i]) # data 2
            Z2.append(Z[i]) # index 2
X2 = np.array(X2)
print('X2', X2.shape, '(종목수, 길이)')
print('동일체크', X2.shape[0], len(Z2))

# In X2에서 moving average ratio 구하고 가자..

#moving_avg_ratio = []
#for ms_event in range(len(X2)):
#    ms_graph_value = np.array(X2[ms_event][:,0])
#    
#    for beta in [0.8]:
#        moving_avg = np.zeros(ms_graph_value.shape[0]); moving_avg[1:] = np.nan
#        for i in range(1, ms_graph_value.shape[0]):
#            moving_avg[i] = ((beta*moving_avg[i-1]) + ((1-beta)*ms_graph_value[i]))
#            
#    moving_avg_ratio.append(moving_avg)
#moving_avg_ratio = np.array(moving_avg_ratio)   

# In[] X2 nmr
plt.figure()
X2_nmr = list(np.array(X2))
cnt=0
for ms_event in range(len(X2)):
    X2_signal = np.array(X2[ms_event][:,0])
    X2_signal = X2_signal/np.mean(X2_signal)
    cnt += 1
    if cnt < 50 and False:
        print(ms_event, Z[ms_event])
        plt.plot(X2_signal)
        
    X2_nmr[ms_event][:,0] = X2_signal
        
# In[]    

def genarator_by_year(test_year, month_list2, test_only):
    buy = np.nan
    duration_days = 30
    yearlist = []
    for u in range(5, -1, -1):
        yearlist.append(test_year-u)
        
    X2_nmr_forY = list(np.array(X2_nmr))
    X2_nmr_forX = list(np.array(X2_nmr))
    
    ml_dataset=[]; [ml_dataset.append([]) for u in range(3)] # data, label, index
    [ml_dataset[0].append([]) for u in range(len(yearlist))]
    for msmonth in month_list2:
        for i in range(len(X2_nmr_forY)): # 종목     
            # y
            ydate = test_year*10000 + msmonth
            y_at = np.min(np.where(X2_nmr_forY[i][:,1] > ydate)) # col 1 for date
            
            if not(test_only):
                buy = X2_nmr_forY[i][y_at,0]
                if np.isnan(buy):
                    break
                
                sell = np.mean(X2_nmr_forY[i][y_at+1:y_at+duration_days+1,0])                
                yvalue = (sell/buy)-1
                
                X2_nmr_forY[i][y_at,0] = np.nan
                X2_nmr_forY[i][y_at+1:y_at+duration_days+1,0] = np.nan
                
                if yvalue > 0.08: # good
                    ytmp = [0,1]
                elif yvalue < 0.01: 
                    ytmp = [1,0] # bad
                else:
                    ytmp = [0,0] # neutral
            elif test_only:
                ytmp = [1,0]
                
            ml_dataset[1].append(ytmp)            
            
            # x
            for yix, msyears in enumerate(yearlist):
                sdate = msyears*10000 + msmonth
                x_at = np.min(np.where(X2_nmr_forX[i][:,1] > sdate)) # col 1 for date
            
                if msyears != test_year:
                    xtmp = X2_nmr_forX[i][:,0][x_at: x_at+duration_days]
                elif msyears == test_year: 
                    xtmp = X2_nmr_forX[i][:,0][x_at-duration_days: x_at]
                 
                ml_dataset[0][yix].append(np.reshape(xtmp, (xtmp.shape[0], 1))) # [data][years]
            
            # z
            ml_dataset[2].append(Z2[i] + str(ydate))
        if not(np.isnan(buy)):
            print(msmonth, 'for', test_year)
    return ml_dataset
    
# In[] sampling
test_only = False

month_list = []; start_at = 101
for m in range(1,13):
    for d in range(1,32):
        month_list.append(m*100+d)
month_list = np.array(month_list)
six = int(np.where(month_list==start_at)[0])
month_list2 = np.concatenate((month_list[six:], month_list[:six]), axis=0)

test_year = 2019
ml_dataset = genarator_by_year(test_year, month_list, test_only)

test_year = 2018
ml_dataset2018 = genarator_by_year(test_year, month_list, test_only)

for u in range(len(ml_dataset[0])):
    ml_dataset[0][u] = np.concatenate((ml_dataset[0][u], ml_dataset2018[0][u]), axis=0)
ml_dataset[1] = np.concatenate((ml_dataset[1], ml_dataset2018[1]), axis=0)
ml_dataset[2] = np.concatenate((ml_dataset[2], ml_dataset2018[2]), axis=0)

print('total sample #', len(ml_dataset[1]))
print('sample distributions', np.mean(ml_dataset[1], axis=0), 'cover', np.mean(ml_dataset[1])*2)

 # upsampling, neutral 제거
ml_dataset[1] = np.array(ml_dataset[1])
badix = np.where(ml_dataset[1][:,0]==1)[0]
goodix = np.where(ml_dataset[1][:,1]==1)[0]
print('sample distributions', 'ix 재확인', badix.shape[0], goodix.shape[0])

data0 = np.array(ml_dataset[1])[badix]
data1 = np.array(ml_dataset[1])[goodix]
mul = data0.shape[0] // data1.shape[0]
mod = data0.shape[0] % data1.shape[0]
rix = list(range(data1.shape[0]))
random.seed(seed); rix2 = random.sample(rix, mod)

# X먼저, bad 넣고, good 반복해서 append
trX=[]; [trX.append([]) for u in range(len(ml_dataset[0]))]
for u in range(len(ml_dataset[0])):
    trX[u] = np.array(ml_dataset[0][u][badix])
trY = np.array(ml_dataset[1])[badix]
trZ = np.array(ml_dataset[2])[badix]
print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])

for m in range(mul): 
    for u in range(len(ml_dataset[0])):
        trX[u] = np.append(trX[u], ml_dataset[0][u][goodix], axis=0)
    trY = np.append(trY, np.array(ml_dataset[1])[goodix], axis=0)
    trZ = np.append(trZ, np.array(ml_dataset[2])[goodix], axis=0)
    print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])
    
for u in range(len(ml_dataset[0])):
    trX[u] = np.append(trX[u], ml_dataset[0][u][goodix][rix2], axis=0)
trY = np.append(trY, np.array(ml_dataset[1])[goodix][rix2], axis=0)
trZ = np.append(trZ, np.array(ml_dataset[2])[goodix][rix2], axis=0)

print('sample distributions', np.mean(trY, axis=0), 'total #', trY.shape[0])


# In[]
from keras import regularizers
from keras.layers.core import Dropout
from keras import initializers
import keras
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam

inputsize = np.zeros(len(ml_dataset[0]), dtype=int) 
for unit in range(len(ml_dataset[0])):
    inputsize[unit] = ml_dataset[0][unit].shape[1] # size 정보는 계속사용하므로, 따로 남겨놓는다.

# learning intensity
epochs = 10 # epoch 종료를 결정할 최소 단위.
lr = 1e-3 # learning rate

n_out = 2
acc_thr = 0.90
maxepoch = 600

n_hidden = int(8 * 8) # LSTM node 갯수, bidirection 이기 때문에 2배수로 들어감.
layer_1 = int(8 * 8) # fully conneted laye node 갯수 # 8

# regularization
l2_rate = 0.4 # regularization 상수
dropout_rate1 = 0.20 # dropout late
dropout_rate2 = 0.10 # 
batch_size = 2**11

# In[]
def keras_setup():
    #### keras #### keras  #### keras #### keras  ####keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    init = initializers.he_uniform(seed=seed) # he initializer를 seed 없이 매번 random하게 사용 -> seed 줌
    
    input1 = []; [input1.append([]) for i in range(len(ml_dataset[0]))] # 최초 input layer
    input2 = []; [input2.append([]) for i in range(len(ml_dataset[0]))] # input1을 받아서 끝까지 이어지는 변수
    
    for unit in range(len(ml_dataset[0])):
        input1[unit] = keras.layers.Input(shape=(inputsize[unit], 1)) # 각 병렬 layer shape에 따라 input 받음
        input2[unit] = Bidirectional(LSTM(n_hidden))(input1[unit]) # biRNN -> 시계열에서 단일 value로 나감
        input2[unit] = Dense(n_hidden , kernel_initializer = init, \
              activation='relu')(input2[unit]) # fully conneted layers, relu
        input2[unit] = Dropout(dropout_rate1)(input2[unit]) # dropout
        input2[unit] = keras.layers.Reshape((1,input2[unit].shape[1]))(input2[unit])
    
    added = keras.layers.concatenate(input2, axis=1) 
    added = Bidirectional(LSTM(n_hidden))(added)
    
    merge_1 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate),\
                    activation='relu')(added) # fully conneted layers, relu
    merge_2 = Dropout(dropout_rate2)(merge_1) # dropout
    
    merge_2 = Dense(layer_1, kernel_initializer = init, kernel_regularizer=regularizers.l2(l2_rate), \
                    activation='sigmoid')(merge_2) # fully conneted layers, sigmoid
    merge_3 = Dense(n_out, input_dim=n_out)(merge_2) # regularization 삭제
    merge_4 = Activation('softmax')(merge_3) # activation as softmax function
    
    model = keras.models.Model(inputs=input1, outputs=merge_4) # input output 선언
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), metrics=['accuracy']) # optimizer
    
    #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras  #### keras #### keras
    return model

RESULT_SAVE_PATH = mainpath + 'tf_result_20200217_\\'  # project
if not os.path.exists(RESULT_SAVE_PATH):
    os.mkdir(RESULT_SAVE_PATH)

initial_weightsave = RESULT_SAVE_PATH + 'initial_weight.h5'

#model = keras_setup()
#model.save_weights(initial_weightsave)
#print(model.summary())
# In[]
# In
tfoverwrite = False
#import random
import time
import csv

cvnum = 5; allsetsw = False
tn = trY.shape[0]/cvnum

# cv

badcode = trZ[np.where(trY[:,0]==1)[0]]
goodcode = trZ[np.where(trY[:,1]==1)[0]]
print(badcode.shape[0], goodcode.shape[0])
#
badcode = np.array(list(set(badcode)))
goodcode = np.array(list(set(goodcode)))
print(badcode.shape[0], goodcode.shape[0])

id_list_bad = list(range(badcode.shape[0]))
random.seed(seed); random.shuffle(id_list_bad) 
id_list_good = list(range(goodcode.shape[0]))
random.seed(seed); random.shuffle(id_list_good) 

tn1 = badcode.shape[0]/5
tn2 = goodcode.shape[0]/5

l2_rate = 0.2

for cv in range(0, cvnum+1):
    l2_rate += l2_rate/10
    print('l2_rate', l2_rate)
    
    testset1 = id_list_bad[int(round(tn1*cv)):int(round(tn1*(cv+1)))]
    testset2 = id_list_good[int(round(tn2*cv)):int(round(tn2*(cv+1)))]

    if cv == cvnum-1:
        testset1 = id_list_bad[int(round(tn1*cv)):]
        testset2 = id_list_good[int(round(tn2*cv)):]
    elif cv == cvnum:
        allsetsw = True
    
    vix = []
    for i in range(len(testset1)):
        vix += list(np.where(trZ == badcode[testset1][i])[0])
    print('vix', len(vix))
    for i in range(len(testset2)):
        vix += list(np.where(trZ == goodcode[testset2][i])[0])
    print('vix', len(vix))

    final_weightsave = RESULT_SAVE_PATH + str(cv) + '_my_model_weights_final.h5'
    exist_model = os.path.isfile(final_weightsave)
    print('training을 위한 model 존재 유무', exist_model)
    
    if tfoverwrite or not(exist_model):
        print('cv #', cv, '학습된 model 없음. 새로시작합니다.')
        model = keras_setup()
#        model.load_weights(initial_weightsave) # model reset for start 
        before_loss = -np.inf; stop_cnt = 0; stop_sw = False
        
        if not(allsetsw):
            tr_x = []; x_valid = []; tr_y = []
            for ms_years in range(len(ml_dataset[0])):   
                tr_x.append(np.delete(np.array(trX[ms_years]), vix, axis=0))
                x_valid.append(np.array(trX[ms_years])[vix])
                
            tr_y = np.delete(np.array(trY), vix, axis=0)
            y_valid = np.array(trY)[vix]
            print('cv #', cv, 'distribution', np.mean(tr_y, axis=0))
        elif allsetsw:
            tr_x = []; x_valid = []; tr_y = []
            tr_x = list(trX); tr_y = np.array(trY)
            
        shuffle_ix = list(range(len(tr_y)))
        random.seed(seed); random.shuffle(shuffle_ix)
        for ms_years in range(len(ml_dataset[0])):   
            tr_x[ms_years] = tr_x[ms_years][shuffle_ix]
        tr_y = tr_y[shuffle_ix]
         
        current_acc = -np.inf; cnt = -1
        hist_save_loss = []
        hist_save_acc = []
        hist_save_val_loss = []
        hist_save_val_acc = []
        starttime = time.time()
        current_weightsave = RESULT_SAVE_PATH + str(cv) + '_current_weight.h5'
#        model.save_weights(current_weightsave) 
        while current_acc < acc_thr: # 0.93: # 목표 최대 정확도, epoch limit
            if cnt > maxepoch/epochs or stop_sw:
                seed += 1
                model = keras_setup() # for test
#                model.save_weights(initial_weightsave)
                before_loss = -np.inf; stop_cnt = 0; stop_sw = False
                current_acc = -np.inf; cnt = -1
                print('seed 변경, model reset 후 처음부터 다시 학습합니다.')
                
            cnt += 1;
            isfile1 = os.path.isfile(current_weightsave)
            if isfile1 and cnt > 0:
                model.load_weights(current_weightsave)
#                model.load_weights(initial_weightsave) # model reset for start 

                print('cv #', cv, cnt, '번째 이어서 학습합니다.')
            else:
#                model.load_weights(initial_weightsave) # model reset for start 
                print('학습 진행중인 model 없음. 새로 시작합니다')

            if not(allsetsw):
                hist = model.fit(tr_x, tr_y, batch_size=batch_size, \
                                 epochs=epochs, validation_data = (x_valid, y_valid))
                hist_save_val_loss += list(np.array(hist.history['val_loss']))
                hist_save_val_acc += list(np.array(hist.history['val_accuracy']))
            elif allsetsw:
                hist = model.fit(tr_x, tr_y, batch_size=batch_size, \
                                 epochs=epochs)
                
            
            hist_save_loss += list(np.array(hist.history['loss'])); 
            hist_save_acc += list(np.array(hist.history['accuracy']))
            
            
            model.save_weights(current_weightsave) 
            current_acc = hist_save_acc[-1]
            current_loss = round(hist_save_loss[-1], 3)
            
            if current_acc < 0.51:
                stop_sw = True
                       
        model.save_weights(final_weightsave)   
        print('cv #', cv, 'traning 종료, final model을 저장합니다.')
              
        # hist figure save
        plt.figure();
        mouseNum = cv
        hist_save_loss_plot = np.array(hist_save_loss)/np.max(hist_save_loss)
        
        plt.plot(hist_save_loss_plot, label= '# ' + str(mouseNum) + ' loss')
        plt.plot(hist_save_acc, label= '# ' + str(mouseNum) + ' acc')
        plt.legend()
        plt.savefig(RESULT_SAVE_PATH + str(mouseNum) + '_trainingSet_result.png')
        plt.close()

        savename = RESULT_SAVE_PATH + str(mouseNum) + '_trainingSet_result.csv'
        csvfile = open(savename, 'w', newline='')
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(hist_save_acc)
        csvwriter.writerow(hist_save_loss)
        spendingtime = time.time() - starttime
        csvwriter.writerow([spendingtime, spendingtime/60, spendingtime/60**2])
        csvwriter.writerow([l2_rate])
        csvfile.close()

        if not(allsetsw):
            plt.figure();
            mouseNum = cv
            hist_save_val_loss_plot = np.array(hist_save_val_loss)/np.max(hist_save_val_loss)
            
            plt.plot(hist_save_val_loss_plot, label= '# ' + str(mouseNum) + ' loss')
            plt.plot(hist_save_val_acc, label= '# ' + str(mouseNum) + ' acc')
            plt.legend()
            plt.savefig(RESULT_SAVE_PATH + str(mouseNum) + '_validationSet_result.png')
            plt.close()

            savename = RESULT_SAVE_PATH + str(mouseNum) + '_validationSet_result.csv'
            csvfile = open(savename, 'w', newline='')
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(hist_save_val_acc)
            csvwriter.writerow(hist_save_val_loss)
            csvfile.close()
            
    # In[] 수익률로 평가
for cv in range(5, cvnum+1):
    model = keras_setup() # for test
    final_weightsave = RESULT_SAVE_PATH + str(cv) + '_my_model_weights_final.h5'
    model.load_weights(final_weightsave)
    
    # In[]
    # 2019, 2018 data load
    test_only = True
    month_list = np.array([1215])
    test_year = 2019
    ml_dataset = genarator_by_year(test_year, month_list, test_only)
    
    predict_result = model.predict(ml_dataset[0])
    
    # cv tes t기능 추가 
 
#    profit_thr = 0.5
    for profit_thr in [0.5, 0.7, 0.9]:
        pratio = np.mean(predict_result[:,1] > profit_thr)
        print('pratio', pratio)
        go = np.where(predict_result[:,1] > profit_thr)[0]  # [0,1] good; [1,0] bad
        
        profit_save = np.zeros((go.shape[0], len(list(range(1,30))) )); profit_save[:] = np.nan
        for after_day in range(1,30):
            for i in range(go.shape[0]):
                mscode = ml_dataset[2][go[i]][:6]
                msdate = int(ml_dataset[2][go[i]][6:])
                raw = mssignalss[np.where(np.array(msindex) == mscode)[0][0]]
                
                ix2 = np.where(raw[:,0] > msdate)[0]
                buyix = np.min(ix2);
                
                buy_at = raw[buyix][1]
                sell_at = raw[buyix+after_day][1] # raw[:,1][buyix+after_day]
                profit = (sell_at/buy_at)-1
      
                profit_save[i,after_day-1] = profit
    
        mean_profit = np.mean(profit_save)
#        plt.imshow(profit_save)
        print('profit_thr', profit_thr, 'mean profit', mean_profit)
    
#    savename = RESULT_SAVE_PATH + str(cv) + '_profit.txt'
#    csvfile = open(savename, 'w', newline='')
#    csvwriter = csv.writer(csvfile)
#    csvwriter.writerow([mean_profit])
#    csvwriter.writerow([pratio])
#    csvfile.close()
#                
# In[]

# 중립 데이터에 대한 평가도 추가로 실시, validation만 믿지 말것
            # 수익모델과 연결 -> 평가로 수익모델로, 중립데이터도 포함하여 -> 결국 + - 몇 원으로 표시
# 2019 test 모델 추가 -> 추가
# data tracking, 종목별, 월별 정확도 분석 -> Z(종목), M(월)로 저장해놨음.
            
            # 컨트롤 수익모델
         
# 수익모델
# In[]
# 수익모델 최적화, 파는날짜 ,
            
# 2020년 data test
# cv test 모델에 2018년도 추가, 지금 cv는 중립데이터가 없음


# 독립코드로 사용할때 쓸것
model = keras_setup() # for test
final_weightsave = RESULT_SAVE_PATH + str(5) + '_my_model_weights_final.h5'
model.load_weights(final_weightsave)

month_list = np.array([201])
fortest = True
profit_thr = 0.50
Xtest, _, Ztest, Mtest = genarator_by_year(2020, dt_list2, X2, month_list)
print('test samples...', len(Xtest[0]))

predict_result = model.predict(Xtest)
   
print('ratio', np.mean(predict_result[:,1] > profit_thr))
go = np.where(predict_result[:,1] > profit_thr)[0]  # [0,1] good; [1,0] bad

profit_save = np.zeros((go.shape[0], len(list(range(1,31))) )); profit_save[:] = np.nan
for after_day in range(1,31):
    for i in range(go.shape[0]):
        mscode = Ztest[go[i]]
        msdate = Mtest[go[i]]
        ###몰랑 
        raw = mssignalss[np.where(np.array(msindex3) == mscode)[0][0]]
        buyix = np.where(raw[:,0]==msdate)[0]
        if len(buyix) != 0:
            buy_at = raw[buyix][0][1]
            sell_at = raw[buyix+after_day][0][1]
            profit = (sell_at/buy_at)-1
            profit_save[i,after_day-1] = profit
  
mean_profit = np.nanmean(profit_save)
print('mean profit', mean_profit)

savename = RESULT_SAVE_PATH + str(cv) + '_profit.txt'
csvfile = open(savename, 'w', newline='')
csvwriter = csv.writer(csvfile)
csvwriter.writerow([mean_profit])
csvfile.close()







       

   
        
        
        
    
    
    
    
    

    
















        
        
        
        
        
        
        
        
        
        
        
        
        







