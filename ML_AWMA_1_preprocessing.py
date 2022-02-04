
# 2021-12-14
# MDL값 처리

import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Seoul.date = pd.to_datetime(Seoul.date)

# MDL값보다 낮은 비율 계산

MDLs = {'S':0.8, 'K':0.16, 'Ca':0.02, 'Ti':0.012, 'V':0.004, 'Cr':0.002, 'Mn':0.0012, 'Fe':0.0012, 'Ni':0.0008,
       'Cu':0.0008, 'Zn':0.0004, 'As':0.0004, 'Se':0.0012, 'Br':0.0016, 'Pb':0.0012}

# 기간별 자르기

df = Seoul[Seoul.date>'2016-01-01']
df = df[:-1]

#-----------------------------------------------
# 번외: For EDA

from pandas_profiling import ProfileReport


# EDA Report 생성
profile = ProfileReport(df,
            minimal=True,
            explorative=True,
            title='Data Profiling',
            plot={'histogram': {'bins': 8}},
            pool_size=4,
            progress_bar=False)

# Report 결과 경로에 저장
profile.to_file(output_file="data_profiling.html")

#-----------------------------------------------

# Ratio calculation of values below MDLs

for species in MDLs.keys():
    temp = df[df[species]<MDLs[species]].count()[species]
    print(species, round(temp/len(df)*100,2),"%")

#-------------------------------------------------
# Histogram

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.hist(df['Cr'], bins=200, range=[0,0.004])
plt.show()

#--------------------------------------------------

# Missing ratio calculation

print(round(df.isna().sum()/len(df)*100,2),"%")

#--------------------------------------------------


# MDL 이하값 MDLs*2로 대체..

# Trace elementals

for species in MDLs.keys():
    print(species)
    df[species].loc[df[species]<MDLs[species]] = MDLs[species] * 0.5

# Ions & carbons
df = df.reset_index(drop=True).copy()

MDLs = pd.read_excel('data\\Intensiv_Seoul_MDLs_2018-19.xlsx')
MDLs.date = pd.to_datetime(MDLs.date)

temp = df.copy()

columns = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+',
           'Ca2+', 'OC', 'EC', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni',
           'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

# MDL 이하값 비율 계산용

def calcul_mdls_ratio(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = 1
        elif row.loc[col] >= MDL.iloc[0][col]:
            temp.loc[col] = 0

    return temp

df3_MDL_bool =  df.apply(calcul_mdls_ratio, axis=1)

# 비율 체크

for species in df3_MDL_bool.columns:
    temp = df3_MDL_bool[df3_MDL_bool[species]==1].count()[species]
    print(species, round(temp/len(df)*100,2),"%")

#-------------------------------------------------------

# MDL 이하값 MDL*0.5 대체용

def calcul_mdls(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = MDL.iloc[0][col]*0.5

    return temp

df2 = df.apply(calcul_mdls, axis=1)

# Missing ratio calculation

print(round(df2.isna().sum()/len(df2)*100,2),"%")

df2.to_csv('AWMA_input_preprocessed_MDL_withNa.csv', index=False)

# df3: drop na values

# df3 = df2.dropna()
# df3.to_csv('AWMA_input_preprocessed_MDL_Na.csv', index=False)


#-----------------------------------------------------------------------
# 2021-12-15
# Input 자료에 기상, 대기오염 자료 붙이기

import pandas as pd
import numpy as np


# 1. AirKorea 자료 붙이기

# 1-1. 집중측정소와 가장 가까운 AirKorea Station 찾기

stations = pd.read_csv("D:\\OneDrive - SNU\\data\\AirKorea\\AirKorea_20191103_전국.csv", encoding='euc-kr')
locations = pd.read_excel('D:\\Dropbox\\패밀리룸\\MVI\\Data\\pm25speciation_locations_KoreaNational.xlsx')

Nearstations = []

for loc in ['Seoul']:
    distance = []
    temp = locations.loc[locations['location']==loc]

    for i in stations.index:
        temp_dist = (temp.lat-stations.iloc[i].Latitude)**2+(temp.lon-stations.iloc[i].Longitude)**2
        distance.append(float(temp_dist))

    Nearstations.append(stations.iloc[np.argmin(distance)]['측정소코드'])

import os

# 1-2. 해당 지점의 AirKorea 자료 불러오기

AirKorea_2018_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2018')
AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')

AirKorea_2018 = pd.DataFrame()
AirKorea_2019 = pd.DataFrame()
AirKorea_2020 = pd.DataFrame()

Seoul = pd.DataFrame()

for i in range(len(AirKorea_2018_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2018\\'+AirKorea_2018_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울


# 1-3. 시간 형식 통일시키기
# 시간 기준으로 합치기 위해 시간 형식으로 자료 변환

Seoul['temp'] = Seoul['측정일시'] - 1
Seoul['date'] = pd.to_datetime(Seoul['temp'], format='%Y%m%d%H')
Seoul['date'] = Seoul['date'] + pd.DateOffset(hours=1)

# 1-4. 쓸 자료만 추리기
Seoul = Seoul[['date','PM25','PM10','SO2','CO','O3','NO2']]


# 1-5. 기존 자료와 합치기

df = pd.read_csv('AWMA_input_preprocessed_MDL_withNa.csv')
df['date'] = pd.to_datetime(df['date'])
df = df[df.date>'2018-01-01']

## 이상하게 붙은 초 단위 시간 지우기
df['date'] = df['date'] - pd.to_timedelta(df['date'].dt.second, unit='s')
df['date'] = df['date'].dt.floor('h')

# 합치기
df2 = pd.merge(df, Seoul, how='inner',on='date')

# 저장
df2.to_csv('AWMA_input_preprocessed_MDL_AirKorea_withNa.csv', index=False)

del Seoul

# 2. 기상자료 붙이기

# 2-1. 기상자료 불러오기, 서울 AWOS 지점코드 108
Meteo_Seoul_2018 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2018_2018_2019.csv", encoding='euc-kr')
Meteo_Seoul_2019 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2019_2019_2020.csv", encoding='euc-kr')
Meteo_Seoul_2020 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2020_2020_2021.csv", encoding='euc-kr')

Meteo_Seoul = Meteo_Seoul_2018.append(Meteo_Seoul_2019).append(Meteo_Seoul_2020)

Meteo_Seoul = Meteo_Seoul.loc[:,['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
                                 '이슬점온도(°C)', '현지기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)',
                                 '전운량(10분위)', '시정(10m)', '지면온도(°C)', '30cm 지중온도(°C)']]
Meteo_Seoul.columns = ['date','temp','rainfall','ws','wd','RH','vapor',
                       'dewpoint','pressure','sunshine','insolation','snow',
                       'cloud','visibility','groundtemp','30cmtemp']
Meteo_Seoul.isna().sum()
Meteo_Seoul[['rainfall','sunshine','insolation','snow']] = Meteo_Seoul[['rainfall','sunshine','insolation','snow']].fillna(0)
Meteo_Seoul.isna().sum()

# 불필요한 변수 삭제
del Meteo_Seoul_2018, Meteo_Seoul_2019, Meteo_Seoul_2020

# 시간 기준으로 합치기 위해 시간 형식으로 자료 변환
df2.date = pd.to_datetime(df2.date)
Meteo_Seoul.date = pd.to_datetime(Meteo_Seoul.date)

# 합치기

df3 = pd.merge(df2, Meteo_Seoul, how='inner',on='date')

# 저장

df3.to_csv('AWMA_input_preprocessed_MDL_AirKorea_Meteo_2018_2020_withNa.csv', index=False)

# 결측비율 체크

print(df3.isna().sum()/len(df3)*100)

# <2021-12-16>
# 등간격 자료 만들기

import pandas as pd
df3 = pd.read_csv('AWMA_input_preprocessed_MDL_AirKorea_Meteo_2018_2020_withNa.csv')
df3.date = pd.to_datetime(df3.date)

# 1. 3일 간격
dates = pd.date_range(start = '2018-01-01',end='2020-12-31', freq="D")

dates = pd.DataFrame(dates)
dates = dates.rename(columns={0:'date'})
dates['input'] = 0

dates.iloc[::3,:]['input'] = 1

train = pd.DataFrame()
test = pd.DataFrame()

for i in range(len(df3)):
    day = str(df3.iloc[i]['date'].year)+'-'+str(df3.iloc[i]['date'].month)+'-'+str(df3.iloc[i]['date'].day)
    if dates.loc[dates.date==day]['input'].values == 1:
        train = train.append(df3.iloc[i])
    else:
        test = test.append(df3.iloc[i])

train.to_csv('AWMA_3days_train.csv', index=False)
test.to_csv('AWMA_3days_test.csv', index=False)

# 2. 4일 간격

dates = pd.date_range(start = '2018-01-01',end='2020-12-31', freq="D")

dates = pd.DataFrame(dates)
dates = dates.rename(columns={0:'date'})
dates['input'] = 0

dates.iloc[::4,:]['input'] = 1

train = pd.DataFrame()
test = pd.DataFrame()

for i in range(len(df3)):
    day = str(df3.iloc[i]['date'].year)+'-'+str(df3.iloc[i]['date'].month)+'-'+str(df3.iloc[i]['date'].day)
    if dates.loc[dates.date==day]['input'].values == 1:
        train = train.append(df3.iloc[i])
    else:
        test = test.append(df3.iloc[i])

train.to_csv('AWMA_4days_train.csv', index=False)
test.to_csv('AWMA_4days_test.csv', index=False)

# 3. 5일 간격

dates = pd.date_range(start = '2018-01-01',end='2020-12-31', freq="D")

dates = pd.DataFrame(dates)
dates = dates.rename(columns={0:'date'})
dates['input'] = 0

dates.iloc[::5,:]['input'] = 1

train = pd.DataFrame()
test = pd.DataFrame()

for i in range(len(df3)):
    day = str(df3.iloc[i]['date'].year)+'-'+str(df3.iloc[i]['date'].month)+'-'+str(df3.iloc[i]['date'].day)
    if dates.loc[dates.date==day]['input'].values == 1:
        train = train.append(df3.iloc[i])
    else:
        test = test.append(df3.iloc[i])

train.to_csv('AWMA_5days_train.csv', index=False)
test.to_csv('AWMA_5days_test.csv', index=False)


#
### 자료 재구성 ### 3일 간격 예측으로..
#
# 2017, 2018 training
# 2019 test ?
#
# 일평균으로,, 단, 75% 이상 있는 것만 일평균값으로 사용

# <2021-12-21, 자료 재구성>


import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Seoul.date = pd.to_datetime(Seoul.date)


# 기간별 자르기

df = Seoul[Seoul.date>'2015-01-01']
df = df[:-1]



# MDL 이하값 MDLs*2로 대체..

# Ions & carbons & elementals
df = df.reset_index(drop=True).copy()

MDLs = pd.read_excel('data\\Intensiv_Seoul_MDLs_2018-19.xlsx')
MDLs.date = pd.to_datetime(MDLs.date)

columns = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+',
           'Ca2+', 'OC', 'EC', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni',
           'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

# MDL 이하값 비율 계산용

def calcul_mdls_ratio(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = 1
        elif row.loc[col] >= MDL.iloc[0][col]:
            temp.loc[col] = 0

    return temp

df3_MDL_bool =  df.apply(calcul_mdls_ratio, axis=1)

# 비율 체크

for species in df3_MDL_bool.columns:
    temp = df3_MDL_bool[df3_MDL_bool[species]==1].count()[species]
    print(species, round(temp/len(df)*100,2),"%")

#-------------------------------------------------------

# MDL 이하값 MDL*0.5 대체용

def calcul_mdls(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = MDL.iloc[0][col]*0.5

    return temp

# df = df[:-1]

df2 = df.apply(calcul_mdls, axis=1)
(df2.median()*1000).to_clipboard()

## 이상하게 붙은 초 단위 시간 지우기
df2['date'] = df2['date'] - pd.to_timedelta(df2['date'].dt.second, unit='s')
df2['date'] = df2['date'].dt.floor('h')


# Missing ratio calculation

print(round(df2.isna().sum()/len(df2)*100,2),"%")


dates = pd.date_range(start = '2015-01-01',end='2020-12-31', freq="D")
dates = pd.DataFrame(dates)

df3 = pd.DataFrame()

# Missing이 25% 초과인 자료는 제외하고 일평균 구하기, 한 columns이라도 25% 초과시 모두 삭제

for day in dates[0]:
    temp = df2.loc[(day.year == df2.date.dt.year) & (day.month == df2.date.dt.month) & (day.day == df2.date.dt.day)]
    if temp.isna().sum().max() > 6:
        pass
    else:
        df3 = df3.append(temp)

df4 = df3.groupby(pd.Grouper(freq='D', key='date')).mean()
df5 = df4.dropna()



# <2021-12-22>
# Data combining


# Input 자료에 기상, 대기오염 자료 붙이기

import pandas as pd
import numpy as np

# 1. AirKorea 자료 붙이기

# 1-1. 집중측정소와 가장 가까운 AirKorea Station 찾기

stations = pd.read_csv("D:\\OneDrive - SNU\\data\\AirKorea\\AirKorea_20191103_전국.csv", encoding='euc-kr')
locations = pd.read_excel('D:\\Dropbox\\패밀리룸\\MVI\\Data\\pm25speciation_locations_KoreaNational.xlsx')

Nearstations = []

for loc in ['Seoul']:
    distance = []
    temp = locations.loc[locations['location']==loc]

    for i in stations.index:
        temp_dist = (temp.lat-stations.iloc[i].Latitude)**2+(temp.lon-stations.iloc[i].Longitude)**2
        distance.append(float(temp_dist))

    Nearstations.append(stations.iloc[np.argmin(distance)]['측정소코드'])

import os

# 1-2. 해당 지점의 AirKorea 자료 불러오기

AirKorea_2015_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2015')
AirKorea_2016_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2016')
AirKorea_2017_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2017')
AirKorea_2018_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2018')
AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')

AirKorea_2015 = pd.DataFrame()
AirKorea_2016 = pd.DataFrame()
AirKorea_2017 = pd.DataFrame()
AirKorea_2018 = pd.DataFrame()
AirKorea_2019 = pd.DataFrame()
AirKorea_2020 = pd.DataFrame()

Seoul = pd.DataFrame()

for i in range(len(AirKorea_2015_list)):
    temp = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea\\2015\\'+AirKorea_2015_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2016_list)):
    temp = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea\\2016\\'+AirKorea_2016_list[i], encoding='euc-kr')
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2017_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2017\\'+AirKorea_2017_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2018_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2018\\'+AirKorea_2018_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울

for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울


# 1-3. 시간 형식 통일시키기
# 시간 기준으로 합치기 위해 시간 형식으로 자료 변환

Seoul['temp'] = Seoul['측정일시'] - 1
Seoul['date'] = pd.to_datetime(Seoul['temp'], format='%Y%m%d%H')
Seoul['date'] = Seoul['date'] + pd.DateOffset(hours=1)

# 1-4. 쓸 자료만 추리기
Seoul = Seoul[['date','PM25','PM10','SO2','CO','O3','NO2']]


# 일평균으로 바꾸기

# Missing이 25% 초과인 자료는 제외하고 일평균 구하기, 한 columns이라도 25% 초과시 모두 삭제

Seoul_day = pd.DataFrame()

for day in dates[0]:
    temp = Seoul.loc[(day.year == Seoul.date.dt.year) & (day.month == Seoul.date.dt.month) & (day.day == Seoul.date.dt.day)]
    if temp.isna().sum().max() > 6:
        pass
    else:
        Seoul_day = Seoul_day.append(temp)

Seoul_day = Seoul_day.groupby(pd.Grouper(freq='D', key='date')).mean()


# 1-5. 기존 자료와 합치기
# 합치기
df6 = pd.merge(df4, Seoul_day, how='inner',on='date')

# 2. 기상자료 붙이기

# 2-1. 기상자료 불러오기, 서울 AWOS 지점코드 108
Meteo_Seoul_2015 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2015_2015_2018.csv", encoding='euc-kr')
Meteo_Seoul_2016 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2016_2016_2017.csv", encoding='euc-kr')
Meteo_Seoul_2017 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2017_2017_2018.csv", encoding='euc-kr')
Meteo_Seoul_2018 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2018_2018_2019.csv", encoding='euc-kr')
Meteo_Seoul_2019 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2019_2019_2020.csv", encoding='euc-kr')
Meteo_Seoul_2020 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Seoul_AWOS_hour\\SURFACE_ASOS_108_HR_2020_2020_2021.csv", encoding='euc-kr')

Meteo_Seoul = Meteo_Seoul_2015.append(Meteo_Seoul_2016.append(Meteo_Seoul_2017.append(Meteo_Seoul_2018.append(Meteo_Seoul_2019).append(Meteo_Seoul_2020))))


Meteo_Seoul = Meteo_Seoul.loc[:,['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
                                 '이슬점온도(°C)', '현지기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)',
                                 '전운량(10분위)', '시정(10m)', '지면온도(°C)', '30cm 지중온도(°C)']]
Meteo_Seoul.columns = ['date','temp','rainfall','ws','wd','RH','vapor',
                       'dewpoint','pressure','sunshine','insolation','snow',
                       'cloud','visibility','groundtemp','30cmtemp']
Meteo_Seoul.isna().sum()
Meteo_Seoul[['rainfall','sunshine','insolation','snow']] = Meteo_Seoul[['rainfall','sunshine','insolation','snow']].fillna(0)
Meteo_Seoul.isna().sum()

# 불필요한 변수 삭제
del Meteo_Seoul_2018, Meteo_Seoul_2019, Meteo_Seoul_2020

# 시간 기준으로 합치기 위해 시간 형식으로 자료 변환
Meteo_Seoul.date = pd.to_datetime(Meteo_Seoul.date)


# 일평균으로 바꾸기

# Missing이 25% 초과인 자료는 제외하고 일평균 구하기, 한 columns이라도 25% 초과시 모두 삭제

Meteo_Seoul_day = pd.DataFrame()

for day in dates[0]:
    temp = Meteo_Seoul.loc[(day.year == Meteo_Seoul.date.dt.year) & (day.month == Meteo_Seoul.date.dt.month) & (day.day == Meteo_Seoul.date.dt.day)]
    if temp.isna().sum().max() > 6:
        pass
    else:
        Meteo_Seoul_day = Meteo_Seoul_day.append(temp)

Meteo_Seoul_day = Meteo_Seoul_day.groupby(pd.Grouper(freq='D', key='date')).mean()

# 합치기

df7 = pd.merge(df6, Meteo_Seoul_day, how='inner',on='date')

df7.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea.csv', index=False)


# Case 1: 3일 간격 자료로 입력자료 구성하기, 2017, 2018 training, 2019 test -> 예측 정확도 낮음!

df7 = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea.csv')
df7.date = pd.to_datetime(df7.date)

df8 = pd.DataFrame()
for day in df7.date:

    target_day1 = day +  pd.Timedelta(days=3)
    target_day2 = day + pd.Timedelta(days=6)
    temp = df7.loc[(day.year == df7.date.dt.year) & (day.month == df7.date.dt.month) & (day.day == df7.date.dt.day)]
    temp = temp.append(df7.loc[(target_day1.year == df7.date.dt.year) & (target_day1.month == df7.date.dt.month) & (target_day1.day == df7.date.dt.day)])
    temp = temp.append(df7.loc[(target_day2.year == df7.date.dt.year) & (target_day2.month == df7.date.dt.month) & (target_day2.day == df7.date.dt.day)])

    if temp.isna().sum().sum() != 0:
        pass
    else:
        df8 = df8.append(temp)

df8.loc[df8.date.dt.year == 2015] #6
df8.loc[df8.date.dt.year == 2016] #57
df8.loc[df8.date.dt.year == 2017] #52
df8.loc[df8.date.dt.year == 2018] #99
df8.loc[df8.date.dt.year == 2019] #125
df8.loc[df8.date.dt.year == 2020] #144

df8=df8.reset_index(drop=True)

df8['month'] = df8.date.dt.month
df8['weekday'] = df8.date.dt.weekday +1

df7['month'] = df7.date.dt.month
df7['weekday'] = df7.date.dt.weekday +1

df8.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case1.csv', index=False)




# Case 2: 3일 연속 자료로 입력자료 구성하기, 2017, 2018 training, 2019 test
df7 = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea.csv')
df7.date = pd.to_datetime(df7.date)

df8 = pd.DataFrame()
for day in df7.date:

    target_day1 = day +  pd.Timedelta(days=1)
    target_day2 = day + pd.Timedelta(days=2)
    temp = df7.loc[(day.year == df7.date.dt.year) & (day.month == df7.date.dt.month) & (day.day == df7.date.dt.day)]
    temp = temp.append(df7.loc[(target_day1.year == df7.date.dt.year) & (target_day1.month == df7.date.dt.month) & (target_day1.day == df7.date.dt.day)])
    temp = temp.append(df7.loc[(target_day2.year == df7.date.dt.year) & (target_day2.month == df7.date.dt.month) & (target_day2.day == df7.date.dt.day)])

    if temp.isna().sum().sum() != 0:
        pass
    else:
        df8 = df8.append(temp)

df8.loc[df8.date.dt.year == 2015] #5
df8.loc[df8.date.dt.year == 2016] #84
df8.loc[df8.date.dt.year == 2017] #72
df8.loc[df8.date.dt.year == 2018] #137
df8.loc[df8.date.dt.year == 2019] #171
df8.loc[df8.date.dt.year == 2020] #193

df8=df8.reset_index(drop=True)

df8['month'] = df8.date.dt.month
df8['weekday'] = df8.date.dt.weekday +1

df8.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case2.csv', index=False)



# Case 3: 2일 연속 자료로 입력자료 구성하기, 2017, 2018 training, 2019 test
df7 = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea.csv')
df7.date = pd.to_datetime(df7.date)

df8 = pd.DataFrame()
for day in df7.date:

    target_day1 = day +  pd.Timedelta(days=1)
    temp = df7.loc[(day.year == df7.date.dt.year) & (day.month == df7.date.dt.month) & (day.day == df7.date.dt.day)]
    temp = temp.append(df7.loc[(target_day1.year == df7.date.dt.year) & (target_day1.month == df7.date.dt.month) & (target_day1.day == df7.date.dt.day)])

    if temp.isna().sum().sum() != 0:
        pass
    else:
        df8 = df8.append(temp)

df8=df8.reset_index(drop=True)

df8['month'] = df8.date.dt.month
df8['weekday'] = df8.date.dt.weekday +1

df8.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case3.csv', index=False)


# Case 4: 1일 연속 자료로 입력자료 구성하기, 2017, 2018 training, 2019 test
df7 = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea.csv')
df7.date = pd.to_datetime(df7.date)

df8 = df7.dropna()

df8=df8.reset_index(drop=True)

df8['month'] = df8.date.dt.month
df8['weekday'] = df8.date.dt.weekday +1

df8.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case4.csv', index=False)



## 2021-12-29, 서울 광공업생산지수 추가해서 Case 6 만들기

df = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case4.csv')
df=df.drop(columns=['PM25', 'groundtemp', '30cmtemp','dewpoint','wd','sunshine','insolation'])
df.date = pd.to_datetime(df.date)

df2 = pd.read_excel('d:\\Dropbox\\Shared_Air_ML\\ref.data\\시도_광공업생산지수_월별_2015-2020.xlsx', sheet_name='데이터')
df2.date = pd.to_datetime(df2.date)

df6 = pd.DataFrame()

for day in df.date:

    temp = df.loc[df.date==day]
    temp2 = df2.loc[(day.year == df2.date.dt.year) & (day.month == df2.date.dt.month)]['서울특별시'].values
    temp['production_index'] = temp2
    df6 = df6.append(temp)

df6.to_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case6.csv', index=False)