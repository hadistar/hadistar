import pandas as pd
import random
import numpy as np

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Seoul.date = pd.to_datetime(Seoul.date)

df = Seoul[Seoul.date>'2018-01-01'].dropna()
df.to_csv('1_Basic_Seoul_raw.csv', index=False)

round(df.isnull().sum()/len(df)*100, 2).to_clipboard()
(df.dropna().mean()*1000).to_clipboard()

df.drop(columns=['date','PM2.5']).isnull().sum().mean()/len(df)*100

df.sample(int(len(df)*0.2), random_state=777)


BR = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='백령도')
BR.date = pd.to_datetime(BR.date, format="%y-%m-%d %H:%M:%S")

df = BR[BR.date>'2018-01-01'].dropna()
df.to_csv('1_Basic_BR_raw.csv', index=False)

df.sample(int(len(df)*0.2), random_state=777)


Ulsan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='영남권')
Ulsan.date = pd.to_datetime(Ulsan.date, format="%y-%m-%d %H:%M:%S")

df = Ulsan[Ulsan.date>'2018-01-01'].dropna()
df.to_csv('1_Basic_Ulsan_raw.csv', index=False)

df.sample(int(len(df)*0.2), random_state=777)

# 2

df = Seoul[Seoul.date>'2018-01-01'].dropna()
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour
df['weekday'] = df.date.dt.dayofweek+1
df.to_csv('2_Informed_Seoul_raw.csv', index=False)

df = BR[BR.date>'2018-01-01'].dropna()
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour
df['weekday'] = df.date.dt.dayofweek+1
df.to_csv('2_Informed_BR_raw.csv', index=False)

df = Ulsan[Ulsan.date>'2018-01-01'].dropna()
df['month'] = df.date.dt.month
df['hour'] = df.date.dt.hour
df['weekday'] = df.date.dt.dayofweek+1
df.to_csv('2_Informed_Ulsan_raw.csv', index=False)

# Seeds: 777,1004,322,224,417

# 3.Adding air pollution data

Seoul_2 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\2_Informed_1_Seoul_raw.csv')
BR_2 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\2_Informed_2_BR_raw.csv')
Ulsan_2 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\2_Informed_3_Ulsan_raw.csv')

# Nearest station finding

stations = pd.read_csv("D:\\OneDrive - SNU\\data\\AirKorea\\AirKorea_20191103_전국.csv", encoding='euc-kr')
locations = pd.read_excel('D:\\Dropbox\\패밀리룸\\MVI\\Data\\pm25speciation_locations_KoreaNational.xlsx')

Nearstations = []

for loc in ['Seoul','BR', 'Ulsan']:
    distance = []
    temp = locations.loc[locations['location']==loc]

    for i in stations.index:
        temp_dist = (temp.lat-stations.iloc[i].Latitude)**2+(temp.lon-stations.iloc[i].Longitude)**2
        distance.append(float(temp_dist))

    Nearstations.append(stations.iloc[np.argmin(distance)]['측정소코드'])

import os

AirKorea_2018_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2018')
AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')

AirKorea_2018 = pd.DataFrame()
AirKorea_2019 = pd.DataFrame()
AirKorea_2020 = pd.DataFrame()

Seoul = pd.DataFrame()
BR = pd.DataFrame()
Ulsan = pd.DataFrame()

for i in range(len(AirKorea_2018_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2018\\'+AirKorea_2018_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울
    BR = BR.append(temp.loc[temp['측정소코드'] == Nearstations[1]])  # 백령
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == Nearstations[2]])  # 울산

for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울
    BR = BR.append(temp.loc[temp['측정소코드'] == Nearstations[1]])  # 백령
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == Nearstations[2]])  # 울산


for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == Nearstations[0]])  # 서울
    BR = BR.append(temp.loc[temp['측정소코드'] == Nearstations[1]])  # 백령
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == Nearstations[2]])  # 울산


Seoul['temp'] = Seoul['측정일시'] - 1
Seoul['date'] = pd.to_datetime(Seoul['temp'], format='%Y%m%d%H')
Seoul['date'] = Seoul['date'] + pd.DateOffset(hours=1)

Seoul = Seoul[['date','PM25','PM10','SO2','CO','O3','NO2']]

Seoul_2['date'] = pd.to_datetime(Seoul_2['date'])
Seoul_2['date'] = Seoul_2['date'] - pd.to_timedelta(Seoul_2['date'].dt.second, unit='s')
Seoul_2['date'] = Seoul_2['date'].dt.floor('h')

Seoul_3 = pd.merge(Seoul_2, Seoul, how='inner',on='date')
Seoul_3 = Seoul_3.dropna()

Seoul_3.to_csv('3_AP_1_Seoul_raw.csv', index=False)



BR['temp'] = BR['측정일시'] - 1
BR['date'] = pd.to_datetime(BR['temp'], format='%Y%m%d%H')
BR['date'] = BR['date'] + pd.DateOffset(hours=1)

BR = BR[['date','PM25','PM10','SO2','CO','O3','NO2']]

BR_2['date'] = pd.to_datetime(BR_2['date'])
BR_2['date'] = BR_2['date'] - pd.to_timedelta(BR_2['date'].dt.second, unit='s')
BR_2['date'] = BR_2['date'].dt.floor('h')

BR_3 = pd.merge(BR_2, BR, how='inner',on='date')
BR_3 = BR_3.dropna()

BR_3.to_csv('3_AP_2_BR_raw.csv', index=False)


Ulsan['temp'] = Ulsan['측정일시'] - 1
Ulsan['date'] = pd.to_datetime(Ulsan['temp'], format='%Y%m%d%H')
Ulsan['date'] = Ulsan['date'] + pd.DateOffset(hours=1)

Ulsan = Ulsan[['date','PM25','PM10','SO2','CO','O3','NO2']]

Ulsan_2['date'] = pd.to_datetime(Ulsan_2['date'])
Ulsan_2['date'] = Ulsan_2['date'] - pd.to_timedelta(Ulsan_2['date'].dt.second, unit='s')
Ulsan_2['date'] = Ulsan_2['date'].dt.floor('h')

Ulsan_3 = pd.merge(Ulsan_2, Ulsan, how='inner',on='date')
Ulsan_3 = Ulsan_3.dropna()

Ulsan_3.to_csv('3_AP_3_Ulsan_raw.csv', index=False)


# 4. Meteological data add

Seoul_3 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\3_AP_1_Seoul_raw.csv')
BR_3 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\3_AP_2_BR_raw.csv')
Ulsan_3 = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\3_AP_3_Ulsan_raw.csv')

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

del Meteo_Seoul_2018, Meteo_Seoul_2019, Meteo_Seoul_2020

Meteo_BR_2018 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\BR_AWOS_hour\\SURFACE_ASOS_102_HR_2018_2018_2019.csv", encoding='euc-kr')
Meteo_BR_2019 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\BR_AWOS_hour\\SURFACE_ASOS_102_HR_2019_2019_2020.csv", encoding='euc-kr')
Meteo_BR_2020 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\BR_AWOS_hour\\SURFACE_ASOS_102_HR_2020_2020_2021.csv", encoding='euc-kr')

Meteo_BR = Meteo_BR_2018.append(Meteo_BR_2019).append(Meteo_BR_2020)
Meteo_BR = Meteo_BR.loc[:,['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
                           '이슬점온도(°C)', '현지기압(hPa)', '일조(hr)', '적설(cm)',
                           '전운량(10분위)', '시정(10m)', '지면온도(°C)']]
Meteo_BR.columns = ['date','temp','rainfall','ws','wd','RH','vapor',
                    'dewpoint','pressure','sunshine','snow',
                    'cloud','visibility','groundtemp']

Meteo_BR.isna().sum()
Meteo_BR[['rainfall','sunshine','snow']] = Meteo_BR[['rainfall','sunshine','snow']].fillna(0)

del Meteo_BR_2018, Meteo_BR_2019, Meteo_BR_2020



Meteo_Ulsan_2018 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Ulsan_AWOS_hour\\SURFACE_ASOS_152_HR_2018_2018_2019.csv", encoding='euc-kr')
Meteo_Ulsan_2019 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Ulsan_AWOS_hour\\SURFACE_ASOS_152_HR_2019_2019_2020.csv", encoding='euc-kr')
Meteo_Ulsan_2020 = pd.read_csv("D:\\OneDrive - SNU\\data\\Meteorological\\Ulsan_AWOS_hour\\SURFACE_ASOS_152_HR_2020_2020_2021.csv", encoding='euc-kr')

Meteo_Ulsan = Meteo_Ulsan_2018.append(Meteo_Ulsan_2019).append(Meteo_Ulsan_2020)
Meteo_Ulsan = Meteo_Ulsan.loc[:,['일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)',
                              '이슬점온도(°C)', '현지기압(hPa)', '일조(hr)', '적설(cm)',
                              '전운량(10분위)', '시정(10m)', '지면온도(°C)']]
Meteo_Ulsan.columns = ['date','temp','rainfall','ws','wd','RH','vapor',
                       'dewpoint','pressure','sunshine','snow',
                       'cloud','visibility','groundtemp']

Meteo_Ulsan.isna().sum()
Meteo_Ulsan[['rainfall','sunshine','snow']] = Meteo_Ulsan[['rainfall','sunshine','snow']].fillna(0)

del Meteo_Ulsan_2018, Meteo_Ulsan_2019, Meteo_Ulsan_2020

Seoul_3.date = pd.to_datetime(Seoul_3.date)
BR_3.date = pd.to_datetime(BR_3.date)
Ulsan_3.date = pd.to_datetime(Ulsan_3.date)

Meteo_Seoul.date = pd.to_datetime(Meteo_Seoul.date)
Meteo_BR.date = pd.to_datetime(Meteo_BR.date)
Meteo_Ulsan.date = pd.to_datetime(Meteo_Ulsan.date)


Seoul_4 = pd.merge(Seoul_3, Meteo_Seoul, how='inner',on='date').dropna()
BR_4 = pd.merge(BR_3, Meteo_BR, how='inner',on='date').dropna()
Ulsan_4 = pd.merge(Ulsan_3, Meteo_Ulsan, how='inner',on='date').dropna()


Seoul_4.to_csv('4_AP+Meteo_1_Seoul_raw.csv', index=False)
BR_4.to_csv('4_AP+Meteo_2_BR_raw.csv', index=False)
Ulsan_4.to_csv('4_AP+Meteo_3_Ulsan_raw.csv', index=False)
