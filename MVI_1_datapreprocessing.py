import pandas as pd
import random
import numpy as np

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Seoul.date = pd.to_datetime(Seoul.date)

df = Seoul[Seoul.date>'2018-01-01'].dropna()
df.to_csv('1_Basic_Seoul_raw.csv', index=False)

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

