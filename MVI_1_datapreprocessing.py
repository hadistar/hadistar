import pandas as pd
import random

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

# Nearest location finding

stations = pd.read_csv("D:\\OneDrive - SNU\\data\\AirKorea\\AirKorea_20191103_전국.csv", encoding='euc-kr')





import os

AirKorea_2018_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2018')
AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')

AirKorea_2018 = pd.DataFrame()
AirKorea_2019 = pd.DataFrame()
AirKorea_2020 = pd.DataFrame()

Seoul = pd.DataFrame()
Incheon = pd.DataFrame()
Yeosu = pd.DataFrame()
Siheung = pd.DataFrame()
Daebu = pd.DataFrame()
Ulsan = pd.DataFrame()

for i in range(len(AirKorea_2018_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2018\\'+AirKorea_2018_list[i])
    Siheung = Siheung.append(temp.loc[temp['측정소코드'] == 131231])  # 시흥 측정소
    Incheon = Incheon.append(temp.loc[temp['측정소코드'] == 823671])  # 인천 남동공단
    Yeosu = Yeosu.append(temp.loc[temp['측정소코드'] == 336124])  # 여수산단로 1201
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == 111125])  # 서울 종로
    Daebu = Daebu.append(temp.loc[temp['측정소코드'] == 131196])  # 대부도
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == 238123])  # 울산 남구 부두로 9 (울산 산업단지)

for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    Siheung = Siheung.append(temp.loc[temp['측정소코드'] == 131231])  # 시흥 측정소
    Incheon = Incheon.append(temp.loc[temp['측정소코드'] == 823671])  # 인천 남동공단
    Yeosu = Yeosu.append(temp.loc[temp['측정소코드'] == 336124])  # 여수산단로 1201
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == 111125])  # 서울 종로
    Daebu = Daebu.append(temp.loc[temp['측정소코드'] == 131196])  # 대부도
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == 238123])  # 울산 남구 부두로 9 (울산 산업단지)

for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    Siheung = Siheung.append(temp.loc[temp['측정소코드'] == 131231])  # 시흥 측정소
    Incheon = Incheon.append(temp.loc[temp['측정소코드'] == 823671])  # 인천 남동공단
    Yeosu = Yeosu.append(temp.loc[temp['측정소코드'] == 336124])  # 여수산단로 1201
    Seoul = Seoul.append(temp.loc[temp['측정소코드'] == 111125])  # 서울 종로
    Daebu = Daebu.append(temp.loc[temp['측정소코드'] == 131196])  # 대부도
    Ulsan = Ulsan.append(temp.loc[temp['측정소코드'] == 238123])  # 울산 남구 부두로 9 (울산 산업단지)
