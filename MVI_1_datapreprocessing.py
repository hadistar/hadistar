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