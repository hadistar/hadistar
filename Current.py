import pandas as pd
import os
from math import cos, sin, asin, sqrt, radians

def calc_distance(row):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lat1 = 37.3468122
    lon1 = 126.7400834
    lat2 = row['Latitude']
    lon2 = row['Longitude']

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

df1 = pd.DataFrame()
df2 = pd.DataFrame()

AirKorea_2019_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2019')
AirKorea_2020_list = os.listdir('D:\\OneDrive - SNU\\data\\AirKorea\\2020')


for i in range(len(AirKorea_2019_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2019\\'+AirKorea_2019_list[i])
    df1 = df1.append(temp)

for i in range(len(AirKorea_2020_list)):
    temp = pd.read_excel('D:\\OneDrive - SNU\\data\\AirKorea\\2020\\'+AirKorea_2020_list[i])
    df2 = df2.append(temp)


df = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea_20191103.csv', encoding='euc-kr')

lat1 = 37.3468122
lon1 = 126.7400834

df['distance'] = df.apply(calc_distance, axis=1)
df = df.loc[df.distance<=100]


temp = df1.append(df2)

temp['temp'] = temp['측정일시'] - 1
temp['date'] = pd.to_datetime(temp['temp'], format='%Y%m%d%H')
temp['date'] = temp['date'] + pd.DateOffset(hours=1)

data = pd.DataFrame()

for i, row in df.iterrows():
    target = temp.loc[temp['측정소코드']==row['측정소코드']]
    target = target.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
    target['lat'] = row['Latitude']
    target['lon'] = row['Longitude']
    target['address'] = row['주소']
    target['distance'] = row['distance']

    data = data.append(target)

data.to_csv('AirKora_2019_2020_SH_100km.csv', encoding='euc-kr')






from plotnine.data import mpg
from plotnine import *

ggplot(mpg) + aes(x="class") + geom_bar()
