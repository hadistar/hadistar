import os
import pandas as pd


import numpy as np
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3) #KNN

a = np.array([[1,2,3],
              [1,np.nan,3],
              [2,3,4],
              [3,4,5]])
b = imputer.fit_transform(a)

print(b)

airpollution = pd.read_csv("Seoul_preprocessed_withoutKNN_hourly.csv")
airpollution = airpollution.drop(['location', 'lat','lon'], axis=1)


meteo_seoul_2016 = pd.read_csv("Meteo_raw/SURFACE_ASOS_108_HR_2016.csv", encoding='EUC_KR')
meteo_seoul_2017 = pd.read_csv("Meteo_raw/SURFACE_ASOS_108_HR_2017.csv", encoding='EUC_KR')
meteo_seoul_2018 = pd.read_csv("Meteo_raw/SURFACE_ASOS_108_HR_2018.csv", encoding='EUC_KR')
meteo_seoul_2019 = pd.read_csv("Meteo_raw/SURFACE_ASOS_108_HR_2019.csv", encoding='EUC_KR')

meteo_gwanak_2016 = pd.read_csv("Meteo_raw/SURFACE_AWS_116_HR_2016.csv", encoding='EUC_KR')
meteo_gwanak_2017 = pd.read_csv("Meteo_raw/SURFACE_AWS_116_HR_2017.csv", encoding='EUC_KR')
meteo_gwanak_2018 = pd.read_csv("Meteo_raw/SURFACE_AWS_116_HR_2018.csv", encoding='EUC_KR')
meteo_gwanak_2019 = pd.read_csv("Meteo_raw/SURFACE_AWS_116_HR_2019.csv", encoding='EUC_KR')

meteo_seoul = pd.concat([meteo_seoul_2016, meteo_seoul_2017, meteo_seoul_2018, meteo_seoul_2019], axis=0)
meteo_gwanak = pd. concat([meteo_gwanak_2016,meteo_gwanak_2017,meteo_gwanak_2018,meteo_gwanak_2019], axis=0)

del(meteo_seoul_2016, meteo_seoul_2017, meteo_seoul_2018, meteo_seoul_2019)
del(meteo_gwanak_2016,meteo_gwanak_2017,meteo_gwanak_2018,meteo_gwanak_2019)

# null check for Seoul data
meteo_seoul.isnull().sum()
meteo_seoul = meteo_seoul.drop(meteo_seoul.columns[[0,11,12,14,15,17,18,20,21]], axis=1)
meteo_seoul.isnull().sum()
meteo_seoul.columns = ['date','temp_s','rainfall_s','ws_s','wd_s','rh_s',
                       'vapor_pressure_s','dew_point_s','pressure_s','sea_pressure_s',
                       'snow_s','cloud_s','visibility_s','temp_ground_s','temp_5cm_s',
                       'temp_10cm_s','temp_20cm_s','temp_30cm_s']
meteo_seoul['rainfall_s'] = meteo_seoul['rainfall_s'].fillna(0)
meteo_seoul['snow_s'] = meteo_seoul['snow_s'].fillna(0)


meteo_seoul.iloc[:,1:] = imputer.fit_transform(meteo_seoul.iloc[:,1:])
meteo_seoul.isnull().sum()

meteo_gwanak.isnull().sum()
meteo_gwanak = meteo_gwanak.drop(meteo_gwanak.columns[[0,6,7,9,10]], axis=1)
meteo_gwanak.columns = ['date', 'temp_g', 'wd_g','ws_g','rainfall_g','rh_g']

meteo_gwanak.iloc[:,1:] = imputer.fit_transform(meteo_gwanak.iloc[:,1:])
meteo_gwanak.isnull().sum()

meteo = pd.merge(meteo_seoul,meteo_gwanak, how='outer', on='date')

airpollution['date']=pd.to_datetime(airpollution['date'], format= '%Y-%m-%d %H')
meteo['date']=pd.to_datetime(meteo['date'], format= '%Y-%m-%d %H')

# For AirKorea data

jg_2016_list = os.listdir('AirKorea/Junggu/2016')
jg_2017_list = os.listdir('AirKorea/Junggu/2017')
jg_2018_list = os.listdir('AirKorea/Junggu/2018')
jg_2019_list = os.listdir('AirKorea/Junggu/2019')

jg_2016 = pd.DataFrame()
jg_2017 = pd.DataFrame()
jg_2018 = pd.DataFrame()
jg_2019 = pd.DataFrame()

for i in range(len(jg_2016_list)):
    temp = pd.read_excel('AirKorea/Junggu/2016/'+jg_2016_list[i])
    temp = temp.drop(0)
    temp.columns=['date', 'pm10', 'pm25','o3','no2','co','so2']
    temp['date'] = '2016-'+temp['date']
    temp['date'] = pd.to_datetime(temp['date'], format ="%Y-%m-%d-%H", exact=False)
    jg_2016 = jg_2016.append(temp)

for i in range(len(jg_2017_list)):
    temp = pd.read_excel('AirKorea/Junggu/2017/'+jg_2017_list[i])
    temp = temp.drop(0)
    temp.columns=['date', 'pm10', 'pm25','o3','no2','co','so2']
    temp['date'] = '2017-'+temp['date']
    temp['date'] = pd.to_datetime(temp['date'], format ="%Y-%m-%d-%H", exact=False)
    jg_2017 = jg_2017.append(temp)

for i in range(len(jg_2018_list)):
    temp = pd.read_excel('AirKorea/Junggu/2018/'+jg_2018_list[i])
    temp = temp.drop(0)
    temp.columns=['date', 'pm10', 'pm25','o3','no2','co','so2']
    temp['date'] = '2018-'+temp['date']
    temp['date'] = pd.to_datetime(temp['date'], format ="%Y-%m-%d-%H", exact=False)
    jg_2018 = jg_2018.append(temp)

for i in range(len(jg_2019_list)):
    temp = pd.read_excel('AirKorea/Junggu/2019/'+jg_2019_list[i])
    temp = temp.drop(0)
    temp.columns=['date', 'pm10', 'pm25','o3','no2','co','so2']
    temp['date'] = '2019-'+temp['date']
    temp['date'] = pd.to_datetime(temp['date'], format ="%Y-%m-%d-%H", exact=False)
    jg_2019 = jg_2019.append(temp)

jg = pd.concat([jg_2016,jg_2017,jg_2018,jg_2019], axis=0)
jg.columns = ['date', 'pm10_jg', 'pm25_jg','o3_jg','no2_jg','co_jg','so2_jg']

data = pd.merge(airpollution, meteo, how='outer', on='date')
data = pd.merge(data, jg, how='outer', on='date')

data.to_csv('data_withNa.csv', index=False)
data = data.dropna(axis=0)

data['hour'] = data['date'].dt.hour
data['weekday'] = data['date'].dt.weekday

data.to_csv('data.csv', index=False)
