import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import math

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
import seaborn as sns

# For Fig. 1. Comparison between cities

## 1. AQI to mass concentration

Beijing = pd.read_csv('D:\\Data_backup\\Dropbox\\PMF_paper\\data_YSLEE\\beijing-us embassy-air-quality.csv')
Beijing['date'] = pd.to_datetime(Beijing['date'])
Beijing[' pm25'] = pd.to_numeric(Beijing[' pm25'], errors='coerce')


Hamburg = pd.read_csv('D:\\Data_backup\\Dropbox\\PMF_paper\\data_YSLEE\\Hamburg, germany-air-quality.csv')
Hamburg['date'] = pd.to_datetime(Hamburg['date'])
Hamburg[' pm25'] = pd.to_numeric(Hamburg[' pm25'], errors='coerce')

Shanghai = pd.read_csv('D:\\Data_backup\\Dropbox\\PMF_paper\\data_YSLEE\\shanghai-air-quality.csv')
Shanghai['date'] = pd.to_datetime(Shanghai['date'])
Shanghai[' pm25'] = pd.to_numeric(Shanghai[' pm25'], errors='coerce')

Shenzhen = pd.read_csv('D:\\Data_backup\\Dropbox\\PMF_paper\\data_YSLEE\\shenzhen-air-quality.csv')
Shenzhen['date'] = pd.to_datetime(Shenzhen['date'])
Shenzhen[' pm25'] = pd.to_numeric(Shenzhen[' pm25'], errors='coerce')

# AQI and PM2.5 conversion, US EPA calculation method

def to_con(row):
    # conc. 0-12 -> AQI 0-50,
    # conc. 12-35.5 -> AQI 50-100,
    # conc. 35.5-55.5 -> AQI 100-150,
    # conc. 55.5-150.5 -> AQI 150-200,
    # conc. 150.5-250.5 -> AQI 200-300,
    # conc. 250.5-350.5 -> AQI 300-400,
    # conc. 350.5-500.5 -> AQI 400-500,
    aqi = row[' pm25']
    val = np.nan
    if aqi <= 50:
        val= aqi/50 * 12
    elif aqi <= 100:
        val = (aqi-50) / (100-50) * (35.5-12) + 12
    elif aqi <= 150:
        val = (aqi-100) / (150-100) * (55.5-35.5) + 35.5
    elif aqi <= 200:
        val = (aqi-150) / (200-150) * (150.5-55.5) + 55.5
    elif aqi <= 300:
        val = (aqi-200) / (300-200) * (250.5-150.5) + 150.5
    elif aqi <= 400:
        val = (aqi-300) / (400-300) * (350.5-250.5) + 250.5
    elif aqi <= 500:
        val = (aqi-400) / (500-400) * (500.5-350.5) + 350.5
    return val

Beijing['pm25_conc'] = Beijing.apply(to_con, axis=1)
Beijing = Beijing.sort_values(by='date')
Hamburg['pm25_conc'] = Hamburg.apply(to_con, axis=1)
Hamburg = Hamburg.sort_values(by='date')
Shanghai['pm25_conc'] = Shanghai.apply(to_con, axis=1)
Shanghai = Shanghai.sort_values(by='date')
Shenzhen['pm25_conc'] = Shenzhen.apply(to_con, axis=1)
Shenzhen = Shenzhen.sort_values(by='date')

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


Siheung['date'] = pd.to_datetime(Siheung['측정일시'], format='%Y%m%d%H', errors='coerce')
Incheon['date'] = pd.to_datetime(Incheon['측정일시'], format='%Y%m%d%H', errors='coerce')
Yeosu['date'] = pd.to_datetime(Yeosu['측정일시'], format='%Y%m%d%H', errors='coerce')
Seoul['date'] = pd.to_datetime(Seoul['측정일시'], format='%Y%m%d%H', errors='coerce')
Daebu['date'] = pd.to_datetime(Daebu['측정일시'], format='%Y%m%d%H', errors='coerce')
Ulsan['date'] = pd.to_datetime(Ulsan['측정일시'], format='%Y%m%d%H', errors='coerce')

Siheung_daily = Siheung.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Incheon_daily = Incheon.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Yeosu_daily = Yeosu.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Seoul_daily = Seoul.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Daebu_daily = Daebu.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Ulsan_daily = Ulsan.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily

# For daily plot

plt.figure(figsize=(10,10))
plt.plot(Beijing['date'], Beijing['pm25_conc'], label='Beijing, China')
plt.plot(Shanghai['date'], Shanghai['pm25_conc'], label = 'Shanghai, China')
plt.plot(Shenzhen['date'], Shenzhen['pm25_conc'], label = 'Shenzhen, China')
plt.plot(Hamburg['date'], Hamburg['pm25_conc'], label='Hamburg, Germany')
plt.plot(Siheung_daily.index, Siheung_daily['PM25'], label='Siheung, South Korea')
plt.plot(Incheon_daily.index, Incheon_daily['PM25'], label='Incheon, South Korea')
plt.plot(Yeosu_daily.index, Yeosu_daily['PM25'], label='Yeosu, South Korea')
plt.plot(Seoul_daily.index, Seoul_daily['PM25'], label='Seoul, South Korea')
plt.plot(Daebu_daily.index, Daebu_daily['PM25'], label='Daebu, South Korea')
plt.plot(Ulsan_daily.index, Ulsan_daily['PM25'], label='Ulsan, South Korea')

plt.xlim([datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)])
plt.xticks(rotation=45)
plt.ylim([0,350])
plt.tight_layout()
plt.legend()
plt.show()




Beijing_monthly = Beijing.groupby(pd.Grouper(freq='M', key='date')).mean() # 'M' means monthly
Shanghai_monthly = Shanghai.groupby(pd.Grouper(freq='M', key='date')).mean()
Shenzhen_monthly = Shenzhen.groupby(pd.Grouper(freq='M', key='date')).mean()
Hamburg_monthly = Hamburg.groupby(pd.Grouper(freq='M', key='date')).mean()

Siheung_monthly = Siheung.groupby(pd.Grouper(freq='M', key='date')).mean()
Incheon_monthly = Incheon.groupby(pd.Grouper(freq='M', key='date')).mean()
Yeosu_monthly = Yeosu.groupby(pd.Grouper(freq='M', key='date')).mean()
Seoul_monthly = Seoul.groupby(pd.Grouper(freq='M', key='date')).mean()
Daebu_monthly = Daebu.groupby(pd.Grouper(freq='M', key='date')).mean()
Ulsan_monthly = Ulsan.groupby(pd.Grouper(freq='M', key='date')).mean()

# For monthly plot

plt.figure(figsize=(10,10))
plt.plot(Beijing_monthly.index, Beijing_monthly['pm25_conc'], label='Beijing, China')
plt.plot(Shanghai_monthly.index, Shanghai_monthly['pm25_conc'], label = 'Shanghai, China')
plt.plot(Shenzhen_monthly.index, Shenzhen_monthly['pm25_conc'], label = 'Shenzhen, China')
plt.plot(Hamburg_monthly.index, Hamburg_monthly['pm25_conc'], label='Hamburg, Germany')
plt.plot(Siheung_monthly.index, Siheung_monthly['PM25'], label='Siheung, South Korea')
plt.plot(Incheon_monthly.index, Incheon_monthly['PM25'], label='Incheon, South Korea')
plt.plot(Yeosu_monthly.index, Yeosu_monthly['PM25'], label='Yeosu, South Korea')
plt.plot(Seoul_monthly.index, Seoul_monthly['PM25'], label='Seoul, South Korea')
plt.plot(Daebu_monthly.index, Daebu_monthly['PM25'], label='Daebu, South Korea')
plt.plot(Ulsan_monthly.index, Ulsan_monthly['PM25'], label='Ulsan, South Korea')

plt.xlim([datetime.date(2020, 1, 1), datetime.date(2020, 12, 31)])
plt.xticks(rotation=45)
plt.ylim([0,100])
plt.tight_layout()
plt.legend()
plt.show()


Beijing.to_csv('Beijing_daily.csv')
Shanghai.to_csv('Shanghai_daily.csv')
Shenzhen.to_csv('Shenzhen_daily.csv')
Hamburg.to_csv('Hamburg_daily.csv')
Siheung_daily.to_csv('Siheung_daily.csv', encoding='euc-kr')
Incheon_daily.to_csv('Incheon_daily.csv', encoding='euc-kr')
Yeosu_daily.to_csv('Yeosu_daily.csv', encoding='euc-kr')
Seoul_daily.to_csv('Seoul_daily.csv', encoding='euc-kr')
Daebu_daily.to_csv('Daebu_daily.csv', encoding='euc-kr')
Ulsan_daily.to_csv('Ulsan_daily.csv', encoding='euc-kr')


boxdata = [list(Beijing['pm25_conc'].dropna()),
           list(Shanghai['pm25_conc'].dropna()),
           list(Shenzhen['pm25_conc'].dropna()),
           list(Hamburg['pm25_conc'].dropna()),
           list(Siheung_daily['PM25'].dropna()),
           list(Incheon_daily['PM25'].dropna()),
           list(Yeosu_daily['PM25'].dropna()),
           list(Seoul_daily['PM25'].dropna()),
           list(Daebu_daily['PM25'].dropna()),
           list(Ulsan_daily['PM25'].dropna())]


## For boxplot
plt.figure(figsize=(10,10))
#sns.boxplot(x='Beijing',y='concentration',data=list(Beijing['pm25_conc'].dropna()))
plt.boxplot(boxdata, labels=['Beijing', 'Shanghai','Shenzhen', 'Hamburg', 'Siheung','Incheon','Yeosu','Seoul','Daebu','Ulsan'])
plt.ylim([0,150])
plt.show()


# For raw data and PMF results analysis

meteo1 = pd.read_csv('D:\\OneDrive - SNU\\data\\Meteorological\\Siheung_AWS_hour\\'+
                     'SURFACE_AWS_565_HR_2017_2017_2018.csv', encoding='euc-kr')
meteo2 = pd.read_csv('D:\\OneDrive - SNU\\data\\Meteorological\\Siheung_AWS_hour\\'+
                     'SURFACE_AWS_565_HR_2018_2018_2019.csv', encoding='euc-kr')
meteo3 = pd.read_csv('D:\\OneDrive - SNU\\data\\Meteorological\\Siheung_AWS_hour\\'+
                     'SURFACE_AWS_565_HR_2019_2019_2020.csv', encoding='euc-kr')
meteo4 = pd.read_csv('D:\\OneDrive - SNU\\data\\Meteorological\\Siheung_AWS_hour\\'+
                     'SURFACE_AWS_565_HR_2020_2020_2021.csv', encoding='euc-kr')

meteo = meteo1.append(meteo2).append(meteo2).append(meteo3).append(meteo4)

del meteo1, meteo2, meteo3, meteo4

meteo['wind_x'] = meteo['풍속(m/s)']*np.sin(meteo['풍향(deg)']*math.pi/180)
meteo['wind_y'] = meteo['풍속(m/s)']*np.cos(meteo['풍향(deg)']*math.pi/180)

meteo['date'] = pd.to_datetime(meteo['일시'], format='%Y-%m-%d %H')
meteo_daily = meteo.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily

meteo_daily['ws'] = np.sqrt(meteo_daily['wind_x']**2+meteo_daily['wind_y']**2)
meteo_daily['wd'] = np.arctan2(meteo_daily['wind_y'], meteo_daily['wind_x'])*180/math.pi + 180
meteo_daily['date'] = meteo_daily.index
meteo_daily = meteo_daily.reset_index(drop=True)


df = pd.read_csv('D:\Data_backup\Dropbox\PMF_paper\data_YSLEE\PMF results_raw_YSLEE.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

data = pd.merge(df,meteo_daily, how='left', on='date')
data = data.drop(['지점', '현지기압(hPa)', '해면기압(hPa)', '일사(MJ/m^2)', '일조(hr)', 'wind_x', 'wind_y'], axis=1) # Column drop

data.to_csv('SH_PMF_meteo.csv', index=False, encoding='euc-kr')