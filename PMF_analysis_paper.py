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

## - For data preprocessing

### AQI to mass concentration

Beijing = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\beijing-us embassy-air-quality.csv')
Beijing['date'] = pd.to_datetime(Beijing['date'])
Beijing[' pm25'] = pd.to_numeric(Beijing[' pm25'], errors='coerce')


Hamburg = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\Hamburg, germany-air-quality.csv')
Hamburg['date'] = pd.to_datetime(Hamburg['date'])
Hamburg[' pm25'] = pd.to_numeric(Hamburg[' pm25'], errors='coerce')

Shanghai = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\shanghai-air-quality.csv')
Shanghai['date'] = pd.to_datetime(Shanghai['date'])
Shanghai[' pm25'] = pd.to_numeric(Shanghai[' pm25'], errors='coerce')

Shenzhen = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\shenzhen-air-quality.csv')
Shenzhen['date'] = pd.to_datetime(Shenzhen['date'])
Shenzhen[' pm25'] = pd.to_numeric(Shenzhen[' pm25'], errors='coerce')

Kassel1 = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\kassel-fünffensterstr.,-germany-air-quality.csv')
Kassel1['date'] = pd.to_datetime(Kassel1['date'])
Kassel1[' pm25'] = pd.to_numeric(Kassel1[' pm25'], errors='coerce')

Kassel2 = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\kassel-mitte,-germany-air-quality.csv')
Kassel2['date'] = pd.to_datetime(Kassel2['date'])
Kassel2[' pm25'] = pd.to_numeric(Kassel2[' pm25'], errors='coerce')

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
Kassel1['pm25_conc'] = Kassel1.apply(to_con, axis=1)
Kassel1 = Kassel1.sort_values(by='date')

Kassel2['pm25_conc'] = Kassel2.apply(to_con, axis=1)
Kassel2 = Kassel2.sort_values(by='date')




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


#Siheung['date'] = pd.to_datetime(Siheung['측정일시'], format='%Y%m%d%H', errors='coerce')

# improved version
Siheung['temp'] = Siheung['측정일시'] - 1
Siheung['date'] = pd.to_datetime(Siheung['temp'], format='%Y%m%d%H')
Siheung['date'] = Siheung['date'] + pd.DateOffset(hours=1)

#Incheon['date'] = pd.to_datetime(Incheon['측정일시'], format='%Y%m%d%H', errors='coerce')

Incheon['temp'] = Incheon['측정일시'] - 1
Incheon['date'] = pd.to_datetime(Incheon['temp'], format='%Y%m%d%H')
Incheon['date'] = Incheon['date'] + pd.DateOffset(hours=1)

#Yeosu['date'] = pd.to_datetime(Yeosu['측정일시'], format='%Y%m%d%H', errors='coerce')

Yeosu['temp'] = Yeosu['측정일시'] - 1
Yeosu['date'] = pd.to_datetime(Yeosu['temp'], format='%Y%m%d%H')
Yeosu['date'] = Yeosu['date'] + pd.DateOffset(hours=1)

Seoul['temp'] = Seoul['측정일시'] - 1
Seoul['date'] = pd.to_datetime(Seoul['temp'], format='%Y%m%d%H')
Seoul['date'] = Seoul['date'] + pd.DateOffset(hours=1)

#Seoul['date'] = pd.to_datetime(Seoul['측정일시'], format='%Y%m%d%H', errors='coerce')

Daebu['temp'] = Daebu['측정일시'] - 1
Daebu['date'] = pd.to_datetime(Daebu['temp'], format='%Y%m%d%H')
Daebu['date'] = Daebu['date'] + pd.DateOffset(hours=1)

#Daebu['date'] = pd.to_datetime(Daebu['측정일시'], format='%Y%m%d%H', errors='coerce')

Ulsan['temp'] = Ulsan['측정일시'] - 1
Ulsan['date'] = pd.to_datetime(Ulsan['temp'], format='%Y%m%d%H')
Ulsan['date'] = Ulsan['date'] + pd.DateOffset(hours=1)

#Ulsan['date'] = pd.to_datetime(Ulsan['측정일시'], format='%Y%m%d%H', errors='coerce')

Siheung_daily = Siheung.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Incheon_daily = Incheon.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Yeosu_daily = Yeosu.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Seoul_daily = Seoul.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Daebu_daily = Daebu.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
Ulsan_daily = Ulsan.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily

Beijing_monthly = Beijing.groupby(pd.Grouper(freq='M', key='date')).mean() # 'M' means monthly
Shanghai_monthly = Shanghai.groupby(pd.Grouper(freq='M', key='date')).mean()
Shenzhen_monthly = Shenzhen.groupby(pd.Grouper(freq='M', key='date')).mean()
Hamburg_monthly = Hamburg.groupby(pd.Grouper(freq='M', key='date')).mean()
Kassel1_monthly = Kassel1.groupby(pd.Grouper(freq='M', key='date')).mean()
Kassel2_monthly = Kassel2.groupby(pd.Grouper(freq='M', key='date')).mean()

Siheung_monthly = Siheung.groupby(pd.Grouper(freq='M', key='date')).mean()
Incheon_monthly = Incheon.groupby(pd.Grouper(freq='M', key='date')).mean()
Yeosu_monthly = Yeosu.groupby(pd.Grouper(freq='M', key='date')).mean()
Seoul_monthly = Seoul.groupby(pd.Grouper(freq='M', key='date')).mean()
Daebu_monthly = Daebu.groupby(pd.Grouper(freq='M', key='date')).mean()
Ulsan_monthly = Ulsan.groupby(pd.Grouper(freq='M', key='date')).mean()

Beijing.to_csv('1_Beijing_daily.csv')
Shanghai.to_csv('2_Shanghai_daily.csv')
Shenzhen.to_csv('3_Shenzhen_daily.csv')
Hamburg.to_csv('4_Hamburg_daily.csv')
Kassel1.to_csv('11_Kassel1_daily.csv')
Kassel2.to_csv('12_Kassel2_daily.csv')


Siheung_daily.to_csv('5_Siheung_daily.csv', encoding='euc-kr')
Incheon_daily.to_csv('6_Incheon_daily.csv', encoding='euc-kr')
Yeosu_daily.to_csv('7_Yeosu_daily.csv', encoding='euc-kr')
Seoul_daily.to_csv('8_Seoul_daily.csv', encoding='euc-kr')
Daebu_daily.to_csv('9_Daebu_daily.csv', encoding='euc-kr')
Ulsan_daily.to_csv('10_Ulsan_daily.csv', encoding='euc-kr')

## End of data preprocessing


## Data loading and plotting

Beijing_daily = pd.read_csv('1_Beijing_daily.csv')
Shanghai_daily = pd.read_csv('2_Shanghai_daily.csv')
Shenzhen_daily = pd.read_csv('3_Shenzhen_daily.csv')
Hamburg_daily = pd.read_csv('4_Hamburg_daily.csv')
Kassel1_daily = pd.read_csv('11_Kassel1_daily.csv')
Kassel2_daily = pd.read_csv('12_Kassel2_daily.csv')

Siheung_daily = pd.read_csv('5_Siheung_daily.csv', encoding='euc-kr')
Incheon_daily = pd.read_csv('6_Incheon_daily.csv', encoding='euc-kr')
Yeosu_daily = pd.read_csv('7_Yeosu_daily.csv', encoding='euc-kr')
Seoul_daily = pd.read_csv('8_Seoul_daily.csv', encoding='euc-kr')
Daebu_daily = pd.read_csv('9_Daebu_daily.csv', encoding='euc-kr')
Ulsan_daily = pd.read_csv('10_Ulsan_daily.csv', encoding='euc-kr')


# Beijing_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\1_Beijing_daily.csv')
# Shanghai_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\3_Shenzhen_daily.csv')
# Hamburg_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\4_Hamburg_daily.csv')
# Kassel1_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\11_Kassel1_daily.csv')
# Kassel2_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\12_Kassel2_daily.csv')
#
# Siheung_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\5_Siheung_daily.csv', encoding='euc-kr')
# Incheon_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\6_Incheon_daily.csv', encoding='euc-kr')
# Yeosu_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\7_Yeosu_daily.csv', encoding='euc-kr')
# Seoul_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\8_Seoul_daily.csv', encoding='euc-kr')
# Daebu_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\9_Daebu_daily.csv', encoding='euc-kr')
# Ulsan_daily = pd.read_csv('D:\\Dropbox\\PMF_paper\\data_YSLEE\\10_Ulsan_daily.csv', encoding='euc-kr')


Beijing_daily['date'] = pd.to_datetime(Beijing_daily['date'])
Shanghai_daily['date'] = pd.to_datetime(Shanghai_daily['date'])
Shenzhen_daily['date'] = pd.to_datetime(Shenzhen_daily['date'])
Hamburg_daily['date'] = pd.to_datetime(Hamburg_daily['date'])
Kassel1_daily['date'] = pd.to_datetime(Kassel1_daily['date'])
Kassel2_daily['date'] = pd.to_datetime(Kassel2_daily['date'])

Siheung_daily['date'] = pd.to_datetime(Siheung_daily['date'])
Incheon_daily['date'] = pd.to_datetime(Incheon_daily['date'])
Yeosu_daily['date'] = pd.to_datetime(Yeosu_daily['date'])
Seoul_daily['date'] = pd.to_datetime(Seoul_daily['date'])
Daebu_daily['date'] = pd.to_datetime(Daebu_daily['date'])
Ulsan_daily['date'] = pd.to_datetime(Ulsan_daily['date'])

#### Data splitting from start to end

start_date = '2020-04-01'
end_date = '2020-10-31'

df1_p2 = Beijing_daily.loc[(Beijing_daily['date'] > start_date) & (Beijing_daily['date'] < end_date)]
df2_p2 = Shanghai_daily.loc[(Shanghai_daily['date'] > start_date) & (Shanghai_daily['date'] < end_date)]
df3_p2 = Shenzhen_daily.loc[(Shenzhen_daily['date'] > start_date) & (Shenzhen_daily['date'] < end_date)]
df4_p2 = Hamburg_daily.loc[(Hamburg_daily['date'] > start_date) & (Hamburg_daily['date'] < end_date)]
df5_p2 = Siheung_daily.loc[(Siheung_daily['date'] > start_date) & (Siheung_daily['date'] < end_date)]
df6_p2 = Incheon_daily.loc[(Incheon_daily['date'] > start_date) & (Incheon_daily['date'] < end_date)]
df7_p2 = Yeosu_daily.loc[(Yeosu_daily['date'] > start_date) & (Yeosu_daily['date'] < end_date)]
df8_p2 = Seoul_daily.loc[(Seoul_daily['date'] > start_date) & (Seoul_daily['date'] < end_date)]
df9_p2 = Daebu_daily.loc[(Daebu_daily['date'] > start_date) & (Daebu_daily['date'] < end_date)]
df10_p2 = Ulsan_daily.loc[(Ulsan_daily['date'] > start_date) & (Ulsan_daily['date'] < end_date)]
df11_p2 = Kassel1_daily.loc[(Kassel1_daily['date']  > start_date) & (Kassel1_daily['date'] < end_date)]
df12_p2 = Kassel2_daily.loc[(Kassel2_daily['date']  > start_date) & (Kassel2_daily['date'] < end_date)]


boxdata = [list(df1_p2['pm25_conc'].dropna()),
           list(df2_p2['pm25_conc'].dropna()),
           list(df3_p2['pm25_conc'].dropna()),
           list(df4_p2['pm25_conc'].dropna()),
           list(df5_p2['PM25'].dropna()),
           list(df6_p2['PM25'].dropna()),
           list(df7_p2['PM25'].dropna()),
           list(df8_p2['PM25'].dropna()),
           list(df9_p2['PM25'].dropna()),
           list(df10_p2['PM25'].dropna()),
           list(df11_p2['pm25_conc'].dropna())]

### Reordring

names = ['Beijing', 'Shanghai','Shenzhen', 'Hamburg', 'Siheung',
         'Incheon','Yeosu','Seoul','Daebu','Ulsan', 'Kassel']
order = [0,1,3,10,4,9,6,5,7,8]
boxdata = [boxdata[i] for i in order]

## For boxplot
plt.figure(figsize=(8,8))
#sns.boxplot(x='Beijing',y='concentration',data=list(Beijing['pm25_conc'].dropna()))
plt.boxplot(boxdata, labels=[names[i] for i in order])
plt.plot([0,11],[25,25], 'b--', label='WHO guideline')
plt.xlim([0.5,10.5])
plt.ylim([0,100])
plt.xticks(rotation=45)
plt.yticks(np.arange(0,110,10))
plt.title(str(start_date) + " to " + str(end_date))
plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.grid('--', linewidth=0.5)
plt.tight_layout()
plt.legend()
plt.show()


## For time-series plot

plt.figure(figsize=(10,10))
plt.plot(Beijing_daily['date'], Beijing_daily['pm25_conc'], label='Beijing, China')
plt.plot(Shanghai_daily['date'], Shanghai_daily['pm25_conc'], label = 'Shanghai, China')
plt.plot(Shenzhen_daily['date'], Shenzhen_daily['pm25_conc'], label = 'Shenzhen, China')
plt.plot(Hamburg_daily['date'], Hamburg_daily['pm25_conc'], label='Hamburg, Germany')
plt.plot(Siheung_daily['date'], Siheung_daily['PM25'], label='Siheung, South Korea')
plt.plot(Incheon_daily['date'], Incheon_daily['PM25'], label='Incheon, South Korea')
plt.plot(Yeosu_daily['date'], Yeosu_daily['PM25'], label='Yeosu, South Korea')
plt.plot(Seoul_daily['date'], Seoul_daily['PM25'], label='Seoul, South Korea')
plt.plot(Daebu_daily['date'], Daebu_daily['PM25'], label='Daebu, South Korea')
plt.plot(Ulsan_daily['date'], Ulsan_daily['PM25'], label='Ulsan, South Korea')

plt.xlim([datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)])
plt.xticks(rotation=45)
plt.ylim([0,350])
plt.tight_layout()
plt.legend()
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


# For hourly data generation
df = pd.read_csv('D:\Dropbox\PMF_paper\SH_PMF_meteo_daily_v8.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

meteo = meteo3.append(meteo4)
meteo = meteo.rename(columns = {"일시":'date'})
meteo['date'] = pd.to_datetime(meteo['date'])

data = pd.DataFrame()

for n in range(len(df)):
    print(n)
    for i in range(24):
        temp = df.iloc[n].copy()
        temp['date'] = temp['date'] + pd.to_timedelta(i, unit='h')
        data = data.append(temp, sort=False)

data = data[df.columns] # Ordering
data['date'] = pd.to_datetime(data['date'])

data = pd.merge(data, meteo, how='left', on='date')

data.to_csv('SH_PMF_meteo_hourly.csv', index=False, encoding='euc-kr')



# 2021-06-10, recal

df = pd.read_csv('D:\Dropbox\PMF_paper\PMF results_raw\PMF results_YSLEE.csv')
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

data = pd.merge(data, Siheung, how='left', on='date')

data.to_csv('SH_PMF_meteo_hourly.csv', index=False, encoding='euc-kr')


Siheung_daily = Siheung.groupby(pd.Grouper(freq='D', key='date')).mean()

