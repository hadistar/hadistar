import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

dir = 'D:/OneDrive - SNU/data/Sensors/signal process/'

#signals = pd.read_csv(dir+'sensors_test1.csv') # for 1st test data
signals = pd.read_csv(dir+'sensors_2nd_trial_SH.csv') # for 2nd test data

station = pd.read_csv(dir+'station.csv')

signals.rename(columns = {'time':'date'}, inplace = True)

signals['date'] = pd.to_datetime(signals['date'], format="%y-%m-%d %H:%M:%S", exact=False)
station['date'] = pd.to_datetime(station['date'], format="%y-%m-%d:%H", exact=False)

signals_hour = signals.groupby(pd.Grouper(freq='H', key='date')).mean()

meteo1 = pd.read_csv(dir+'SYNOP_AWOS_1562_MI_2021-03_2021-03_2021.csv', encoding='euc-kr')
meteo2 = pd.read_csv(dir+'SYNOP_AWOS_1562_MI_2021-04_2021-04_2021.csv', encoding='euc-kr')

meteo1.columns = ['station','date','temp','rain','rain2','wd','ws','rh','pressure','pressure_sea','sun','sun2']
meteo2.columns = ['station','date','temp','rain','rain2','wd','ws','rh','pressure','pressure_sea','sun','sun2']

meteo = meteo1.append(meteo2)

meteo = meteo[['date','temp','rh','wd','ws']]

meteo['date'] = pd.to_datetime(meteo['date'], format="%y-%m-%d %H:%M", exact=False)
meteo = meteo.groupby(pd.Grouper(freq='H', key='date')).mean()


data = pd.merge(signals_hour, station, how='inner', on='date')

#data = pd.merge(data, meteo, how='inner', on='date')

del meteo, meteo1, meteo2, signals, station, signals_hour

data = data.sort_values(by=['date'])

# 1. Factory calculation
'''
data['sensor_NO2'] = ((data['NO2_vol1'] - 225) - (data['NO2_vol2'] - 241))/309
data['sensor_SO2'] = ((data['SO2_vol1'] - 355) - (data['SO2_vol2'] - 360))/280
data['sensor_CO'] = ((data['CO_vol1'] - 270) - (data['CO_vol2'] - 330))/420
data['sensor_O3'] = ((data['O3+NO2_vol1'] - 260) - (data['O3+NO2_vol2'] - 300))/298 - data['sensor_NO2']
'''

# for 2nd test
data['sensor_NO2'] = ((data['NO2_vol1'] - 225) - (data['NO2_vol2'] - 241))/309
data['sensor_SO2'] = ((data['SO2_vol1'] - 355) - (data['SO2_vol2'] - 360))/280
data['sensor_CO'] = ((data['CO_vol1'] - 270) - (data['CO_vol2'] - 330))/420
data['sensor_O3'] = ((data['O3+NO2_vol1'] - 260) - (data['O3+NO2_vol2'] - 300))/298 - data['sensor_NO2'] -0.12

# Standards
# SO2 : 0.15 ppm (hourly)
# NO2 : 0.10 ppm (hourly)
# CO : 25 ppm (hourly)
# O3 : 0.1 ppm (hourly)

plt.figure()
plt.plot(data['date'], data['no2'], 'ro-', label='NO'+r'$_2$' + ', station')
plt.plot(data['date'], data['sensor_NO2'], 'bo-', label='NO'+r'$_2$' + ', sensor')
plt.axhline(y=0.10, color='k', linestyle='--', label='National standards of South Korea (hourly)')
#plt.plot(data['date'], [0.10]*len(data['date']) , 'k--', label='National standards of South Korea (hourly)')
plt.legend(loc='upper right')
plt.xlabel('time (month-day hour)')
plt.ylabel('concentration (ppm)')
plt.xticks(rotation=45)
plt.ylim(0,0.15)
ax = plt.gca()
plt.tight_layout()
plt.savefig('2nd_NO2.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['date'], data['so2'], 'ro-', label='SO'+r'$_2$' + ', station')
plt.plot(data['date'], data['sensor_SO2'], 'bo-', label='SO'+r'$_2$' + ', sensor')
plt.axhline(y=0.15, color='k', linestyle='--', label='National standards of South Korea (hourly)')
#plt.plot(data['date'], [0.10]*len(data['date']) , 'k--', label='National standards of South Korea (hourly)')
plt.legend(loc='upper right')
plt.xlabel('time (month-day hour)')
plt.ylabel('concentration (ppm)')
plt.xticks(rotation=45)
plt.ylim(0,0.25)
ax = plt.gca()
plt.tight_layout()
plt.savefig('2nd_SO2.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['date'], data['o3'], 'ro-', label='O'+r'$_3$'+', station')
plt.plot(data['date'], data['sensor_O3'], 'bo-', label='O'+r'$_3$'+', sensor')
plt.axhline(y=0.1, color='k', linestyle='--', label='National standards of South Korea (hourly)')
#plt.plot(data['date'], [0.10]*len(data['date']) , 'k--', label='National standards of South Korea (hourly)')
plt.legend(loc='upper right')
plt.xlabel('time (month-day hour)')
plt.ylabel('concentration (ppm)')
plt.xticks(rotation=45)
plt.ylim(0,0.15)
ax = plt.gca()
plt.tight_layout()
plt.savefig('2nd_O3.png', bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(data['date'], data['co'], 'ro-', label='CO, station')
plt.plot(data['date'], data['sensor_CO'], 'bo-', label='CO, sensor')
plt.plot([],[],' ', label='National standards (hourly): 25 ppm')
#plt.axhline(y=25, color='k', linestyle='--', label='National standards of South Korea (hourly)')
#plt.plot(data['date'], [0.10]*len(data['date']) , 'k--', label='National standards of South Korea (hourly)')
plt.legend(loc='upper right')
plt.xlabel('time (month-day hour)')
plt.ylabel('concentration (ppm)')
plt.xticks(rotation=45)
plt.ylim(0,2)
ax = plt.gca()
plt.tight_layout()
plt.savefig('2nd_CO.png', bbox_inches='tight')
plt.show()




plt.figure()
plt.scatter(data['co'], data['sensor_CO'])
plt.show()

plt.figure()
plt.scatter(data['no2'], data['sensor_NO2'])
plt.show()

plt.figure()
plt.scatter(data['so2'], data['sensor_SO2'])
plt.show()

plt.figure()
plt.scatter(data['o3'], data['sensor_O3'])
plt.show()
