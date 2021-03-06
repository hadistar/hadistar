import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

from sklearn import linear_model
import sklearn


dir = 'D:/OneDrive - SNU/data/Sensors/signal process/'

signals = pd.read_csv(dir+'sensors_test1.csv') # for 1st test data
#signals = pd.read_csv(dir+'sensors_2nd_trial_SH.csv') # for 2nd test data

signals = signals.append(pd.read_csv(dir+'sensors_2nd_trial_SH.csv'))

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

signals_hour = signals_hour.dropna(axis=0)

data = pd.merge(signals_hour, station, how='inner', on='date')

data = pd.merge(data, meteo, how='inner', on='date')

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
plt.plot(data['date'], data['no2'], 'ro-', label='NO'+r'$_2$' + ', National station')
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
plt.grid(True, linestyle='--')
plt.savefig('2nd_NO2.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['date'], data['so2'], 'ro-', label='SO'+r'$_2$' + ', National station')
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
plt.grid(True, linestyle='--')
plt.savefig('2nd_SO2.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.plot(data['date'], data['o3'], 'ro-', label='O'+r'$_3$'+', National station')
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
plt.grid(True, linestyle='--')
plt.savefig('2nd_O3.png', bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(data['date'], data['co'], 'ro-', label='CO, National station')
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
plt.grid(True, linestyle='--')
plt.savefig('2nd_CO.png', bbox_inches='tight')
plt.show()

# Temp and humidity - Time series


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(data['date'], data['temp'], 'ko-', label=u'Temperature (\u00B0C) (left)')
ax2.plot(data['date'], data['rh'], 'bo-', label='Relative humidity (%) (right)')
ax1.plot([],[], 'bo-', label='Relative humidity (%) (right)')  # Make an agent in ax

ax1.set_xlabel('time (month-day hour)')
ax1.set_ylabel(u'Temperature (\u00B0C)', color='k')
ax2.set_ylabel('Relative humidity (%)', color='k')

ax1.tick_params(axis='x',labelrotation=45)
ax1.set_ylim(0, 30)
ax2.set_ylim(0, 100)

plt.tight_layout()
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--')
plt.show()



# for 1:1 plot - all combined

x = data['co']
y = data['sensor_CO']

# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.to_numpy().reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure()
plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., CO (ppm)')
plt.ylabel('Sensor conc., CO (ppm)')
plt.text(x.max() * 0.85, y.max() * 0.7, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 1.4, 0, 1.4])
plt.grid(True, linestyle='--')
plt.show()




x = data['no2']
y = data['sensor_NO2']




# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.to_numpy().reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure()
plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., NO' + r'$_2$' + ' (ppm)')
plt.ylabel('Sensor conc., NO' + r'$_2$' + ' (ppm)')
plt.text(x.max() * 0.65, y.max() * 0.5, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 0.08, 0, 0.08])
plt.grid(True, linestyle='--')
plt.show()


x = data['so2']
y = data['sensor_SO2']


# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.to_numpy().reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure()
plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., SO' + r'$_2$' + ' (ppm)')
plt.ylabel('Sensor conc., SO' + r'$_2$' + ' (ppm)')
plt.text(x.max() * 0.65, y.max() * 0.5, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 0.05, 0, 0.05])
plt.grid(True, linestyle='--')
plt.show()


x = data['o3']
y = data['sensor_O3']

# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.to_numpy().reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure()
plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., O' + r'$_3$' + ' (ppm)')
plt.ylabel('Sensor conc., O' + r'$_3$' + ' (ppm)')
plt.text(x.max() * 0.9, y.max() * 0.6, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 0.10, 0, 0.10])
plt.grid(True, linestyle='--')
plt.show()


# box  - temp

plt.figure()
plt.boxplot(data['temp'])
plt.xticks([])
plt.grid(True, linestyle='--')
plt.xlabel(u'Temperature (\u00B0C)')
plt.tight_layout()
plt.text(1.1, data['temp'].mean(),
         u'Mean = {:.2f} \u00B0C \nMedian = {:.2f} \u00B0C'.format(
             data['temp'].mean(), data['temp'].median()))
plt.show()


# box plot - RH
plt.figure()
plt.boxplot(data['rh'])
plt.xticks([])
plt.grid(True, linestyle='--')
plt.xlabel('Relative humidity (%)')
plt.tight_layout()
plt.text(1.1, data['rh'].median(),
         'Mean = {:.2f} % \nMedian = {:.2f} %'.format(
             data['rh'].mean(), data['rh'].median()))
plt.show()




# for 1:1 plot - Temp separation

x_h = data.loc[data['temp'] > data['temp'].median()]['co']
y_h = data.loc[data['temp'] > data['temp'].median()]['sensor_CO']

x_l = data.loc[data['temp'] < data['temp'].median()]['co']
y_l = data.loc[data['temp'] < data['temp'].median()]['sensor_CO']


plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label=u'Temp > {:.1f} \u00B0C'.format(data['temp'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label=u'Temp < {:.1f} \u00B0C'.format(data['temp'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., CO (ppm)')
plt.ylabel('Sensor conc., CO (ppm)')
plt.axis([0, 1.4, 0, 1.4])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()



x_h = data.loc[data['temp'] > data['temp'].median()]['no2']
y_h = data.loc[data['temp'] > data['temp'].median()]['sensor_NO2']

x_l = data.loc[data['temp'] < data['temp'].median()]['no2']
y_l = data.loc[data['temp'] < data['temp'].median()]['sensor_NO2']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label=u'Temp > {:.1f} \u00B0C'.format(data['temp'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label=u'Temp < {:.1f} \u00B0C'.format(data['temp'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., NO' + r'$_2$' + ' (ppm)')
plt.ylabel('Sensor conc., NO' + r'$_2$' + ' (ppm)')
plt.axis([0, 0.1, 0, 0.1])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


x_h = data.loc[data['temp'] > data['temp'].median()]['so2']
y_h = data.loc[data['temp'] > data['temp'].median()]['sensor_SO2']

x_l = data.loc[data['temp'] < data['temp'].median()]['so2']
y_l = data.loc[data['temp'] < data['temp'].median()]['sensor_SO2']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label=u'Temp > {:.1f} \u00B0C'.format(data['temp'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label=u'Temp < {:.1f} \u00B0C'.format(data['temp'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., SO' + r'$_2$' + ' (ppm)')
plt.ylabel('Sensor conc., SO' + r'$_2$' + ' (ppm)')
plt.axis([0, 0.05, 0, 0.05])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


x_h = data.loc[data['temp'] > data['temp'].median()]['o3']
y_h = data.loc[data['temp'] > data['temp'].median()]['sensor_O3']

x_l = data.loc[data['temp'] < data['temp'].median()]['o3']
y_l = data.loc[data['temp'] < data['temp'].median()]['sensor_O3']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label=u'Temp > {:.1f} \u00B0C'.format(data['temp'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label=u'Temp < {:.1f} \u00B0C'.format(data['temp'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., O' + r'$_3$' + ' (ppm)')
plt.ylabel('Sensor conc., O' + r'$_3$' + ' (ppm)')
plt.axis([0, 0.10, 0, 0.10])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


# for 1:1 plot - RH separation

x_h = data.loc[data['rh'] > data['rh'].median()]['co']
y_h = data.loc[data['rh'] > data['rh'].median()]['sensor_CO']

x_l = data.loc[data['rh'] < data['rh'].median()]['co']
y_l = data.loc[data['rh'] < data['rh'].median()]['sensor_CO']


plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label='RH > {:.1f} %'.format(data['rh'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label='RH < {:.1f} %'.format(data['rh'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., CO (ppm)')
plt.ylabel('Sensor conc., CO (ppm)')
plt.axis([0, 1.4, 0, 1.4])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()



x_h = data.loc[data['rh'] > data['rh'].median()]['no2']
y_h = data.loc[data['rh'] > data['rh'].median()]['sensor_NO2']

x_l = data.loc[data['rh'] < data['rh'].median()]['no2']
y_l = data.loc[data['rh'] < data['rh'].median()]['sensor_NO2']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label='RH > {:.1f} %'.format(data['rh'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label='RH < {:.1f} %'.format(data['rh'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., NO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., NO' + r'$_2$' + '(ppm)')
plt.axis([0, 0.08, 0, 0.08])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


x_h = data.loc[data['rh'] > data['rh'].median()]['so2']
y_h = data.loc[data['rh'] > data['rh'].median()]['sensor_SO2']

x_l = data.loc[data['rh'] < data['rh'].median()]['so2']
y_l = data.loc[data['rh'] < data['rh'].median()]['sensor_SO2']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label='RH > {:.1f} %'.format(data['rh'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label='RH < {:.1f} %'.format(data['rh'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., SO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., SO' + r'$_2$' + '(ppm)')
plt.axis([0, 0.05, 0, 0.05])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


x_h = data.loc[data['rh'] > data['rh'].median()]['o3']
y_h = data.loc[data['rh'] > data['rh'].median()]['sensor_O3']

x_l = data.loc[data['rh'] < data['rh'].median()]['o3']
y_l = data.loc[data['rh'] < data['rh'].median()]['sensor_O3']

plt.figure()
plt.scatter(x_h, y_h, s=40, facecolors='none', edgecolors='r',
            label='RH > {:.1f} %'.format(data['rh'].median()))
plt.scatter(x_l, y_l, s=40, facecolors='none', edgecolors='b',
            label='RH < {:.1f} %'.format(data['rh'].median()))
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('National station conc., O' + r'$_3$' + '(ppm)')
plt.ylabel('Sensor conc., O' + r'$_3$' + '(ppm)')
plt.axis([0, 0.10, 0, 0.10])
plt.grid(True, linestyle='--')
plt.legend()
plt.show()


diff = (data['o3']-data['sensor_O3'])**2/data['o3']
data['rh'].loc[diff>0.1]
data['temp'].loc[diff>0.1]


diff = (data['no2']-data['sensor_NO2'])**2/data['no2']
print(data['rh'].loc[diff>0.05])
print(data['temp'].loc[diff>0.05])

diff = (data['co']-data['sensor_CO'])**2/data['co']
print(data['rh'].loc[diff>0.1])
print(data['temp'].loc[diff>0.1])