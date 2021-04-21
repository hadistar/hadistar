import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

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
plt.xlabel('Station conc., CO (ppm)')
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
plt.xlabel('Station conc., NO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., NO' + r'$_2$' + '(ppm)')
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
plt.xlabel('Station conc., SO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., SO' + r'$_2$' + '(ppm)')
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
plt.xlabel('Station conc., O' + r'$_3$' + '(ppm)')
plt.ylabel('Sensor conc., O' + r'$_3$' + '(ppm)')
plt.text(x.max() * 0.75, y.max() * 0.5, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 0.10, 0, 0.10])
plt.grid(True, linestyle='--')
plt.show()


# box plot
plt.figure()
plt.boxplot(data['temp'])
plt.xticks([])
plt.grid(True, linestyle='--')
plt.xlabel('Temperature (Degree)')
plt.tight_layout()
plt.text(1.1, data['temp'].mean(),
         'Mean = {:.2f} \nMedian = {:.2f}'.format(
             data['temp'].mean(), data['temp'].median()))
plt.show()


# box plot
plt.figure()
plt.boxplot(data['rh'])
plt.xticks([])
plt.grid(True, linestyle='--')
plt.xlabel('Relative humidity (%)')
plt.tight_layout()
plt.text(1.1, data['rh'].median(),
         'Mean = {:.2f} \nMedian = {:.2f}'.format(
             data['rh'].mean(), data['rh'].median()))
plt.show()




# for 1:1 plot - Temp separation

x = data['co']
y = data['sensor_CO']


plt.figure()
plt.scatter(x, y, s=80, facecolors='none', edgecolors='r')
plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,1.4],[0,1.4], 'k--')
plt.xlabel('Station conc., CO (ppm)')
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
plt.xlabel('Station conc., NO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., NO' + r'$_2$' + '(ppm)')
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
plt.xlabel('Station conc., SO' + r'$_2$' + '(ppm)')
plt.ylabel('Sensor conc., SO' + r'$_2$' + '(ppm)')
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
plt.xlabel('Station conc., O' + r'$_3$' + '(ppm)')
plt.ylabel('Sensor conc., O' + r'$_3$' + '(ppm)')
plt.text(x.max() * 0.75, y.max() * 0.5, '$R^2$ = %0.2f (n=%d)' % (r_squared, len(x)))
plt.axis([0, 0.10, 0, 0.10])
plt.grid(True, linestyle='--')
plt.show()

