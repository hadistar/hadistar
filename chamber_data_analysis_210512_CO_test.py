import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

dir = 'D:/OneDrive - SNU/data/Chamber_CO_test_210511/'

chamber = 'Chamber/'
Arduino = 'Arduino/'
Sniffer = 'Sniffer/'

flow_dir = os.path.join(dir+chamber,'Flow/')
rh_t_dir = os.path.join(dir+chamber,'Temp_Humidity/')

rh_t_files = os.listdir(rh_t_dir)
flow_files = os.listdir(flow_dir)

data_RH_T = pd.DataFrame()
for i in rh_t_files:
    data_RH_T=data_RH_T.append(pd.read_csv(rh_t_dir+i))


data_RH_T['Time'] = pd.to_datetime(data_RH_T['Time'])
data_RH_T = data_RH_T.groupby(pd.Grouper(freq='1min', key='Time')).mean().dropna(axis=0)
data_RH_T = data_RH_T.reset_index()
#data_RH_T['Time'] = data_RH_T.index

data_sensor = pd.read_csv(dir+Arduino+'2021-05-10.csv').dropna(axis=0)
data_sensor['Time'] = pd.to_datetime(data_sensor.Date, format='%Y-%m-%d %H:%M:%S', errors='raise')
data_sensor = data_sensor.groupby(pd.Grouper(freq='1min', key='Time')).mean().dropna(axis=0)
data_sensor = data_sensor.reset_index()
data_sensor = data_sensor[data_sensor.columns[[0,2,9,16,22]]]

data_reference = pd.read_table(dir+'CO_reference_210511.txt', sep=' ', header=None).dropna(axis=1)
data_reference['CO_ref'] = data_reference[8].astype(str).str[6:]
data_reference[2] = '2021-05-10 '+data_reference[2].astype(str).str[4:]
data_reference['Time'] = pd.to_datetime(data_reference[2], format='%Y-%m-%d %H:%M')
data_reference = data_reference.drop(data_reference[data_reference['CO_ref']=='XXXX'].index)
data_reference.CO_ref = data_reference.CO_ref.astype(float)

# time splitting
#
data_RH_T = data_RH_T.loc[data_RH_T["Time"].
    between('2021-05-10 14:30', '2021-05-10 23:59')]
data_sensor = data_sensor.loc[data_sensor["Time"].
    between('2021-05-10 14:30', '2021-05-10 23:59')]
data_reference = data_reference.loc[data_reference["Time"].
    between('2021-05-10 14:30', '2021-05-10 23:59')]
#
# data_RH_T = data_RH_T.loc[data_RH_T["Time"].
#     between('2021-05-10 21:30', '2021-05-10 23:59')]
# data_sensor = data_sensor.loc[data_sensor["Time"].
#     between('2021-05-10 21:30', '2021-05-10 23:59')]
# data_reference = data_reference.loc[data_reference["Time"].
#     between('2021-05-10 21:30', '2021-05-10 23:59')]



#plt.figure()
#plt.plot(data_RH_T['Time'],data_RH_T['HumidityPV'],'ro', markersize=3, label='Humidity PV')
#plt.plot(data_RH_T['Time'],data_RH_T[' HumiditySV'],'bo', markersize=3, label='Humidity SV')
#plt.plot(data_RH_T['Time'],data_RH_T[' Temperature'],'ko', markersize=3, label='Temperature PV')
#plt.gcf().autofmt_xdate()
#plt.xlabel('Time (month-day hour)')
#plt.ylabel('Temp or RH')
#plt.legend()
#plt.show()


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(data_RH_T['Time'],data_RH_T[' Temperature'], 'ro-', label=u'Temperature (\u00B0C) (left)')
ax2.plot(data_RH_T['Time'], data_RH_T['HumidityPV'], 'bo-', label='Relative humidity (%) (right)')
ax1.plot([],[], 'bo-', label='Relative humidity (%) (right)')  # Make an agent in ax
ax1.set_xlabel('time (month-day hour)')
ax1.set_ylabel(u'Temperature (\u00B0C)', color='k')
ax2.set_ylabel('Relative humidity (%)', color='k')

ax1.tick_params(axis='x',labelrotation=45)
plt.locator_params(axis='x', nbins=6)

ax1.set_ylim(0, 40)
ax2.set_ylim(0, 100)

plt.tight_layout()
ax1.legend(loc='lower right')
ax1.grid(True, linestyle='--')
plt.show()


plt.figure()
plt.plot(data_sensor['Time'], data_sensor[' NO2 CONC'], 'ro', label='Alphasense sensor')
plt.plot(data_reference['Time'], data_reference['CO_ref'], 'bo', label='reference sensor')
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.locator_params(axis='x', nbins=6)
plt.xlabel('Time (month-day hour)')
plt.ylabel('Concentration (ppm)')
plt.ylim([0.0,12.0])
#plt.xlim([0,84])
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

# 3D plot
for3d = pd.merge(data_RH_T,data_sensor, on='Time', how='inner')
for3d = pd.merge(for3d, data_reference, on='Time',how='inner')
x = for3d.HumidityPV
y = for3d[' Temperature']
z1 = for3d[' NO2 CONC']
z2 = for3d['CO_ref']

fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(111, projection='3d') # Axe3D object
#X,Y = np.meshgrid(x,y)
ax.scatter(x, y, z1, cmap='plasma', label = 'Alphasense sensor')
ax.scatter(x, y, z2, alpha=0.6, label= 'Reference sensor')
ax.set_xlabel('Relative humidity (%)', color='k')
ax.set_ylabel(u'Temperature (\u00B0C)', color='k')
ax.set_zlabel('Concentration (ppm)')
plt.legend()
plt.show()


# -----------------------
# Interpolation on a grid
# -----------------------
# A contour plot of irregularly spaced data coordinates
# via interpolation on a grid.

# Create grid values first.
xi = np.linspace(35, 80, 450)
yi = np.linspace(5, 35, 300)

# Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, z)
Xi, Yi = np.meshgrid(xi, yi)
zi = interpolator(Xi, Yi)

plt.show()
