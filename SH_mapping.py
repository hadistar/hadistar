import geopandas
import contextily as ctx
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

import pandas as pd
data = pd.read_csv('AirKora_2019_2020_SH_50km.csv')


# SH

df = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df = df.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]


plt.figure()
ax = df.plot(figsize=(10, 10), alpha=1.0, edgecolor='r', facecolor='none')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
plt.plot(data['lon'], data['lat'], color='blue', marker='X',
         linestyle='None', markersize=10, label='Monitoring Site')
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.show()





# Sudokwon

df = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df = df.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.5, 127.3, 37.0, 37.7
extent = [lon1, lon2, lat1, lat2]

plt.figure()
ax = df.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='none')
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.plot(data['lon'], data['lat'], color='blue', marker='X',
         linestyle='None', markersize=10, label='Monitoring Site')
ctx.add_basemap(ax, zoom=10, crs='epsg:4326')
plt.show()



# SH

data = pd.read_csv('loc_SH_100mx100m.csv')

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]


plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
plt.plot(data['lon'], data['lat'], color='blue', marker='X',
         linestyle='None', markersize=0.15, label='Monitoring Site')
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.show()







# 2021-08-25 스마트시티 베이지안 매핑용 연습

import folium
import pandas as pd
import branca.colormap as cm
from selenium import webdriver
import time
import math
import matplotlib.pyplot as plt
import cartopy
import numpy as np
import datetime
from scalebar import scale_bar


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20


df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]


data2 = pd.read_csv('results__.csv')


data3 = data2[data2['Var9']=='1/01/2020']

plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
# plt.plot(data['lon'], data['lat'], color='blue', marker='X',
#          linestyle='None', markersize=0.15, label='Monitoring Site')

points = plt.scatter(data3['Var12'], data3['Var11'], c=data3['temp6'],
                     vmin=0, vmax=30,
#                     vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max()) % 5,
                     cmap='jet', alpha=0.8, s=0.2)
cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
cb.ax.tick_params(labelsize=20)
plt.title('[ 2020-01-01 ]')

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.show()













plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
plt.plot(data['lon'], data['lat'], color='blue', marker='X',
         linestyle='None', markersize=0.1, label='Monitoring Site')


points = plt.scatter(s_new['lon'], s_new['lat'], c=df_z,
                     vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max()) % 5,
                     cmap='Reds', alpha=0.8, s=3.0)
cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()

plt.legend(title='[ 2020-01-01 ]', loc='upper right')

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.show()



