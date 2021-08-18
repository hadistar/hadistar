import geopandas
import contextily as ctx
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

import pandas as pd
data = pd.read_csv('AirKora_2019_2020_SH_100km.csv')


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

lon1, lon2, lat1, lat2 = 126.5, 127.1, 37.1, 37.7
extent = [lon1, lon2, lat1, lat2]


plt.figure()
ax = df.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='none')
ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.plot(data['lon'], data['lat'], color='blue', marker='X',
         linestyle='None', markersize=10, label='Monitoring Site')
ctx.add_basemap(ax, zoom=10, crs='epsg:4326')
plt.show()


