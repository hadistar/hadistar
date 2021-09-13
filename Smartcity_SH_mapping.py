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
# 2021-09-13 스마트시티 베이지안 매핑

import geopandas
import contextily as ctx
import pandas as pd
import math
import matplotlib.pyplot as plt


plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20


df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]


data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_BSMRM_210913.csv')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\매핑결과_샘플.csv', encoding='euc-kr')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\results.csv', encoding='euc-kr')
#data2.columns = ['해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차','date','Location number','lat','lon']

data2.columns = ['Salts', 'Soil', 'SS', 'Coal', 'Industry', 'Combustions', 'SN', 'Traffic','date','Location number','lat','lon']

data2.date = pd.to_datetime(data2.date)

for i in range(len(data2.drop_duplicates('date').date)):
    day = data2.drop_duplicates('date').date[i]
    day = str(day)[:10]
    for source in data2.columns[:8]:
        data3 = data2[data2['date'] == day]

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
        #ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
        # plt.plot(data['lon'], data['lat'], color='blue', marker='X',
        #          linestyle='None', markersize=0.15, label='Monitoring Site')

        points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
        #                     vmin=0, vmax=30,
        #                     vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max()) % 5,
                             cmap='jet', alpha=0.8, s=0.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('['+day+', '+source+']')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.savefig('mappingresults_210913/'+day+', '+source+'.png')
        plt.close()













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


import haversine as hs
import geopandas
import contextily as ctx
import pandas as pd
import math
import matplotlib.pyplot as plt


plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20


df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.5, 127.0, 37.2, 37.70
extent = [lon1, lon2, lat1, lat2]

# version 1
# w = [[37.30035, 126.69301],
#     [37.30035, 126.83489],
#     [37.30035, 126.76390],
#     [37.37135, 126.69301],
#     [37.37135, 126.83489],
#     [37.37135, 126.76390],
#     [37.44235, 126.69301],
#     [37.44235, 126.83489],
#     [37.44235, 126.76390]]


# version 2
w = [[37.30717, 126.73441], # 1
    [37.428585, 126.817205], # 2
    [37.30588, 126.90802], # 3
    [37.53942, 126.71340], # 4
    [37.45481, 126.64639], # 5
    [37.37135, 126.63958], # 6
    [37.4315, 126.9911], # 7
    [37.55, 126.9]] # 9

w = pd.DataFrame(w)

hs.haversine(w.iloc[1], w.iloc[6], unit='km')
# 15.36

N = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\pm25speciation_locations_KoreaNational.csv')
N = N.drop(7)

plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
# plt.plot(w[1], w[0], color='blue', marker='X',
#          linestyle='None', markersize=10, label='w9')

for i in range(len(w)):
    x = w[1][i] # lat
    y = w[0][i] # lon

    plt.plot(x, y, color='blue', marker='X',
             linestyle='None', markersize=8.15)
    plt.text(x*1.00000, y*0.9998, i+1, color='red',fontsize=14)

plt.plot(N['lon'], N['lat'], color='red', marker='o',
         linestyle='None', markersize=10, label='Monitoring site')
plt.plot(126.739900, 37.347200, color='red', marker='o',
         linestyle='None', markersize=10)

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.legend()
plt.show()


# Data generation

df = data.sort_values(by=['distance'], axis=0, ascending=True)

df2 = df.loc[df.distance<17].copy() # 17 km

Stations = df2['Station code'].drop_duplicates()

Stations = Stations.reset_index(drop=True)
Stations = Stations.reset_index()
Stations.columns = ['No.', 'Station code']
Stations['No.']+=1


df2['date'] = pd.to_datetime(df2['date'])
df2 = df2.sort_values(by=['date', 'distance'])
dates = pd.DataFrame(df2['date'].drop_duplicates())

APs = ['SO2','CO','O3','NO2','PM10','PM25']

for ap in APs:
    df3 = dates.copy()

    for s in Stations.iterrows():
        number = s[1]['No.']
        code = s[1]['Station code']

        temp = df2.loc[df2['Station code']==code][['date',ap]]
        temp = temp.rename(columns = {ap : number})

        df3 = pd.merge(df3,temp, how='inner', on='date')

    df3.to_csv('SH_AP_'+ap+'_yslee_210903.csv', index=False)


Stations = pd.merge(Stations, df2.drop_duplicates('Station code'), how='inner', on='Station code')
Stations = Stations[['No.','Station code','lat','lon','distance']]

# SH

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.60, 126.95, 37.2, 37.51
extent = [lon1, lon2, lat1, lat2]


plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
plt.plot(126.7399, 37.3472, color='red', marker='o', markersize=10)

for i in range(len(Stations)):
    x = Stations['lon'][i]
    y = Stations['lat'][i]

    plt.plot(x, y, color='blue', marker='X',
             linestyle='None', markersize=5.15)
    plt.text(x*1.00000, y*0.9998, i+1, color='red',fontsize=12)

ax.set_xlim(lon1, lon2)
ax.set_ylim(lat1, lat2)
plt.savefig('SH_APs_locations_yslee_210903.png')
plt.show()


Stations.to_csv('SH_APs_locations.csv', index=False)