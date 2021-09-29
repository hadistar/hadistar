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
# 2021-09-13 스마트시티 베이지안 매핑 - BSMRM

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


#data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_BSMRM_210913.csv')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\results_BSMRM_210913.csv')
data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_BSMRM_210917_sigma025.csv')

#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\매핑결과_샘플.csv', encoding='euc-kr')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\results.csv', encoding='euc-kr')
#data2.columns = ['해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차','date','Location number','lat','lon']

data2.columns = ['Salts', 'Soil', 'SS', 'Coal', 'Industry', 'Combustions', 'SN', 'Traffic','date','Location number','lat','lon']

data2.date = pd.to_datetime(data2.date)

for i in range(len(data2.drop_duplicates('date').date)):

    if i % 35 ==0:

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
            plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\mappingresults_210917\\'+day+', '+source+'.png')
            plt.close()





# 2021-08-25 스마트시티 베이지안 매핑용 연습
# 2021-09-13 스마트시티 베이지안 매핑 - BSMRM - Monotone version

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


#data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_BSMRM_210913.csv')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\results_BSMRM_210913.csv')
data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_BSMRM_210923_w24_sigma05.csv')

#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\매핑결과_샘플.csv', encoding='euc-kr')
#data2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\results.csv', encoding='euc-kr')
#data2.columns = ['해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차','date','Location number','lat','lon']

data2.columns = ['Salts', 'Soil', 'SS', 'Coal', 'Industry', 'Combustions', 'SN', 'Traffic','date','Location number','lat','lon']
colors = {'Salts':'Blues', 'Soil':'pink_r', 'SS':'Oranges', 'Coal':'Purples',
          'Industry':'bone_r', 'Combustions':'Reds', 'SN':'Greens', 'Traffic':'Wistia'}
data2.date = pd.to_datetime(data2.date)

for i in range(len(data2.drop_duplicates('date').date)):

    if i % 20 ==0:

        day = data2.drop_duplicates('date').date[i]
        day = str(day)[:10]
        for source in data2.columns[:8]:
            data3 = data2[data2['date'] == day]

            plt.figure()
            ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')
            #ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
            # plt.plot(data['lon'], data['lat'], color='blue', marker='X',
            #          linestyle='None', markersize=0.15, label='Monitoring Site')

            points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
            #                     vmin=0, vmax=30,
            #                     vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max()) % 5,
                                 cmap=colors[source], alpha=0.85, s=0.2)
            cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

            cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
            cb.ax.tick_params(labelsize=20)
            plt.title('['+day+', '+source+']')

            ax.set_xlim(lon1, lon2)
            ax.set_ylim(lat1, lat2)
            plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\mappingresults_210923_sigmaK05\\'+day+', '+source+'.png')
            plt.close()





# 2021-09-13 스마트시티 베이지안 매핑 - BNFA

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

lon1, lon2, lat1, lat2 = 126.60, 126.95, 37.2, 37.51
extent = [lon1, lon2, lat1, lat2]

data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\BNFA_24sites_PM25_210914_yslee.csv', header=None).T
q = data2.shape[1]
locations = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_APs_locations_yslee_210903.csv')

data2['No.'] = [x for x in range(1,25)]
data2 = pd.merge(data2, locations, on='No.', how='inner')

for source in range(q):

    plt.figure()
    ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
    points = plt.scatter(data2['lon'], data2['lat'], c=data2[source],
                         cmap='jet', alpha=0.8, s=100.2)
    cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
    cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
    cb.ax.tick_params(labelsize=20)
    for i in range(data2.shape[0]):
        x = data2['lon'][i]
        y = data2['lat'][i]
        plt.text(x * 1.00000, y * 0.9998, data2['No.'][i], color='red', fontsize=12)

    plt.title('[PM10, source '+str(source+1)+']')

    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)
    plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\BNFA_PM10_source ' + str(source+1) +'_yslee_210914.png')
    plt.show()
    plt.close()

# End of BNFA mapping


# 2021-09-17 BNFA results Interpolation


import geopandas
import contextily as ctx
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.60, 126.95, 37.2, 37.51
extent = [lon1, lon2, lat1, lat2]

data2 = pd.read_csv('D:\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\BNFA_24sites_CO_210914_yslee.csv', header=None).T
q = data2.shape[1]
locations = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_APs_locations_yslee_210903.csv')

data2['No.'] = [x for x in range(1,25)]
data2['No.'] = [x for x in range(1,25) if x !=3]
data2 = pd.merge(data2, locations, on='No.', how='inner')

for source in range(q):

    plt.figure()
    ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
    points = plt.scatter(data2['lon'], data2['lat'], c=data2[source],
                         cmap='jet', alpha=0.8, s=100.2)
    cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
    cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
    cb.ax.tick_params(labelsize=20)
    for i in range(data2.shape[0]):
        x = data2['lon'][i]
        y = data2['lat'][i]
        plt.text(x * 1.00000, y * 0.9998, data2['No.'][i], color='red', fontsize=12)


    # Interpolation for contour mapping
    x, y, z = data2['lon'], data2['lat'], data2[source]
    xi = np.arange(lon1, lon2, 0.01)
    yi = np.arange(lat1, lat2, 0.01)
    xi,yi = np.meshgrid(xi,yi)
    zi = griddata((x,y),z,(xi,yi),method='linear') # linear, cubic, nearest

    levels = np.linspace(0,z.max(),7)
    levels = np.round_(levels,2)

    mapping = plt.contourf(xi,yi,zi, cmap='jet', alpha=0.8)
    cb = plt.colorbar(mapping, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
    #cb.set_label(label='WPSCF', size=20)
    cb.ax.tick_params(labelsize=15)
    plt.title('[CO, source '+str(source+1)+']')

    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)
    plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\BNFA_CO_source ' + str(source+1) +'_yslee_210914.png')
    plt.show()
    plt.close()

# End of BNFA contour mapping



#


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






# For Bayesian, w mapping

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

# version: w8
w = [[37.30717, 126.73441], # 1
    [37.428585, 126.817205], # 2
    [37.30588, 126.90802], # 3
    [37.53942, 126.71340], # 4
    [37.45481, 126.64639], # 5
    [37.37135, 126.63958], # 6
    [37.4315, 126.9911], # 7
    [37.55, 126.9]] # 8
w=pd.DataFrame(w)
w.to_csv('w8.csv')
# version: w16

w = [
    [37.336111, 126.694680], # 1
    [37.325796, 126.749504], # 2
    [37.315482, 126.804328], # 3
    [37.305168, 126.859153], # 4
    [37.336111+0.043766, 126.694680+0.043101],  # 1
    [37.325796+0.043766, 126.749504+0.043101],  # 2
    [37.315482+0.043766, 126.804328+0.043101],  # 3
    [37.305168+0.043766, 126.859153+0.043101],  # 4
    [37.336111 + 0.043766*2, 126.694680 + 0.043101*2],  # 1
    [37.325796 + 0.043766*2, 126.749504 + 0.043101*2],  # 2
    [37.315482 + 0.043766*2, 126.804328 + 0.043101*2],  # 3
    [37.305168 + 0.043766*2, 126.859153 + 0.043101*2],  # 4
    [37.336111 + 0.043766 * 3, 126.694680 + 0.043101 * 3],  # 1
    [37.325796 + 0.043766 * 3, 126.749504 + 0.043101 * 3],  # 2
    [37.315482 + 0.043766 * 3, 126.804328 + 0.043101 * 3],  # 3
    [37.305168 + 0.043766 * 3, 126.859153 + 0.043101 * 3],  # 4
    ]
w=pd.DataFrame(w)
w.to_csv('w16.csv')

# version: w24-iso

w = [
    [37.336111, 126.694680], # 1
    [37.325796, 126.749504], # 2
    [37.315482, 126.804328], # 3
    [37.305168, 126.859153], # 4

    [37.336111+0.032260, 126.694680+0.0298608],  # 1
    [37.325796+0.032260, 126.749504+0.0298608],  # 2
    [37.315482+0.032260, 126.804328+0.0298608],  # 3
    [37.305168+0.032260, 126.859153+0.0298608],  # 4

    [37.336111 + 0.032260*2, 126.694680 + 0.0298608*2],  # 1
    [37.325796 + 0.032260*2, 126.749504 + 0.0298608*2],  # 2
    [37.315482 + 0.032260*2, 126.804328 + 0.0298608*2],  # 3
    [37.305168 + 0.032260*2, 126.859153 + 0.0298608*2],  # 4

    [37.336111 + 0.032260 * 3, 126.694680 + 0.0298608 * 3],  # 1
    [37.325796 + 0.032260 * 3, 126.749504 + 0.0298608 * 3],  # 2
    [37.315482 + 0.032260 * 3, 126.804328 + 0.0298608 * 3],  # 3
    [37.305168 + 0.032260 * 3, 126.859153 + 0.0298608 * 3],  # 4

    [37.336111 + 0.032260 * 4, 126.694680 + 0.0298608 * 4],  # 1
    [37.325796 + 0.032260 * 4, 126.749504 + 0.0298608 * 4],  # 2
    [37.315482 + 0.032260 * 4, 126.804328 + 0.0298608 * 4],  # 3
    [37.305168 + 0.032260 * 4, 126.859153 + 0.0298608 * 4],  # 4

    [37.336111 + 0.032260 * 5, 126.694680 + 0.0298608 * 5],  # 1
    [37.325796 + 0.032260 * 5, 126.749504 + 0.0298608 * 5],  # 2
    [37.315482 + 0.032260 * 5, 126.804328 + 0.0298608 * 5],  # 3
    [37.305168 + 0.032260 * 5, 126.859153 + 0.0298608 * 5],  # 4
    ]

w=pd.DataFrame(w)
w.to_csv('w24_iso.csv')
# Version 3 -> w24
#
# w = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_APs_locations_yslee_210903.csv')
# w = w[['lat','lon']]

# hs.haversine(w.iloc[1], w.iloc[6], unit='km')
# # 15.36
#
N = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\pm25speciation_locations_KoreaNational.csv')
N = N.drop(7)

plt.figure()
ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
#ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
# plt.plot(w[1], w[0], color='blue', marker='X',
#          linestyle='None', markersize=10, label='w9')

for i in range(len(w)):
    # csv file version
    # y = w['lat'][i] # lat
    # x = w['lon'][i] # lon

    # Manual generated version
    y = w[i][0] # lat
    x = w[i][1] # lon

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










# 공모전용_전국 매핑


import geopandas
#import contextily as ctx
import pandas as pd
import math
import matplotlib.pyplot as plt


plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20


df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)


lon1, lon2, lat1, lat2 = 123.0, 131.0, 32.5, 39.5
extent = [lon1, lon2, lat1, lat2]


data2 = pd.read_csv('D:\\Dropbox\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\results_전국_210928_2.csv')


#data2.columns = ['Salts', 'Soil', 'SS', 'Coal', 'Industry', 'Combustions', 'SN', 'Traffic','date','Location number','lat','lon']
data2.columns = ['Salts', 'Industry', 'SN', 'Secondary', 'Soil', 'Traffic','date','Location number','lat','lon']

data2.date = pd.to_datetime(data2.date)

colors = {'Salts':'Blues', 'Soil':'pink_r', 'SS':'Oranges', 'Coal':'Purples',
          'Industry':'bone_r', 'Combustions':'Reds', 'SN':'Greens', 'Traffic':'Wistia', 'Secondary':'Reds'}

sources = ['Secondary','Soil','Traffic']

#max_range = {'Industry':1,'Secondary':80,'Mobile':10}

for i in range(len(data2.drop_duplicates('date').date)):
    day = data2.drop_duplicates('date').date[i]
    day = str(day)[:10]
    for source in sources:
        data3 = data2[data2['date'] == day]

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='white')
        #ctx.add_basemap(ax, zoom=13, crs='epsg:4326')
        # plt.plot(data['lon'], data['lat'], color='blue', marker='X',
        #          linestyle='None', markersize=0.15, label='Monitoring Site')
        points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
        #                     vmin=0, vmax=30,
        #                     vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max()) % 5,
                             cmap=colors[source],
                             alpha=0.6,
                             s=0.7,
                             vmin=0, vmax=data2[source].max()/1.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('['+day+', '+source+']')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\mappingresults\\'+day+', '+source+'.png')
        plt.close()






# 2021-09-29 BNFA results Interpolation by sources (24 APs)


import geopandas
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.60, 126.95, 37.2, 37.51
extent = [lon1, lon2, lat1, lat2]


for source_number in range(1,9):

    # Contribution data: date by locations
    data2 = pd.read_csv('D:\\Dropbox\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\BNFA_24sites_PM25_210929_yslee_source_'
                        +str(source_number)+'.csv', header=None).T
    locations = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_APs_locations_yslee_210903.csv')


    data2['No.'] = [x for x in range(1,25)]
    data2 = pd.merge(data2, locations, on='No.', how='inner')

    date = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_AP_PM25_kNN_yslee_210910.csv')['date']

    for i, day in enumerate(date):

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='lightgray')
        points = plt.scatter(data2['lon'], data2['lat'], c=data2[i],
                             cmap='jet', alpha=0.8, s=100.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        for i in range(data2.shape[0]):
            x = data2['lon'][i]
            y = data2['lat'][i]
            plt.text(x * 1.00000, y * 0.9998, data2['No.'][i], color='red', fontsize=12)


        # Interpolation for contour mapping
        x, y, z = data2['lon'], data2['lat'], data2[i]
        xi = np.arange(lon1, lon2, 0.01)
        yi = np.arange(lat1, lat2, 0.01)
        xi,yi = np.meshgrid(xi,yi)
        zi = griddata((x,y),z,(xi,yi),method='linear') # linear, cubic, nearest

        levels = np.linspace(0,z.max(),7)
        levels = np.round_(levels,2)

        mapping = plt.contourf(xi,yi,zi, cmap='jet', alpha=0.8)
        cb = plt.colorbar(mapping, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        #cb.set_label(label='WPSCF', size=20)
        cb.ax.tick_params(labelsize=15)
        plt.title('['+day+', '+'source 1'+']')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.savefig('D:\\OneDrive - SNU\\바탕 화면\\mappingresults\\BNFA_PM25_source_'+str(source_number)+day+'_yslee.png')
        plt.show()
        plt.close()

# End of BNFA contour mapping
