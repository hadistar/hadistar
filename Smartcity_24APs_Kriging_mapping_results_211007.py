# 2021-10-06 BNFA results Interpolation mapping - 10 sources
# Weight matrix calculation

import geopandas
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from haversine import haversine
from math import cos, sin, asin, sqrt, radians

def calc_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

plt.rc('font', family='aerial')
plt.rcParams['font.size'] = 20

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.60, 126.95, 37.2, 37.51
extent = [lon1, lon2, lat1, lat2]

data2 = pd.read_csv('D:\\Dropbox\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\BNFA_24sites_PM25_211005_yslee_q10.csv', header=None).T
q = data2.shape[1]
locations = pd.read_csv('D:\\Dropbox\\Bayesian modeling\\Young Su Lee\\SH_APs_locations_yslee_210903.csv')
sources = ['Salts',	'Soil', 'SS', 'Coal', 'biomass', 'indus_smel', 'indus_oil', 'heating', 'SN', 'traffic']
data2.columns = sources
data2['No.'] = [x for x in range(1,25)]
data2 = pd.merge(data2, locations, on='No.', how='inner')

s_new = pd.read_csv('D:\\Dropbox\\hadistar\\Matlab\\Smartcity_BSMRM_202108\\loc_SH_100mx100m.csv')
s_new[sources] = np.nan
# dist_degree_new = np.zeros([s_new.shape[0],data2.shape[0]])
# Knew = np.zeros([s_new.shape[0],data2.shape[0]])
#
# sigma_K = 0.05
#
# for jj in range(s_new.shape[0]):
#     for m in range(data2.shape[0]):
#         x = (s_new.iloc[jj,0],s_new.iloc[jj,1])
#         y = (data2['lat'][m],data2['lon'][m])
#         dist_degree_new[jj,m] = haversine(x,y, unit='km')/110
#         Knew[jj,m] = 1/(math.pi*2*sigma_K**2)*math.exp(-dist_degree_new[jj,m]**2/(2*sigma_K**2))
#

source = 'Salts'
for source in sources:

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
    # xi = np.arange(lon1, lon2, 0.01)
    # yi = np.arange(lat1, lat2, 0.01)
    # xi,yi = np.meshgrid(xi,yi)
    # zi = griddata((x,y),z,(xi,yi),method='cubic') # linear, cubic, nearest

    xi = s_new['lon'].drop_duplicates().sort_values()
    yi = s_new['lat'].drop_duplicates().sort_values()
    xi,yi = np.meshgrid(xi,yi)
    zi = griddata((x,y),z,(xi,yi),method='linear') # linear, cubic, nearest

    for s in range(s_new.shape[0]):
        x_loc = np.where(yi[:,0]==s_new.iloc[s][0])
        y_loc = np.where(xi[0]==s_new.iloc[s][1])
        s_new[source][s] = zi[x_loc,y_loc]


    levels = np.linspace(0,z.max(),7)
    levels = np.round_(levels,2)

    mapping = plt.contourf(xi,yi,zi, cmap='jet', alpha=0.8)
    cb = plt.colorbar(mapping, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
    #cb.set_label(label='WPSCF', size=20)
    cb.ax.tick_params(labelsize=15)
    plt.title('[PM25, source '+source+']')

    ax.set_xlim(lon1, lon2)
    ax.set_ylim(lat1, lat2)
    # plt.savefig('D:\\mappingresults\\BNFA_PM25_source ' + str(source+1) +'_yslee_211005.png')
    plt.show()
    plt.close()

s_new.to_csv('BNFA_PMF_weight_matrix_211006.csv',index=False)


# End of BNFA contour mapping

# Loading PMF results
s_new = pd.read_csv('BNFA_PMF_weight_matrix_211006.csv')

PMF = pd.read_csv('data\\SH_PMF_meteo_daily_v8.csv', encoding='euc-kr')
PMF = PMF[['date', 'PM25_observed', 'salts', 'soil', 'SS', 'coal',
           'biomass', 'indus_s', 'indus_o', 'heating', 'SN', 'mobile']]

s_new = s_new.dropna().reset_index()
s_new['location No.'] = range(1,s_new.shape[0]+1)

result = pd.DataFrame()
for p in range(PMF.shape[0]):
    temp = pd.DataFrame(np.multiply(np.array(PMF.iloc[p][2:]),np.array(s_new[sources])))

    temp.columns = ['Salts',	'Soil', 'SS', 'Coal', 'biomass', 'indus_smel', 'indus_oil', 'heating', 'SN', 'traffic']
    temp['date'] = PMF['date'][p]
    temp['lat'] = s_new['lat']
    temp['lon'] = s_new['lon']
    temp['location No.'] = s_new['location No.']

    result = result.append(temp)

result.to_csv('BNFA_PMF_results_for_mapping_211006_yslee.csv',index=False)

# Station No = 2531인 곳이 시흥시 필터 포집 장소

# 1:1 Plot comparison
from sklearn import linear_model
import sklearn
import math

spatial_results = pd.read_csv('BNFA_PMF_results_for_mapping_211006_yslee.csv')
SH_point = spatial_results.loc[spatial_results['location No.']==2531]

x = np.array(PMF['PM25_observed'])
#y = np.array(SH_point[sources].sum(axis=1))
y = np.array(spatial_results[sources+['date']].groupby('date').mean().sum(axis=1))

linreg = linear_model.LinearRegression()
model = linreg.fit(x.reshape(-1,1), y.reshape(-1,1))

intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure(figsize=(10,10))
#plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
plt.plot(x, y, 'ro', markersize=8, mfc='none')

plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,100],[0,100], 'k--')
#plt.plot([0,math.ceil(max(x.max(),y.max()))],[0,math.ceil(max(x.max(),y.max()))], 'k--')
plt.xlabel('Concentration of sampled filter ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.ylabel('Concentration of modeling ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.text(x.max() * 0.1, y.max() * 0.6,
         'y = %0.2fx + %0.2f\nr$^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))
#plt.axis([0, math.ceil(max(x.max(),y.max())), 0, math.ceil(max(x.max(),y.max()))])
plt.axis([0,100,0,100])
plt.grid(True, linestyle='--')

plt.tight_layout()
plt.show()


# Time-series plot

## For time-series plot

PMF['date'] = pd.to_datetime(PMF['date'])

plt.figure(figsize=(12,6))
plt.grid('--')
plt.plot(PMF['date'], x, 'bo-', label='Sampled filter')
plt.plot(PMF['date'], y, 'ro-', label='Modeling results')

#plt.xlim([datetime.date(2019, 1, 1), datetime.date(2019, 12, 31)])
plt.xticks(rotation=45)
plt.ylim([0,100])
plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.tight_layout()
plt.legend()

plt.show()



# Mapping!!!
# 2021-09-13 스마트시티 베이지안 매핑 - BSMRM - Monotone version

import geopandas
import pandas as pd
import matplotlib.pyplot as plt



plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20


df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]

spatial_results = pd.read_csv('BNFA_PMF_results_for_mapping_211006_yslee.csv')

spatial_results['산업 배출'] = spatial_results['indus_smel'] + spatial_results['indus_oil']
spatial_results['기타 연소'] = spatial_results['biomass'] + spatial_results['heating']
spatial_results = spatial_results.rename(columns={'Salts':'해염 입자','Soil':'토양','SS':'2차 황산염','Coal':'석탄 연소','SN':'2차 질산염','traffic':'자동차'})

spatial_results = spatial_results[['date','location No.','lat','lon','해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차']]

colors = {'해염 입자':'Blues', '토양':'pink_r', '2차 황산염':'Oranges', '석탄 연소':'Purples',
          '산업 배출':'bone_r', '기타 연소':'Reds', '2차 질산염':'Greens', '자동차':'Wistia'}

vmaxs = {'해염 입자':3.0, '토양':2.0, '2차 황산염':20.0, '석탄 연소':3.0,
          '산업 배출':7.0, '기타 연소':20.0, '2차 질산염':30.0, '자동차':20.0}
sources = ['해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차']

spatial_results.date = pd.to_datetime(spatial_results.date)

for i in range(len(spatial_results.drop_duplicates('date').date)):

    if True:

        day = spatial_results.drop_duplicates('date').reset_index().date[i]
        day = str(day)[:10]

        # sum(PM2.5) plot

        data3 = spatial_results[spatial_results['date'] == day]


        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

        points = plt.scatter(data3['lon'], data3['lat'], c=data3[sources].sum(axis=1),
                             vmin=0, vmax=80,
                             cmap='jet', alpha=0.75, s=0.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('[' + day + ', PM2.5_total]')

        # North arrow
        ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)

        # Scale bar

        plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
        plt.text(126.8694, 37.305, '3 km', ha='center')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.tight_layout()

        plt.savefig('D:\\mappingresults\\PM25_total_' + day + '.png')
        plt.close('all')



        # mapping by source
        for source in sources:
            # Box plot
            # plt.boxplot(spatial_results[source])
            # plt.title(source)
            # plt.show()
            # plt.close()
            #
            data3 = spatial_results[spatial_results['date'] == day]

            plt.figure()
            ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

            points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
                                 vmin=0, vmax=30,#vmaxs[source],
                                 cmap=colors[source], alpha=0.75, s=0.2)
            cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
            cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
            cb.ax.tick_params(labelsize=20)
            plt.title('['+day+', '+source+']')

            # North arrow
            ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                        arrowprops=dict(facecolor='black', width=5, headwidth=15),
                        ha='center', va='center', fontsize=20,
                        xycoords=ax.transAxes)

            # Scale bar

            plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
            plt.text(126.8694,37.305,'3 km', ha='center')

            ax.set_xlim(lon1, lon2)
            ax.set_ylim(lat1, lat2)
            plt.tight_layout()

            plt.savefig('D:\\mappingresults\\'+source + '_'+ day+'.png')
            plt.close('all')



spatial_results.to_csv('스마트시티_매핑결과_서울대_raw.csv',index=False)


## 2021-10-08
## Case별 분석

import pandas as pd

spatial_results = pd.read_csv('스마트시티_매핑결과_서울대_raw.csv')
sources = ['해염 입자','석탄 연소','기타 연소','산업 배출','토양','2차 질산염','2차 황산염','자동차']

spatial_results['PM25'] = spatial_results[sources].sum(axis=1)
spatial_results['date'] = pd.to_datetime(spatial_results['date'])

## 여기부터

spatial_results['dow'] = spatial_results['date'].dt.day_name()    #<- 요일 붙이기 코드임
spatial_results['month'] = spatial_results['date'].dt.month

def season_recog(row):
    if row.month <=2:
        return 'winter'
    elif row.month <=5:
        return 'spring'
    elif row.month <=8:
        return 'summer'
    elif row.month <=11:
        return 'fall'
    else:
        return 'winter'

spatial_results['season'] = spatial_results.apply(season_recog, axis=1)


### 요일별, 월별, 계절별



import geopandas
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 20

df1 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SIG_202101\\TL_SCCO_SIG.shp')
df1 = df1.to_crs(epsg=4326)

df2 = geopandas.read_file('D:\\OneDrive - SNU\\QGIS\\SH.shp')
df2 = df2.to_crs(epsg=4326)

lon1, lon2, lat1, lat2 = 126.65, 126.9, 37.3, 37.5
extent = [lon1, lon2, lat1, lat2]

colors = {'해염 입자':'Blues', '토양':'pink_r', '2차 황산염':'Oranges', '석탄 연소':'Purples',
          '산업 배출':'bone_r', '기타 연소':'Reds', '2차 질산염':'Greens', '자동차':'Wistia'}

vmaxs = {'해염 입자':3.0, '토양':2.0, '2차 황산염':20.0, '석탄 연소':3.0,
          '산업 배출':7.0, '기타 연소':20.0, '2차 질산염':30.0, '자동차':20.0}

spatial_results.date = pd.to_datetime(spatial_results.date)


# 1. 요일별

for i in range(len(spatial_results.drop_duplicates('dow').dow)):
    if True:

        dow = spatial_results.drop_duplicates('dow').reset_index().dow[i]
        # sum(PM2.5) plot
        data3 = spatial_results[spatial_results['dow'] == dow]
        data3 = data3.groupby('location No.').mean()

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

        points = plt.scatter(data3['lon'], data3['lat'], c=data3[sources].sum(axis=1),
                             vmin=0, vmax=80,
                             cmap='jet', alpha=0.75, s=0.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('[' + dow + ', PM2.5_total]')

        # North arrow
        ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)

        # Scale bar

        plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
        plt.text(126.8694, 37.305, '3 km', ha='center')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.tight_layout()

        plt.savefig('D:\\mappingresults\\dow_PM25_total_' + dow + '.png')
        plt.close('all')



        # mapping by source
        for source in sources:
            # Box plot
            # plt.boxplot(spatial_results[source])
            # plt.title(source)
            # plt.show()
            # plt.close()
            #
            data3 = spatial_results[spatial_results['dow'] == dow]
            data3 = data3.groupby('location No.').mean()

            plt.figure()
            ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

            points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
                                 vmin=0, vmax=vmaxs[source],
                                 cmap=colors[source], alpha=0.75, s=0.2)
            cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
            cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
            cb.ax.tick_params(labelsize=20)
            plt.title('['+dow+', '+source+']')

            # North arrow
            ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                        arrowprops=dict(facecolor='black', width=5, headwidth=15),
                        ha='center', va='center', fontsize=20,
                        xycoords=ax.transAxes)

            # Scale bar

            plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
            plt.text(126.8694,37.305,'3 km', ha='center')

            ax.set_xlim(lon1, lon2)
            ax.set_ylim(lat1, lat2)
            plt.tight_layout()

            plt.savefig('D:\\mappingresults\\dow_'+source + '_'+ dow+'.png')
            plt.close('all')


# 2. 계절별

for i in range(len(spatial_results.drop_duplicates('season').season)):
    if True:

        season = spatial_results.drop_duplicates('season').reset_index().season[i]
        # sum(PM2.5) plot
        data3 = spatial_results[spatial_results['season'] == season]
        data3 = data3.groupby('location No.').mean()

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

        points = plt.scatter(data3['lon'], data3['lat'], c=data3[sources].sum(axis=1),
                             vmin=0, vmax=80,
                             cmap='jet', alpha=0.75, s=0.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('[' + season + ', PM2.5_total]')

        # North arrow
        ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)

        # Scale bar

        plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
        plt.text(126.8694, 37.305, '3 km', ha='center')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.tight_layout()

        plt.savefig('D:\\mappingresults\\season_PM25_total_' + season + '.png')
        plt.close('all')



        # mapping by source
        for source in sources:
            # Box plot
            # plt.boxplot(spatial_results[source])
            # plt.title(source)
            # plt.show()
            # plt.close()
            #
            data3 = spatial_results[spatial_results['season'] == season]
            data3 = data3.groupby('location No.').mean()

            plt.figure()
            ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

            points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
                                 vmin=0, vmax=vmaxs[source],
                                 cmap=colors[source], alpha=0.75, s=0.2)
            cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
            cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
            cb.ax.tick_params(labelsize=20)
            plt.title('['+season+', '+source+']')

            # North arrow
            ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                        arrowprops=dict(facecolor='black', width=5, headwidth=15),
                        ha='center', va='center', fontsize=20,
                        xycoords=ax.transAxes)

            # Scale bar

            plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
            plt.text(126.8694,37.305,'3 km', ha='center')

            ax.set_xlim(lon1, lon2)
            ax.set_ylim(lat1, lat2)
            plt.tight_layout()

            plt.savefig('D:\\mappingresults\\season_'+source + '_'+ season+'.png')
            plt.close('all')


# 3. 월별

for i in range(len(spatial_results.drop_duplicates('month').month)):
    if True:

        month = spatial_results.drop_duplicates('month').reset_index().month[i]
        # sum(PM2.5) plot
        data3 = spatial_results[spatial_results['month'] == month]
        data3 = data3.groupby('location No.').mean()

        plt.figure()
        ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

        points = plt.scatter(data3['lon'], data3['lat'], c=data3[sources].sum(axis=1),
                             vmin=0, vmax=80,
                             cmap='jet', alpha=0.75, s=0.2)
        cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
        cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
        cb.ax.tick_params(labelsize=20)
        plt.title('[Month_' + str(month) + ', PM2.5_total]')

        # North arrow
        ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=20,
                    xycoords=ax.transAxes)

        # Scale bar

        plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
        plt.text(126.8694, 37.305, '3 km', ha='center')

        ax.set_xlim(lon1, lon2)
        ax.set_ylim(lat1, lat2)
        plt.tight_layout()

        plt.savefig('D:\\mappingresults\\month_PM25_total_' + str(month) + '.png')
        plt.close('all')



        # mapping by source
        for source in sources:
            # Box plot
            # plt.boxplot(spatial_results[source])
            # plt.title(source)
            # plt.show()
            # plt.close()
            #
            data3 = spatial_results[spatial_results['month'] == month]
            data3 = data3.groupby('location No.').mean()

            plt.figure()
            ax = df1.plot(figsize=(10, 10), alpha=1.0, edgecolor='k', facecolor='whitesmoke')

            points = plt.scatter(data3['lon'], data3['lat'], c=data3[source],
                                 vmin=0, vmax=vmaxs[source],
                                 cmap=colors[source], alpha=0.75, s=0.2)
            cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
            cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
            cb.ax.tick_params(labelsize=20)
            plt.title('[Month_'+str(month)+', '+source+']')

            # North arrow
            ax.annotate('N', xy=(0.9, 0.9), xytext=(0.9, 0.8),
                        arrowprops=dict(facecolor='black', width=5, headwidth=15),
                        ha='center', va='center', fontsize=20,
                        xycoords=ax.transAxes)

            # Scale bar

            plt.plot([126.85, 126.88839235], [37.315, 37.315], color='k', linestyle='-', linewidth=6)
            plt.text(126.8694,37.305,'3 km', ha='center')

            ax.set_xlim(lon1, lon2)
            ax.set_ylim(lat1, lat2)
            plt.tight_layout()

            plt.savefig('D:\\mappingresults\\month_'+source + '_'+ str(month)+'.png')
            plt.close('all')







### 2021-10-12 회의 내용, 1번 위치에 대한 소스별 기여도 도출 후 PMF 결과와 비교해보기(검증용)


import pandas as pd
import matplotlib.pyplot as plt


df_contri = pd.read_csv('D:\Dropbox\hadistar\Matlab\Smartcity_BSMRM_202108\BNFA_loc_contri_location1_yslee_211013.csv')

df_contri.columns = ['date','Salts','SN','Combustion','Coal','Biomaa','Industry-oil','SS','Industry-smelting','Traffic','Soil']
# Contribution plot
Sources_name = ['Salts','SN','Combustion','Coal','Biomaa','Industry-oil','SS','Industry-smelting','Traffic','Soil']
df_contri['date'] = pd.to_datetime(df_contri['date'])

contri_percent = df_contri.mean(numeric_only=True)/df_contri.mean(numeric_only=True).sum()*100

fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(12,24), sharex=True)

for i, s in enumerate(Sources_name):

    print(i, s)
    ax1 = axes[i]
    ax1.plot(df_contri['date'], df_contri[s], 'k-')
    ax1.fill_between(df_contri['date'],0,df_contri[s], color='lightgrey')
    ax1.text(1.0,0.8,
             s+" "+str(round(df_contri[s].mean(), 2))+" ${\mu}$" +'g/m'+"$^3$ ("+str(contri_percent[i].round(2))+"%)  ",
             size=18, ha='right', transform=ax1.transAxes)
    ax1.set_ylim(bottom=0)
    max = round(df_contri[s].max(), 2) #math.ceil(df_contri[s].max()) + 4 - math.ceil(df_contri[s].max()) % 4
    ax1.set_ylim([0,max])
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.grid(True, axis='y', linestyle='--')

fig.text(0.05, 0.5, 'Mass Concentration ('+ "${\mu}$" +'g/m'+"$^3$"+")",
         ha='center',va='center',rotation='vertical', fontsize=20)
fig.autofmt_xdate(rotation=45)
plt.show()