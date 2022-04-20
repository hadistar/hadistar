import geopandas
import pandas as pd
import cartopy
import matplotlib.pyplot as plt
from scalebar import scale_bar

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

source_1 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_Korea_source_1.csv', header=None)
source_2 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_Korea_source_2.csv', header=None)
source_3 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_Korea_source_3.csv', header=None)
source_4 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_Korea_source_4.csv', header=None)
source_5 = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_Korea_source_5.csv', header=None)

latlon = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\s_new_map.csv')

dates = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\BSMRM_dates.csv')

locations_monitoring = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_BSMRM\\Analysis\\MATLAB files\\locations_monitoring.csv', header=None)
locations_monitoring.columns = ['lat','lon']

location_Daejeon = [36.1900, 127.2400]

cmaps = ['Reds','Oranges','Purples','Greens','Blues']

# 기본셋!
'''
plt.figure(figsize=(10,10))
ax = plt.axes(projection=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='darkslategray')
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
#ax.add_feature(cartopy.feature.RIVERS)

lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.0, 39.5
ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
gl.xlabels_top = False
gl.ylabels_left = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

plt.plot(locations_monitoring['lon'], locations_monitoring['lat'], color='blue', marker='X',
         linestyle='None' , markersize=10, label='Monitoring site')
plt.plot(location_Daejeon[1], location_Daejeon[0], color='black', marker='^',
         linestyle='None' , markersize=10, label='Validation site')

z = source_1.iloc[5,:].copy()

points = plt.scatter(latlon['lon'], latlon['lat'], c=z,
                     vmin=0, vmax=30,
                     cmap='Reds', alpha=0.8, s=3.0)
cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=15)
cb.ax.tick_params(labelsize=15)

plt.legend(loc='upper right', prop={'size': 14})

scale_bar(ax, (0.05, 0.05), 100)
plt.title('Date', fontsize=15)
plt.tight_layout()

plt.show()
#plt.savefig('underlying_locations_w12.jpg')
plt.close()

'''




# For 사용

for i in range(103):

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='darkslategray')
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

    lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.0, 39.5
    ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
    gl.top_labels_top = False
    gl.left_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    plt.plot(locations_monitoring['lon'], locations_monitoring['lat'], color='blue', marker='X',
             linestyle='None' , markersize=9, alpha=0.7, label='Monitoring site')
    plt.plot(location_Daejeon[1], location_Daejeon[0], color='black', marker='^',
             linestyle='None' , markersize=9, alpha=0.7, label='Validation site')

    z = source_5.iloc[i,:].copy() #

    points = plt.scatter(latlon['lon'], latlon['lat'], c=z,
                         vmin=0, vmax=0.8, #
                         cmap=cmaps[5-1], alpha=0.8, s=3.0) #
    cb = plt.colorbar(points, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)
    cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=15)
    cb.ax.tick_params(labelsize=15)

    plt.legend(loc='upper right', prop={'size': 14})

    scale_bar(ax, (0.05, 0.05), 100)
    #plt.title(dates.iloc[i][0], fontsize=15)
    plt.tight_layout()
    #plt.show()
    plt.savefig('D:\\temp_source_5\\Source_5_'+dates.iloc[i][0]+'.png') ##
    plt.close()

