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

df = pd.read_csv('Anew_source5.csv', header=None)
s2 = pd.read_csv('s2.csv')
s_new = pd.read_csv('s_new.csv')
date = pd.read_csv('date.csv')

# 1. Using Folium

for i, day in enumerate(df.columns[2:]):
    m = folium.Map(location=[35.5502, 126.982],
                   zoom_start=7,
                   tiles='cartodbpositron',
                   control_scale=True)

    c_max = math.ceil(df.iloc[:,i+2].max())
    colormap = cm.LinearColormap(colors=['white', 'yellow', 'orange', 'red'],
                                 index=[0, c_max/3, c_max*2/3, c_max],
                                 #                             tick_labels=[0,15,30,45],
                                 #                             scale_width=400, scale_heigh=50,
                                 vmin=0, vmax=c_max,
                                 caption='concentration (ug/m3)')

    print(i, day)
    for pt in range(int(len(df))):
        pt = pt
        color = colormap(df.iloc[pt][i+2])
        folium.CircleMarker(location = [df.iloc[pt][0],df.iloc[pt][1]],
                            radius=0.01,
                            fill=True,
                            color=color,
                            fill_color=color,
                            fill_opacity=0.2,
                            line_opacity=0.2).add_to(m)
    m.add_child(colormap)
    m.save('./test/test_'+str(day).replace('/','_')+'.html')


    browser = webdriver.Chrome('D:/chromedriver.exe')
    browser.get('D:/hadistar/test/test_'+str(day).replace('/','_')+'.html')

    #Give the map tiles some time to load
    time.sleep(10)
    browser.save_screenshot('./test/test_'+str(day).replace('/','_')+'.png')
    browser.quit()

# 2. Using Pandas and Cartopy_

for i in range(len(df.columns)):
    day = date.iloc[i].date
    print(i, day)

    df_z = df.iloc[:,i]

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.5, 39.5
    ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    #plt.plot(np.nan, np.nan, color=None, markersize=0, label='['+df.iloc[:,2].name+']')
    plt.plot(s2['lon'], s2['lat'], color='blue', marker='X',
             linestyle='None' , markersize=15, label='Monitoring Site')

    points = plt.scatter(s_new['lon'], s_new['lat'], c=df_z,
                         vmin=0, vmax=math.ceil(df_z.max()) + 5 - math.ceil(df_z.max())%5,
                         cmap='Reds', alpha=0.8, s=3.0)
    cb = plt.colorbar(points, orientation='vertical',ticklocation='auto', shrink=0.5, pad=0.1)

    cb.set_label(label='Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')', size=20)
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()

    plt.legend(title='['+day+']', loc='upper right')
    plt.savefig('./test2/test_'+str(day).replace('/','_')+'.jpg')
    plt.close()



# 2. Using Pandas and Cartopy_ Same legend ver.

for i in range(1400,1461): #range(len(df.columns)):
    day = date.iloc[i].date
    day_ = datetime.datetime.strptime(day, '%m/%d/%Y')
    print(i, day)

#    df_z = df.iloc[:,i]
    df_z = pd.read_csv('Anew_source3_w12_T61.csv', header=None).iloc[:,i-1400]
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='dimgray')
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.5, 39.5
    ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    #plt.plot(np.nan, np.nan, color=None, markersize=0, label='['+df.iloc[:,2].name+']')
    plt.plot(s2['lon'], s2['lat'], color='blue', marker='X',
             linestyle='None' , markersize=15, label='Monitoring Site')

    # For annotation (city name)
    for loc in range(len(s2)):
        plt.annotate(s2['location'][loc], (s2['lon'][loc]-0.15, s2['lat'][loc]+0.15),
                     fontsize=15, color='blue')

#    points = plt.scatter

#    [35.86588034847358, 128.5934264814463] # Daegu
#    [37.346780, 126.740031] # Siheung

    plt.plot([126.740031,128.5934264814463], [37.346780, 35.86588034847358], # Siheung, Daegu
                color='black', marker='v',
                linestyle='None', markersize=15, label='Unmonitored Site')
    plt.annotate('Siheung', (126.740031 - 0.3, 37.346780 - 0.3),
                 fontsize=15, color='black')
    plt.annotate('Daegu', (128.593426 - 0.3, 35.86588 - 0.3),
                 fontsize=15, color='black')


    points = plt.scatter(s_new['lon'], s_new['lat'], c=df_z,
                         vmin=0, vmax=10,
                         cmap='Wistia', alpha=0.8, s=3.0)
    cb = plt.colorbar(points, orientation='horizontal',ticklocation='auto', shrink=0.5, pad=0.1)
    cb.set_label(label='Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')', size=20)
    cb.ax.tick_params(labelsize=20)
    plt.tight_layout()

    plt.legend(title='['+str(day_.date())+']', loc='upper right')
    scale_bar(ax, (0.05, 0.05), 100)
#    ax.annotate("N", xy=(124, 39), xytext=(0, 0.5), arrowprops = dict(arrowstyle="->"))
#    plt.show()


    plt.savefig('./test2/Anew_source3_w12_T61_'+str(day_.date())+'.jpg')

    plt.close()




# 2-3. Using Pandas and Cartopy_ Underlying locations

underloc = pd.read_csv('w16.csv')

#underloc = underloc.drop([0,3,4,8,15],0) # for new w11

underloc = underloc.drop([4,8,12,15],0) # for w12

plt.figure(figsize=(10,10))
ax = plt.axes(projection=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='dimgray')
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)

lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.0, 39.5
ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

#plt.plot(np.nan, np.nan, color=None, markersize=0, label='['+df.iloc[:,2].name+']')
plt.plot(s2['lon'], s2['lat'], color='blue', marker='X',
         linestyle='None' , markersize=15, label='Monitoring site')

# For annotation (city name)
for loc in range(len(s2)):
    plt.annotate(s2['location'][loc], (s2['lon'][loc]-0.15, s2['lat'][loc]-0.25),
                 fontsize=15, color='blue')

plt.plot(underloc['lon'], underloc['lat'], color='red', marker='P',
         linestyle='None' , markersize=13, label='Underlying process locations')


#    points = plt.scatter

#    [35.86588034847358, 128.5934264814463] # Daegu
#    [37.346780, 126.740031] # Siheung

plt.plot([126.740031,128.5934264814463], [37.346780, 35.86588034847358], # Siheung, Daegu
            color='black', marker='v',
            linestyle='None', markersize=15, label='Unmonitored site')
plt.annotate('Siheung', (126.740031 - 0.3, 37.346780 - 0.3),
             fontsize=15, color='black')
plt.annotate('Daegu', (128.593426 - 0.3, 35.86588 - 0.3),
             fontsize=15, color='black')


plt.tight_layout()

plt.legend(loc='upper right')
scale_bar(ax, (0.05, 0.05), 100)
#plt.show()
plt.savefig('underlying_locations_w12.jpg')

plt.close()



# For AirKorea data

AirKorea = pd.read_csv('AirKorea_20191103.csv', encoding='euc-kr')

plt.figure(figsize=(10, 10))
ax = plt.axes(projection=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)

lon1, lon2, lat1, lat2 = 123.5, 130.5, 32.5, 39.5
ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}


points = plt.scatter(AirKorea['Longitude'], AirKorea['Latitude'], c=AirKorea['PM25'],
                     vmin=0, vmax=60,
                     label='AirKorea stations',
                     cmap='Reds', s=20.0)
cb = plt.colorbar(points, orientation='horizontal', ticklocation='auto', shrink=0.5, pad=0.1)

cb.set_label(label='Concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')', size=20)
cb.ax.tick_params(labelsize=20)
plt.tight_layout()

plt.legend(title='[2019-11-03]', loc='upper right')
scale_bar(ax, (0.05, 0.05), 100)

plt.savefig('AirKorea_test_191103.jpg')

plt.close()
