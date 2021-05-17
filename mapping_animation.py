import folium
import pandas as pd
import branca.colormap as cm
from selenium import webdriver
import time
import math
import matplotlib.pyplot as plt
import cartopy
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 20

df = pd.read_csv('Anew_source1.csv')

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

# Using Pandas and Cartopy


fig = plt.figure(figsize=(10,10))

ax = plt.axes(projection=cartopy.crs.PlateCarree())

ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)

lon1, lon2, lat1, lat2 = 124, 130, 33, 39
ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
gl.xlabel_style = {'size': 15}
gl.ylabel_style = {'size': 15}

import numpy as np
lon = np.linspace(125,128,100)
lat = np.linspace(35,38,100)
data = np.linspace(10,20,100)

points = plt.scatter(lon, lat, c=data, cmap='Reds', alpha=0.5)
cb = plt.colorbar(points, orientation='vertical',ticklocation='auto', shrink=0.6, pad=0.1)
cb.set_label(label='Concentration (ug/m3)', size=20)
cb.ax.tick_params(labelsize=20)
fig.tight_layout()


plt.show()