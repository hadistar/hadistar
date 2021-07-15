import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

import os
os.getcwd()
os.chdir('D:/hadistar/Python_study_2021_summer')

df1 = pd.read_csv('210713_PM2.5_mean_day.csv')
df2 = pd.read_csv('210713_AirKorea_20191103.csv', encoding='euc-kr')

df2 = df2.rename(columns={'측정소코드':'Station code'})

df3 = pd.merge(df1, df2[['Station code', 'Latitude', 'Longitude']], how='inner', on='Station code')

df3.head(3)

df4 = df3.loc[df3['Station code']==111121]

# Matplotlib manual and examples: https://matplotlib.org/stable/index.html

# Basic plot
plt.figure()
plt.plot(df4.date, df4.PM25_mean,
         color = 'red', marker='o', markersize=5, linestyle='')
plt.show()

# other styles
# color: https://matplotlib.org/stable/gallery/color/named_colors.html
# marker: https://matplotlib.org/stable/api/markers_api.html

plt.figure()
plt.plot(df4.date, df4.PM25_mean, 'g>', markersize=5)
plt.show()

#
plt.figure()
plt.plot(df4.date, df4.PM25_mean, 'k-', linewidth=5)
plt.show()

#
plt.figure()
plt.plot(df4.date, df4.PM25_mean, 'k-', linewidth=0.5)
plt.plot(df3.loc[df3['Station code']==823801]['date'], df3.loc[df3['Station code']==823801]['PM25_mean'],
         'g>', markersize=2)
plt.plot(df3.loc[df3['Station code']==131144]['date'], df3.loc[df3['Station code']==131144]['PM25_mean'],
         'r-', linewidth=0.5)
plt.show()


# Drop Na
df3['date'] = pd.to_datetime(df3['date'])
plt.figure()
plt.plot(pd.to_datetime(df4.date), df4.PM25_mean, 'k-', linewidth=0.5)
plt.plot(df3.loc[df3['Station code']==823801]['date'], df3.loc[df3['Station code']==823801]['PM25_mean'],
         'g>', markersize=2)
plt.plot(df3.loc[df3['Station code']==131144]['date'], df3.loc[df3['Station code']==131144]['PM25_mean'],
         'r-', linewidth=0.5)
plt.show()


# Adding options
plt.figure()
plt.plot(pd.to_datetime(df4.date), df4.PM25_mean, 'k-', linewidth=0.5)
plt.plot(df3.loc[df3['Station code']==823801]['date'], df3.loc[df3['Station code']==823801]['PM25_mean'],
         'g>', markersize=2)
plt.plot(df3.loc[df3['Station code']==131144]['date'], df3.loc[df3['Station code']==131144]['PM25_mean'],
         'r-', linewidth=0.5)

plt.title('Test graph')
plt.xlabel('Date')
plt.xlim()
plt.ylabel('Concentration')
plt.ylim([0,100])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import datetime

# Adding options
plt.figure()
plt.plot(pd.to_datetime(df4.date), df4.PM25_mean, 'k-', linewidth=0.5)
plt.plot(df3.loc[df3['Station code']==823801]['date'], df3.loc[df3['Station code']==823801]['PM25_mean'],
         'g>', markersize=2)
plt.plot(df3.loc[df3['Station code']==131144]['date'], df3.loc[df3['Station code']==131144]['PM25_mean'],
         'r-', linewidth=0.5)

plt.title('Test graph')
plt.xlabel('Date')
plt.xlim([datetime.date(2019,1,1), datetime.date(2019,6,30)])
plt.ylabel('Concentration')
plt.ylim([0,100])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('testplot_210720.png')
plt.show()

# Scatter plot

plt.figure()
plt.scatter(df2.Longitude, df2.Latitude)
plt.show()

# cmap list:
plt.figure()
plt.scatter(df2.Longitude, df2.Latitude, c=df2.PM25,
            vmin = 0, vmax = 100,
            cmap = 'Reds')

plt.show()


# cmap list: https://matplotlib.org/stable/tutorials/colors/colormaps.html
plt.figure()
plt.scatter(df2.Longitude, df2.Latitude, c=df2.PM25,
            vmin = 0, vmax = 50,
            cmap = 'bwr', alpha = 0.9, s = 3.0) # alpha: 투명도, s: size
plt.show()


# with colorbar

plt.figure()
points = plt.scatter(df2.Longitude, df2.Latitude, c=df2.PM25,
            vmin = 0, vmax = 50,
            cmap = 'bwr', alpha = 0.9, s = 3.0)
cb = plt.colorbar(points, orientation='vertical', # horizontal or vertical
                  ticklocation='auto', shrink=0.5, pad=0.1)
plt.show()


# Last: mapping
import cartopy
# Map
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

plt.show()


# Map + data
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

points = plt.scatter(df2.Longitude, df2.Latitude, c=df2.PM25,
            vmin = 0, vmax = 50,
            cmap = 'Reds', alpha = 0.9, s = 10.0)
cb = plt.colorbar(points, orientation='vertical', # horizontal or vertical
                  ticklocation='auto', shrink=0.5, pad=0.1)
plt.show()



# Matplotlib cheatsheets: https://github.com/matplotlib/cheatsheets#cheatsheets
