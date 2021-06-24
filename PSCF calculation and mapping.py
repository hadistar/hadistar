
# 1. PSCF calculation

import pandas as pd
import os

import matplotlib.pyplot as plt
import cartopy
import numpy as np
import datetime
from scalebar import scale_bar
from scipy.interpolate import griddata



def cal_wpscf(row):
    if row['No_of_Data'] > 3 * avg_endpoint:
        return row['PSCF']
    elif row['No_of_Data'] > 1.5 * avg_endpoint:
        return row['PSCF'] * 0.7
    elif row['No_of_Data'] > 1 * avg_endpoint:
        return row['PSCF'] * 0.4
    else:
        return row['PSCF'] * 0.2


results = pd.DataFrame()

# File loading
for file in os.listdir('./PSCF_txtfiles'):

    # print source name
    name = file[12:-4]

    print(name)

    data = pd.read_table('./PSCF_txtfiles/'+file)
    data = data[22:-1]

    data = data.iloc[:, 0].str.split(",", expand=True)
    data = data.rename(columns=data.iloc[0])
    data = data[1:]
    data = data.astype(float)

    # WPSCF calculation

    n = len(data)
    sum_endpoint = data['No_of_Data'].sum()
    avg_endpoint = sum_endpoint/n

    data['WPSCF'] = data.apply(cal_wpscf, axis=1)

    results[name] = data['WPSCF']

    # Data saving to excel file using pandas ExcelWriter
    # with pd.ExcelWriter('results.xlsx', mode='a') as writer:
    #     data.to_excel(writer, sheet_name=name, index=False)


results['Lon'] = data['Lon']
results['Lat'] = data['Lat']

results.to_csv('results_summary_210618.csv', index=False)

print('Done!')


# 2. Mapping
import matplotlib.colors as mcolors

# colors = [(0,0,1,c) for c in np.linspace(0,1,100)]
# cmapblue = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

for source in results.columns[:-2]:

    results = pd.read_csv('results_summary_210618.csv')

    df = pd.DataFrame()
    df[source] = results[source]
    df['Lon'] = results['Lon']
    df['Lat'] = results['Lat']

    x, y = df['Lon'], df['Lat']
    z = df[source]

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=cartopy.crs.PlateCarree())

    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
    ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
    ax.add_feature(cartopy.feature.RIVERS)

    lon1, lon2, lat1, lat2 = 115.0, 135.0, 25.0, 48.0
    ax.set_extent([lon1, lon2, lat1, lat2], crs=cartopy.crs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, crs=cartopy.crs.PlateCarree(), linestyle='--')
    gl.xlabel_style = {'size': 15}
    gl.ylabel_style = {'size': 15}

    df_z = df[source]

    xi = np.arange(lon1, lon2, 0.01)
    yi = np.arange(lat1, lat2, 0.01)
    xi,yi = np.meshgrid(xi,yi)
    zi = griddata((x,y),z,(xi,yi),method='linear') # linear, cubic, nearest

    levels = np.linspace(0,z.max(),7)
    levels = np.round_(levels,2)

    colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 10)]
    cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=7)

    mapping = plt.contourf(xi,yi,zi, levels=levels, cmap=cmapred)

    cb = plt.colorbar(mapping, orientation='vertical', ticklocation='auto', shrink=0.5, pad=0.1)

    cb.set_label(label='WPSCF', size=20)
    cb.ax.tick_params(labelsize=15)
    # plt.tight_layout()

    plt.title('[' + source + ']')
    plt.savefig(source + '_24h' + '.jpg')
    plt.show()
    plt.close()