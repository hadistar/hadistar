import pandas as pd
import numpy as np

n_lat = 200
n_lot = 200

lat_start = 32.5
lat_end = 38.5
lon_start = 125
lon_end = 130

lat = np.linspace(lat_start,lat_end,n_lat)
lon = np.linspace(lon_start,lon_end,n_lot)

s_new = []
for y in lat:
    for x in lon:
        s_new.append([y,x])

s_new = pd.DataFrame(s_new)
s_new.columns = ['lat','lon']

s_new.to_csv('s_new.csv', index=False)