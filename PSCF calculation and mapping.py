
# 1. PSCF calculation

import pandas as pd
import os


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

results.to_excel('results_summary.xlsx', index=False)

print('Done!')


# 2. Mapping

import matplotlib.pyplot as plt
import cartopy
import numpy as np
import datetime
from scalebar import scale_bar

