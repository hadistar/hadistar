import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Ansan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='경기권')
SH = pd.read_excel("data\\PM25_Siheung_speciation_total_raw.xlsx", sheet_name='Conc')

Seoul.date = pd.to_datetime(Seoul.date)
Ansan.date = pd.to_datetime(Ansan.date)
SH.data = pd.to_datetime(SH.date)

Seoul = Seoul.groupby(pd.Grouper(freq='D', key='date')).mean()
Ansan = Ansan.groupby(pd.Grouper(freq='D', key='date')).mean()

Seoul=Seoul.reset_index()
Ansan=Ansan.reset_index()

SH['StationNo'] = 1
SH = SH.rename(columns={'PM25_filter':'PM2.5'})

Ansan['StationNo'] = 2
Seoul['StationNo'] = 3

df = pd.DataFrame()

for i in range(SH.shape[0]):
    df = df.append(SH.iloc[i][:])
    df = df.append(Seoul.loc[Seoul.date==SH.date[i]])
    df = df.append(Ansan.loc[Ansan.date==SH.date[i]])

df.columns

df2 = df[['date','StationNo','PM2.5','NO3-','SO42-','Cl-','Na+','NH4+','K+',
          'Mg2+','Ca2+','OC','EC','Ca','Ti','V','Cr','Mn','Fe','Ni',
          'Cu','Zn','As','Pb']]

df2 = df2.reset_index(drop=True)
df2 = df2.iloc[18:369]
df2 = df2.reset_index(drop=True)

df2 = df2.sort_values(by=['date','StationNo'])

locations = pd.read_csv('data\\Smartcity_Bayesian_Locations.csv')
df2 = pd.merge(df2, locations, how='inner', on='StationNo')
df2 = df2.sort_values(by=['date','StationNo'])
df2 = df2.reset_index(drop=True)

del df2['Location']
del df2['Adress']

df2.to_csv('Smartcity_BSMRM_3Locations.csv', index=False)


import matplotlib.pyplot as plt

plt.figure()

plt.plot(df2.date[::3],df2.As[::3], label='Siheung')
plt.plot(df2.date[1::3],df2.As[1::3], label='Ansan')
plt.plot(df2.date[2::3],df2.As[2::3], label='Seoul')
plt.legend()

plt.show()

# KNN처리하기

import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

df = pd.read_csv('Smartcity_BSMRM_3Locations.csv')

result = df.copy()

imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(df.iloc[:,1:].to_numpy())

df.iloc[:,1:] = Y

df.to_csv('Smartcity_BSMRM_3Locations_KNN.csv')