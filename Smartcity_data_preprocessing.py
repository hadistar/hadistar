import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Ansan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='경기권')
Daejeon = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='중부권')
BR = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='백령도')

SH = pd.read_excel("data\\PM25_Siheung_speciation_total_raw_after_QAQC.xlsx", sheet_name='Conc')

Seoul.date = pd.to_datetime(Seoul.date)
Ansan.date = pd.to_datetime(Ansan.date)
SH.data = pd.to_datetime(SH.date)
Daejeon.date = pd.to_datetime(Daejeon.date)
BR.date = pd.to_datetime(BR.date)


Seoul = Seoul.groupby(pd.Grouper(freq='D', key='date')).mean()
Ansan = Ansan.groupby(pd.Grouper(freq='D', key='date')).mean()
Daejeon = Daejeon.groupby(pd.Grouper(freq='D', key='date')).mean()
BR = BR.groupby(pd.Grouper(freq='D', key='date')).mean()


Seoul=Seoul.reset_index()
Ansan=Ansan.reset_index()
Daejeon = Daejeon.reset_index()
BR = BR.reset_index()

SH['StationNo'] = 1
SH = SH.rename(columns={'PM25_filter':'PM2.5'})

Ansan['StationNo'] = 2
Seoul['StationNo'] = 3

Daejeon['StationNo'] = 4
BR['StationNo'] = 5


df = pd.DataFrame()

for i in range(SH.shape[0]):
    df = df.append(SH.iloc[i][:])
    df = df.append(Seoul.loc[Seoul.date==SH.date[i]])
    df = df.append(Ansan.loc[Ansan.date==SH.date[i]])
    df = df.append(Daejeon.loc[Daejeon.date==SH.date[i]])
    df = df.append(BR.loc[BR.date==SH.date[i]])

df.columns

df2 = df[['date','StationNo','PM2.5','NO3-','SO42-','Cl-','Na+','NH4+','K+',
          'Mg2+','Ca2+','OC','EC','Ca','Ti','V','Cr','Mn','Fe','Ni',
          'Cu','Zn','As','Pb']]

df2 = df2.reset_index(drop=True)
df2 = df2.iloc[36:551]
df2 = df2.reset_index(drop=True)

df2 = df2.sort_values(by=['date','StationNo'])

locations = pd.read_csv('data\\Smartcity_Bayesian_Locations.csv')
df2 = pd.merge(df2, locations, how='inner', on='StationNo')
df2 = df2.sort_values(by=['date','StationNo'])
df2 = df2.reset_index(drop=True)

del df2['Location']
del df2['Adress']

df2.to_csv('Smartcity_BSMRM_5Locations.csv', index=False)


import matplotlib.pyplot as plt

plt.figure()

plt.plot(df2.date[::3],df2.As[::3], label='Siheung')
plt.plot(df2.date[1::3],df2.As[1::3], label='Ansan')
plt.plot(df2.date[2::3],df2.As[2::3], label='Seoul')
plt.plot(df2.date[3::3],df2.As[3::3], label='Daejeon')
plt.plot(df2.date[4::3],df2.As[4::3], label='BR')
plt.legend()

plt.show()

# KNN처리하기

import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

df = pd.read_csv('Smartcity_BSMRM_5Locations.csv')

result = df.copy()

imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(df.iloc[:,1:].to_numpy())

df.iloc[:,1:] = Y

df.to_csv('Smartcity_BSMRM_5Locations_KNN.csv')