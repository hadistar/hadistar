import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Ansan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='경기권')
Daejeon = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='중부권')
BR = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='백령도')

Jeju = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='제주도')
Gwangju = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='호남권')
Ulsan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='영남권')

SH = pd.read_excel("data\\PM25_Siheung_speciation_total_raw_after_QAQC.xlsx", sheet_name='Conc')

Seoul.date = pd.to_datetime(Seoul.date)
Ansan.date = pd.to_datetime(Ansan.date)
Daejeon.date = pd.to_datetime(Daejeon.date)
BR.date = pd.to_datetime(BR.date)

SH.data = pd.to_datetime(SH.date)

Jeju.date = pd.to_datetime(Jeju.date)
Gwangju.date = pd.to_datetime(Gwangju.date)
Ulsan.date = pd.to_datetime(Ulsan.date)


Seoul_ = Seoul.loc[(Seoul.date >= '2016-01-01') & (Seoul.date <= '2020-12-31')]
Ansan_ = Ansan.loc[(Ansan.date >= '2016-01-01') & (Ansan.date <= '2020-12-31')]
Daejeon_ = Daejeon.loc[(Daejeon.date >= '2016-01-01') & (Daejeon.date <= '2020-12-31')]
BR_ = BR.loc[(BR.date >= '2016-01-01') & (BR.date <= '2020-12-31')]
Jeju_ = Jeju.loc[(Jeju.date >= '2016-01-01') & (Jeju.date <= '2020-12-31')]
Gwangju_ = Gwangju.loc[(Gwangju.date >= '2016-01-01') & (Gwangju.date <= '2020-12-31')]
Ulsan_ = Ulsan.loc[(Ulsan.date >= '2016-01-01') & (Ulsan.date <= '2020-12-31')]

Na_ratio = pd.DataFrame()
Na_ratio['Seoul'] = Seoul_.isnull().sum(axis=0)/43824*100
Na_ratio['Ansan'] = Ansan_.isnull().sum(axis=0)/43824*100
Na_ratio['Daejeon'] = Daejeon_.isnull().sum(axis=0)/43824*100
Na_ratio['BR'] = BR_.isnull().sum(axis=0)/43824*100
Na_ratio['Jeju'] = Jeju_.isnull().sum(axis=0)/43824*100
Na_ratio['Gwangju'] = Gwangju_.isnull().sum(axis=0)/43824*100
Na_ratio['Ulsan'] = Ulsan_.isnull().sum(axis=0)/43824*100

Na_ratio.to_csv('Nan_ratio_from_2016_to_2020_hourly.csv')


Seoul = Seoul.groupby(pd.Grouper(freq='D', key='date')).mean()
Ansan = Ansan.groupby(pd.Grouper(freq='D', key='date')).mean()
Daejeon = Daejeon.groupby(pd.Grouper(freq='D', key='date')).mean()
BR = BR.groupby(pd.Grouper(freq='D', key='date')).mean()

Jeju = Jeju.groupby(pd.Grouper(freq='D', key='date')).mean()
Gwangju = Gwangju.groupby(pd.Grouper(freq='D', key='date')).mean()
Ulsan = Ulsan.groupby(pd.Grouper(freq='D', key='date')).mean()

Seoul=Seoul.reset_index()
Ansan=Ansan.reset_index()
Daejeon = Daejeon.reset_index()
BR = BR.reset_index()

Jeju = Jeju.reset_index()
Gwangju = Gwangju.reset_index()
Ulsan = Ulsan.reset_index()


SH['StationNo'] = 1

Ansan['StationNo'] = 2
Seoul['StationNo'] = 3

Daejeon['StationNo'] = 4
BR['StationNo'] = 5

SH['StationNo'] = 1
SH = SH.rename(columns={'PM25_filter':'PM2.5'})

Seoul['StationNo'] = 2
Ansan['StationNo'] = 3
Daejeon['StationNo'] = 4
BR['StationNo'] = 5
Gwangju['StationNo'] = 6
Ulsan['StationNo'] = 7
Jeju['StationNo'] = 8

df = pd.DataFrame()

for i in range(Seoul.shape[0]):
    df = df.append(SH.loc[SH.date==SH.date[i]])
    df = df.append(Seoul.loc[Seoul.date==SH.date[i]])
    df = df.append(Ansan.loc[Ansan.date==SH.date[i]])
    df = df.append(Daejeon.loc[Daejeon.date==SH.date[i]])
    df = df.append(BR.loc[BR.date==SH.date[i]])


    df = df.append(Gwangju.loc[Gwangju.date==SH.date[i]])
    df = df.append(Ulsan.loc[Ulsan.date==SH.date[i]])
    df = df.append(Jeju.loc[Jeju.date==SH.date[i]])

df.columns

df2 = df[['date','StationNo','PM2.5','NO3-','SO42-','Cl-','Na+','NH4+','K+',
          'Mg2+','Ca2+','OC','EC','Ca','Ti','V','Cr','Mn','Fe','Ni',
          'Cu','Zn','As','Pb']]

df2 = df2.reset_index(drop=True)
df2 = df2.iloc[63:887]
df2 = df2.reset_index(drop=True)

df2 = df2.sort_values(by=['date','StationNo'])

locations = pd.read_csv('data\\Smartcity_Bayesian_Locations.csv')

locations = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\대기환경공학회 공모전\\pm25speciation_locations_KoreaNational.csv')


df2 = pd.merge(df2, locations, how='inner', on='StationNo')
df2 = df2.sort_values(by=['date','StationNo'])
df2 = df2.reset_index(drop=True)

del df2['Location']
del df2['Adress']


df2.to_csv('Smartcity_BSMRM_8Locations.csv', index=False)

df2 = df2.iloc[17532:20094]
df2.to_csv('대기공모전_BSMRM_7Locations_2020.csv', index=False)

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

df = pd.read_csv('Smartcity_BSMRM_8Locations.csv')

result = df.copy()

imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(df.iloc[:,2:-2].to_numpy())

df.iloc[:,2:-2] = Y

df.to_csv('Smartcity_BSMRM_8Locations_KNN.csv', index=False)


temp = df2.iloc[:,2:-3].copy()

imputer = KNNImputer(n_neighbors=3) #KNN
Y= imputer.fit_transform(temp.to_numpy())

results = df2.copy()
results.iloc[:,2:-3] = Y
results.to_csv('대기공모전_BSMRM_7Locations_kNN_2020.csv', index=False)


##
## 2021-10-18, Monday, 전국 BSMRM용 7개 locations, 1년치, 3년치 자료 만들기
##

import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Ansan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='경기권')
Daejeon = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='중부권')
BR = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='백령도')

Jeju = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='제주도')
Gwangju = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='호남권')
Ulsan = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='영남권')

Seoul.date = pd.to_datetime(Seoul.date)
Ansan.date = pd.to_datetime(Ansan.date)
Daejeon.date = pd.to_datetime(Daejeon.date)
BR.date = pd.to_datetime(BR.date)
Jeju.date = pd.to_datetime(Jeju.date)
Gwangju.date = pd.to_datetime(Gwangju.date)
Ulsan.date = pd.to_datetime(Ulsan.date)


Seoul_ = Seoul.loc[(Seoul.date >= '2018-01-01') & (Seoul.date <= '2020-12-31')]
Ansan_ = Ansan.loc[(Ansan.date >= '2018-01-01') & (Ansan.date <= '2020-12-31')]
Daejeon_ = Daejeon.loc[(Daejeon.date >= '2018-01-01') & (Daejeon.date <= '2020-12-31')]
BR_ = BR.loc[(BR.date >= '2018-01-01') & (BR.date <= '2020-12-31')]
Jeju_ = Jeju.loc[(Jeju.date >= '2018-01-01') & (Jeju.date <= '2020-12-31')]
Gwangju_ = Gwangju.loc[(Gwangju.date >= '2018-01-01') & (Gwangju.date <= '2020-12-31')]
Ulsan_ = Ulsan.loc[(Ulsan.date >= '2018-01-01') & (Ulsan.date <= '2020-12-31')]

Seoul = Seoul_.groupby(pd.Grouper(freq='D', key='date')).mean()
Ansan = Ansan_.groupby(pd.Grouper(freq='D', key='date')).mean()
Daejeon = Daejeon_.groupby(pd.Grouper(freq='D', key='date')).mean()
BR = BR_.groupby(pd.Grouper(freq='D', key='date')).mean()
Jeju = Jeju_.groupby(pd.Grouper(freq='D', key='date')).mean()
Gwangju = Gwangju_.groupby(pd.Grouper(freq='D', key='date')).mean()
Ulsan = Ulsan_.groupby(pd.Grouper(freq='D', key='date')).mean()

Seoul=Seoul.reset_index()
Ansan=Ansan.reset_index()
Daejeon = Daejeon.reset_index()
BR = BR.reset_index()
Jeju = Jeju.reset_index()
Gwangju = Gwangju.reset_index()
Ulsan = Ulsan.reset_index()

Seoul['StationNo'] = 2
Ansan['StationNo'] = 3
Daejeon['StationNo'] = 4
BR['StationNo'] = 5
Gwangju['StationNo'] = 6
Ulsan['StationNo'] = 7
Jeju['StationNo'] = 8


# KNN처리하기

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=3) #KNN
Seoul.iloc[:,1:]= imputer.fit_transform(Seoul.iloc[:,1:].to_numpy())
Ansan.iloc[:,1:]= imputer.fit_transform(Ansan.iloc[:,1:].to_numpy())
Daejeon.iloc[:,1:]= imputer.fit_transform(Daejeon.iloc[:,1:].to_numpy())
BR.iloc[:,1:]= imputer.fit_transform(BR.iloc[:,1:].to_numpy())
Gwangju.iloc[:,1:]= imputer.fit_transform(Gwangju.iloc[:,1:].to_numpy())
Ulsan.iloc[:,1:]= imputer.fit_transform(Ulsan.iloc[:,1:].to_numpy())
Jeju.iloc[:,1:]= imputer.fit_transform(Jeju.iloc[:,1:].to_numpy())

# N7 1년치

df = pd.DataFrame()

for i in range(Ansan.shape[0]):
    df = df.append(Seoul.loc[Seoul.date==Ansan.date[i]])
    df = df.append(Ansan.loc[Ansan.date==Ansan.date[i]])
    df = df.append(Daejeon.loc[Daejeon.date==Ansan.date[i]])
    df = df.append(BR.loc[BR.date==Ansan.date[i]])
    df = df.append(Gwangju.loc[Gwangju.date==Ansan.date[i]])
    df = df.append(Ulsan.loc[Ulsan.date==Ansan.date[i]])
    df = df.append(Jeju.loc[Jeju.date==Ansan.date[i]])

df.columns

df2 = df[['date','StationNo','PM2.5','NO3-','SO42-','Cl-','Na+','NH4+','K+',
          'Mg2+','Ca2+','OC','EC','Ca','Ti','V','Cr','Mn','Fe','Ni',
          'Cu','Zn','As','Pb']]

df2 = df2.sort_values(by=['date','StationNo'])

locations = pd.read_csv('data\\Smartcity_Bayesian_Locations.csv')

df2 = pd.merge(df2, locations, how='inner', on='StationNo')
df2 = df2.sort_values(by=['date','StationNo'])
df2 = df2.reset_index(drop=True)

del df2['Location']
del df2['Adress']

df2.to_csv('Smartcity_BSMRM_Korea_N7_1years.csv', index=False)



# N6 3년치

df = pd.DataFrame()

for i in range(Seoul.shape[0]):
    df = df.append(Seoul.loc[Seoul.date==Seoul.date[i]])
    df = df.append(Daejeon.loc[Daejeon.date==Seoul.date[i]])
    df = df.append(BR.loc[BR.date==Seoul.date[i]])
    df = df.append(Gwangju.loc[Gwangju.date==Seoul.date[i]])
    df = df.append(Ulsan.loc[Ulsan.date==Seoul.date[i]])
    df = df.append(Jeju.loc[Jeju.date==Seoul.date[i]])

df.columns

df2 = df[['date','StationNo','PM2.5','NO3-','SO42-','Cl-','Na+','NH4+','K+',
          'Mg2+','Ca2+','OC','EC','Ca','Ti','V','Cr','Mn','Fe','Ni',
          'Cu','Zn','As','Pb']]

df2 = df2.sort_values(by=['date','StationNo'])

locations = pd.read_csv('data\\Smartcity_Bayesian_Locations.csv')

df2 = pd.merge(df2, locations, how='inner', on='StationNo')
df2 = df2.sort_values(by=['date','StationNo'])
df2 = df2.reset_index(drop=True)

del df2['Location']
del df2['Adress']

df2.to_csv('Smartcity_BSMRM_Korea_N6_3years.csv', index=False)