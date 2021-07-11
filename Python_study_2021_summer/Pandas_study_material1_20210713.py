
# For pandas study
# Main contents: merge, groupby (grouper), row calculation function
# Written by Young Su Lee
# 2021-07-11
# required file: 210713_PM2.5_mean_day.csv, 210713_AirKorea_20191103.csv


import pandas as pd
import os
os.getcwd()
os.chdir('D:/hadistar/Python_study_2021_summer')

df1 = pd.read_csv('210713_PM2.5_mean_day.csv')
df2 = pd.read_csv('210713_AirKorea_20191103.csv', encoding='euc-kr')

df1.head()
df2.head()

df2 = df2.rename(columns={'측정소코드':'Station code'})
df2 = df2[['Station code', 'Latitude', 'Longitude']]

df3 = pd.merge(df1, df2, how='inner', on='Station code')

df4 = df3.loc[df3['date']=='2020-11-11'].copy()
df4 = df4.reset_index(drop=True)

# 2. Row calculation function

#PM2.5 criteria: 0< <=15: good , 15< <=35: normal, 35< <=75: bad, 75<: very bad
def cal_criteria(row):
    if row['PM25_mean'] > 75:
        return 'very bad'
    elif row['PM25_mean'] > 35:
        return 'bad'
    elif row['PM25_mean'] > 15:
        return 'normal'
    else:
        return 'good'

df4['criteria'] = df4.apply(cal_criteria, axis=1)


# 2-1. DataFrame row iteration

criteria2 = []

for i in range(df4.shape[0]):
    print(i)
    print(df4.iloc[i])
    if df4.iloc[i]['PM25_mean'] > 75:
        criteria2.append('very bad')
    elif df4.iloc[i]['PM25_mean'] > 35:
        criteria2.append('bad')
    elif df4.iloc[i]['PM25_mean'] > 15:
        criteria2.append('normal')
    else:
        criteria2.append('good')

df4['criteria2'] = criteria2

# 3. Row split

df5 = df3.copy()
df5['date'] = pd.to_datetime(df5['date'])

df6 = df5.loc[df5['Station code']==111124]
df7 = pd.DataFrame()
for n in range(len(df6)):
    print(n)
    for i in range(24):
        temp = df6.iloc[n].copy()
        temp['date'] = temp['date'] + pd.to_timedelta(i, unit='h')
        df7 = df7.append(temp, sort=False)

# 4. Groupby (grouper)

df8 = df7.groupby(pd.Grouper(freq='D', key='date')).mean() # 'D' means daily
df8 = df8.reset_index()

df9 = df8.groupby(pd.Grouper(freq='M', key='date')).mean() # 'M' means monthly
df10 = df8.groupby(pd.Grouper(freq='Y', key='date')).mean() # 'Y' means yearly

df11 = df5.groupby(['Station code', pd.Grouper(freq='M', key='date')], as_index=False).mean()

# 4-1 Groupby using for loop
station_list = df5['Station code'].unique()
df12 = pd.DataFrame()
for station in station_list:
    print (station)
    temp = df5.loc[df5['Station code']==station]
    temp = temp.groupby(pd.Grouper(freq='M', key='date')).mean()
    df12 = df12.append(temp, sort=False)

df12


