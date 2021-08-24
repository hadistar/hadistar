import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams.update({'figure.autolayout': True})


# Station code: 131231 = Siheung, Jeongwang-dong

df_AirKorea = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea\\PM25_mean_day_Korea.csv')
df_SH_AirKorea = df_AirKorea.loc[df_AirKorea['Station code']==131231].copy()
df_SH_AirKorea['date'] = pd.to_datetime(df_SH_AirKorea['date'])

df_SH_2021 = pd.read_csv('D:\\OneDrive - SNU\\data\\AirKorea\\AirKorea_SH_131231_daily_2021.csv')
df_SH_2021.date = pd.to_datetime(df_SH_2021.date)
df_SH_2021['PM25_mean'] = df_SH_2021['PM25']

df_SH = df_SH_AirKorea.append(df_SH_2021)


df_filter = pd.read_excel('D:\\OneDrive - SNU\\data\\PM25_Siheung_speciation_total_raw.xlsx')
df_filter.date = pd.to_datetime(df_filter.date)
df_joined = pd.merge(df_filter, df_SH, how='inner', on='date')



df_joined = df_joined[['date', 'PM25_filter', 'PM25_mean']]
df_joined = df_joined.dropna()

date = df_joined.date
x = df_joined['PM25_filter']
y = df_joined['PM25_mean']

# Time-series plot

plt.figure(figsize=(8,5))
plt.plot(date,y,'k-', label='AirKorea', linewidth=1.2)
plt.plot(date, x, 'b--', label='Sampled filter', linewidth=1.2)

plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
#plt.ylim([0,110])
#plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

# 1:1 plot

import numpy as np
from sklearn import linear_model
import sklearn

x = np.array(x)
y = np.array(y)

# Create linear regression object
linreg = linear_model.LinearRegression()
# Fit the linear regression model
model = linreg.fit(x.reshape(-1,1), y.reshape(-1,1))

# Get the intercept and coefficients
intercept = model.intercept_
coef = model.coef_
result = [intercept, coef]
predicted_y = x.reshape(-1, 1) * coef + intercept
r_squared = sklearn.metrics.r2_score(y, predicted_y)

plt.figure(figsize=(5,5))
plt.plot(x, y, 'ro', markersize=8, mfc='none')

plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0,110],[0,110], 'k--')
plt.xlabel('Concentration of sampled filter ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.ylabel('Concentration of AirKorea ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.text(x.max() * 0.1, y.max() * 0.85,
         'y = %0.2fx + %0.2f\n$r^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))
plt.axis([0, 110,0,110])
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.show()


# Better version

df = pd.read_csv('data\\SH_sampler_vs.AirKorea_210719.csv')





plt.figure(figsize=(8,5))
plt.plot(df_filter['date'], df_filter['Cl-'], label='Cl-')
plt.plot(df_filter['date'], df_filter['NO3-'], label='NO3-')
plt.plot(df_filter['date'], df_filter['SO42-'], label='SO42-')
plt.plot(df_filter['date'], df_filter['Na+'], label='Na+')
plt.plot(df_filter['date'], df_filter['NH4+'], label='NH4+')

plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.show()





plt.figure(figsize=(8,5))
plt.plot(df_filter['date'], df_filter['OC'], label='OC')
plt.plot(df_filter['date'], df_filter['EC'], label='EC')

plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.show()


elec = pd.read_csv('data\\HOME_전력시장_전력거래량_연료원별.csv', encoding='euc-kr')
elec.rename(columns = {'기간' : 'date'}, inplace = True)
elec.rename(columns = {'유연탄' : 'coal'}, inplace = True)

# datetime foramt: Jul-21 => '%b-%y
elec.date = pd.to_datetime(elec.date, format='%b-%y')



plt.figure(figsize=(8,5))
plt.plot(df_filter['date'], df_filter['As'], label='As')
plt.plot(df_filter['date'], df_filter['Cr'], label='Cr')

plt.ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
plt.xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.show()


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(df_filter['date'], df_filter['As'], label='As')
ax1.plot(df_filter['date'], df_filter['Cr'], label='Cr')

ax2.plot(elec.loc[elec['지역']=='충남'].date[:21], elec.loc[elec['지역']=='충남']['coal'][:21], 'ro')


ax1.set_ylabel('Concentration ('+ "${\mu}$" +'g/m' + r'$^3$' + ')')
ax1.set_xlabel('Date (year-month)')
plt.legend()
plt.grid(True, linestyle='--')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(16,4))
plt.bar(elec.loc[elec['지역']=='소계'].date[:36], elec.loc[elec['지역']=='소계']['coal'][:36],
        width=5.5)
plt.show()
