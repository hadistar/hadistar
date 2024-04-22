import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import datetime

df = pd.read_excel("D:\\OneDrive\\1.Paper_research\\PM_modeling_최은화박사님\\2024_02_AAQR\\analysis\\Ulsan input PM2.5.xlsx")

df.Date = pd.to_datetime(df.Date)

df['Date_adjusted'] = df.Date + pd.DateOffset(hours=-10)

day_mean = df[['Date', 'PM2.5']].groupby(pd.Grouper(freq='D', key='Date')).mean().dropna()
day_adjusted_mean = df[['Date_adjusted', 'PM2.5']].groupby(pd.Grouper(freq='D', key='Date_adjusted')).mean().dropna()

plt.figure(figsize=(10,6))
plt.plot(day_mean.index, day_mean['PM2.5'], label='00:00~23:00')
plt.plot(day_adjusted_mean.index, day_adjusted_mean['PM2.5'], '--' ,label='10:00~09:00 (next day)')
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.boxplot([np.array(day_mean['PM2.5']), np.array(day_adjusted_mean['PM2.5'])])
plt.show()

day_mean.describe()
day_adjusted_mean.describe()