import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('210607recal_ConstrainedErrorEstimationSummary_v1.csv')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

# Value들 PM2.5로 나눠야함..
# y축 색 빨간색으로,

fig, ax1 = plt.subplots()

ax1.bar(df.Species, df.loc[:,'Constrained Base Value_Salts'])
ax1.bar(df.Species, df.loc[:,'DISP Average_Salts'], yerr=df.loc[:,'DISP Average_Salts'], capsize=10)
ax1.plot(df.Species, df.loc[:,'DISP Average_Salts'], 'yerr=df.loc[:,'DISP Average_Salts'], capsize=10')
ax1.set_yscale('log')


ax2 = ax1.twinx() # Create another axes that shares the same x-axis as ax.
ax2.plot(df.Species, df.loc[:,'EV_Salts'], 'ro')#, width=0.3)
ax2.set_ylim([0,100])

fig.autofmt_xdate(rotation=45)

plt.show()