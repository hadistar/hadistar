import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import numpy as np

#plt.rcParams['font.family'] = 'Arial'

plt.rc('font', family='Malgun Gothic')
plt.rcParams['font.size'] = 10
rcParams.update({'figure.autolayout': True})


df = pd.read_csv('data/SH_PMF_meteo_daily_v8.csv')
df = pd.read_csv('D:\\OneDrive - SNU\\바탕 화면\\Smartcity_Sampledata\\샘플데이터_오염원유형추적결과.csv', encoding='euc-kr')

data = df.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12]]
data = data.iloc[:,[0,1,10,4,11,9,6,5,8,7,2,3]]

data.columns = ["date","PM25","Secondary nitrate", "Secondary sulfate", "Traffic","Heating", "Biomass burning", "Coal combustion", "Industry (oil)", "Industry (smelting)", "Sea salts", "Soil"]

labels = ["Secondary nitrate", "Secondary sulfate", "Traffic", "Heating", "Biomass burning",
          "Coal combustion", "Industry (oil)", "Industry (smelting)", "Sea salts", "Soil"]
labels = ["이차질산염", "이차황산염", "자동차", "난방연소", "생물체연소",
          "석탄연소", "산업(중유)", "산업(제련)", "해염입자", "토양"]

colors = ['C9','C8','C7','C6','C5','C4','C3','C2','C0']

data = df

for row in range(data.shape[0]):

    data_day = data.iloc[row,:]


    plt.figure(figsize = (8,4))
    plt.title("<"+str(data_day[0])+">"+"\n\n"+"일평균 PM$_{2.5}$: "+str(round(data_day[1],2))+" ${\mu}$g/m$^3$")
    distance = 0.0
    plt.pie(data_day[2:],
            colors=['C9','C8','C7','C6','C5','C4','C3','C2','C1','C0'],
            explode=(distance,distance,distance,distance,distance,distance,distance,distance,distance,distance),
            pctdistance=0.8, labeldistance=1.0,
            wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 4}
            )
    total = sum(data_day[2:])
    plt.legend(
        loc='upper right',
        labels=['%s, %1.1f%%, %1.2f ${\mu}$g/m$^3$' % (
            l, (float(s) / total) * 100, s) for l, s in zip(labels, data_day[2:])],
        prop={'size': 10},
        bbox_to_anchor=(1.8, 1.0),
    )

    plt.savefig('Smartcity_PieChart_'+data_day[0]+'.png')
    plt.close()


labels = data.columns[2:]

for row in range(data.shape[0]):

    data_day = data.iloc[row,:]


    plt.figure(figsize = (8,4))
    plt.title("<"+str(data_day[0])+">"+"\n\n"+"일평균 PM$_{2.5}$: "+str(round(data_day[1],2))+" ${\mu}$g/m$^3$")
    distance = 0.0
    plt.pie(data_day[2:],
            colors=['C9','C8','C7','C6','C5','C4','C3','C2'],
            explode=(distance,distance,distance,distance,distance,distance,distance,distance),
            pctdistance=0.8, labeldistance=1.0,
            wedgeprops={'width': 0.7, 'edgecolor': 'w', 'linewidth': 4}
            )
    total = sum(data_day[2:])
    plt.legend(
        loc='upper right',
        labels=['%s, %1.1f%%, %1.2f ${\mu}$g/m$^3$' % (
            l, (float(s) / total) * 100, s) for l, s in zip(labels, data_day[2:])],
        prop={'size': 10},
        bbox_to_anchor=(1.8, 1.0),
    )

    plt.savefig('Smartcity_PieChart_'+data_day[0]+'.png')
    plt.close()
