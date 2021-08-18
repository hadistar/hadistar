
# 1. pie_chart.py

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
rcParams.update({'figure.autolayout': True})
plt.figure()
# ratio = [1,0, 16, 49, 3, 5, 20,
#          2, 4, 16]
#ratio = [4, 16, 16,  2, 3,  49, 20, 5, 1]

ratio = [29.3, 1.14, 107, 14.4, 17.5, 328, 137, 30.8, 5.51]

labels = ["Secondary nitrate", "Secondary sulrfate", "Traffic", "Heating", "Biomass burning", "Coal combustion", "Industry (oil)",
                     "Industry (smelting)", "Sea salts"]
colors = ['C9','C8','C7','C6','C5','C4','C3','C2','C1']

# labels = ["Salts", "Soil", "SS", "Coal combustion", "Biomass burning", "Industry Smelting", "Industry Oil",
#                      "Heating", "SN", "Mobile"]
# colors = ['C1','C0','C8','C4','C5','C2','C3','C6','C9','C7']
plt.pie(ratio, autopct='%.1f%%',colors=colors,textprops={'fontsize': 10},
        explode=(0, 0.1 ,0,0.1,0,0, 0,0,0.1))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, labels=labels, borderaxespad=0.)
plt.savefig('Results\\pie_source.png')
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rcParams
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
rcParams.update({'figure.autolayout': True})
plt.figure()
ratio = [24.31, 18.75, 18.83, 12.55, 11.76, 3.63, 1.79, 4.03, 2.68, 1.66]
labels = ["Secondary nitrate", "Secondary sulfate", "Traffic", "Heating", "Biomass burning", "Coal combustion", "Industry (oil)",
                     "Industry (smelting)", "Sea salts", "Soil"]
colors = ['C9','C8','C7','C6','C5','C4','C3','C2','C1','C0']
plt.pie(ratio, autopct='%.1f%%',colors=colors,textprops={'fontsize': 10},explode=(0,0,0,0,0,0,
                                                                                  0,0,0,0))
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, labels=labels, borderaxespad=0.)
plt.savefig('Results\\pie_source.png')
plt.show()


# 2. py_stack_var_sources.py

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd

# data import
df = pd.read_csv('rdata.csv').dropna()
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df["Season2"] = "Annual"

# assign font family
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

print(df.columns)
# for ILCR
df_ilcr = df.iloc[:, [21, 36, 51, 66, 81, 96, 111, 126, 141, 156, 168, 169]]

# for HQ
#df_ilcr = df.iloc[:, [31, 46, 61, 76, 91, 106, 121, 136, 151, 166, 168, 169]]
df_ilcr.head(10)
df_ilcr.columns = ["sa", "so", "ss", "cc", "bb", "ism", "io",
                    "h", "sn", "m", "Season","Season2"]

# df_ilcr.columns = ["Salts", "Soil", "SS", "Coal combustion", "Biomass burning", "Industry Smelting", "Industry Oil",
#                    "Heating", "SN", "Mobile", "Season","Season2"]

# Calculate mean for each season
x= df_ilcr[['Season',"sa", "so", "ss", "cc", "bb", "ism", "io",
                    "h", "sn", "m"]]
y= x.set_index('Season')
z = y.groupby(['Season']).mean().reset_index()

# Calculate mean for annual
x2 = df_ilcr[['Season2',"sa", "so", "ss", "cc", "bb", "ism", "io",
                    "h", "sn", "m"]]
y2 = x2.groupby(['Season2']).mean().reset_index()
y2 = y2.rename(columns={'Season2':'Season'})

z = pd.concat([z,y2], ignore_index=True)
# z.to_csv("ILCR_source_mean.csv")

# Order Season names in the table created by groupby
Season = ['Autumn', 'Winter','Spring', 'Summer', 'Annual']
mapping = {Season: i for i, Season in enumerate(Season)}
key = z['Season'].map(mapping)
z = z.iloc[key.argsort()]
# z = z.sort_index(axis=1)
z.iloc[:,1:11] =z.iloc[:,1:11]*10e5

# Draw the bar chart
# ["Salts", "Soil", "SS", "Coal combustion", "Biomass burning", "Industry Smelting", "Industry Oil",
#                     "Heating", "SN", "Mobile", "Season","Season2"]
# ['Season',"sa", "so", "ss", "cc", "bb", "is", "io",
#                     "h", "sn", "Mm"]]
plt.figure()
plt.grid(True, linestyle="--")
plt.rcParams['axes.axisbelow'] = True
plt.bar(Season, z.sn, color='C9',width=0.6,label="Secondary nitrate",bottom=z.ss+z.m+z.h+z.bb+z.cc+z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.ss, color='C8',width=0.6,label="Secondary sulfate",bottom=z.m+z.h+z.bb+z.cc+z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.m, color='C7',width=0.6,label="Traffic",bottom=z.h+z.bb+z.cc+z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.h, color='C6',width=0.6,label="Heating",bottom=z.bb+z.cc+z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.bb, color='C5',width=0.6,label="Biomass burning",bottom=z.cc+z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.cc, color='C4',width=0.6,label="Coal combustion",bottom=z.io+z.ism+z.sa+z.so)
plt.bar(Season, z.io, color='C3',width=0.6,label="Industry (oil)",bottom=z.ism+z.sa+z.so)
plt.bar(Season, z.ism, color='C2',width=0.6,label="Industry (smelting)",bottom=z.sa+z.so)
plt.bar(Season, z.sa, color='C1',width=0.6,label="Sea Salts",bottom=z.so)
plt.bar(Season, z.so, color='C0',width=0.6,label="Soil")
plt.ylabel('Incremental Lifetime Cancer Risk ('+ "x 10" + '$^{-6}$'+')',fontsize=15)
#plt.ylabel('Hazard Quotient',fontsize=15)
plt.xticks(Season,rotation=40, size=10)
#plt.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=10)
plt.savefig('Results\\ILCR_source.png')
#plt.savefig('Results\\HQ_source.png')
plt.show()


# 3. stack_bar_health.py

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
import pandas as pd
# import numpy as np

# data import
df = pd.read_csv('rdata.csv').dropna()
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df["Season2"] = "Annual"

# assign font family
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

print(df.columns)
df_ilcr = df.iloc[:, [2, 3, 4, 5, 168,169]]

# For HQ
#df_ilcr = df.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 168, 169]]
df_ilcr.head(10)

df_ilcr.columns = ["As","Cr6","Ni","Pb","Season","Season2"]

#df_ilcr.columns = ["As","Cr6","Cr3","Cu","Ni","Pb","Zn","V","Mn",'Season', 'Season2']
print(df_ilcr.columns)

# Calculate mean for each season
x= df_ilcr[['Season','As','Cr6','Ni','Pb']]
#x= df_ilcr[['Season',"As","Cr6","Cr3","Cu","Ni","Pb","Zn","V","Mn"]]
y= x.set_index('Season')
z = y.groupby(['Season']).mean().reset_index()

# Calculate mean for annual
x2 = df_ilcr[['Season2','As','Cr6','Ni','Pb']]
#x2= df_ilcr[['Season2',"As","Cr6","Cr3","Cu","Ni","Pb","Zn","V","Mn"]]
y2 = x2.groupby(['Season2']).mean().reset_index()
y2 = y2.rename(columns={'Season2':'Season'})

z = pd.concat([z,y2], ignore_index=True)

# Order Season names in the table
Season = ['Autumn', 'Winter','Spring', 'Summer','Annual']
mapping = {Season: i for i, Season in enumerate(Season)}
key = z['Season'].map(mapping)
z = z.iloc[key.argsort()]
z.iloc[:,1:5] =z.iloc[:,1:5]*10e5

# Draw the bar chart
plt.figure()
plt.grid(True, linestyle="--")
plt.rcParams['axes.axisbelow'] = True
plt.bar(Season, z.Pb, color='red',width=0.6,label='Pb',bottom=z.Ni+z.Cr6+z.As)
plt.bar(Season, z.Ni, color='green',label='Ni',width=0.6,bottom=z.Cr6+z.As) # stacked bar chart
plt.bar(Season, z.Cr6, color='orange',label='Cr$^{6+}$',width=0.6,bottom=z.As) # stacked bar chart
plt.bar(Season, z.As, color='blue',label='As', width=0.6) # stacked bar chart
plt.ylabel('Incremental Lifetime Cancer Risk ('+ "x 10" + '$^{-6}$'+')',fontsize=15)
plt.xticks(Season,rotation=40, size=10)
#plt.legend(loc="upper left", bbox_to_anchor=(0.62,1), fontsize=10)
plt.legend(loc="upper center", ncol = 4, fontsize=10)

plt.savefig('Results\\ILCR_con.png')
plt.show()
#
# plt.figure()
# plt.grid(True, linestyle="--")
# plt.rcParams['axes.axisbelow'] = True
# plt.bar(Season, z.Zn, color='C9',width=0.6,label='Zn',bottom=z.V+z.Cu+z.Cr3+z.Ni+z.Pb+z.Cr6+z.Mn+z.As)
# plt.bar(Season, z.V, color='C8',label='V',width=0.6,bottom=z.Cu+z.Cr3+z.Ni+z.Pb+z.Cr6+z.Mn+z.As) # stacked bar chart
# plt.bar(Season, z.Cu, color='C7',label='Cu',width=0.6,bottom=z.Cr3+z.Ni+z.Pb+z.Cr6+z.Mn+z.As) # stacked bar chart
# plt.bar(Season, z.Cr3, color='C6',label='Cr$^{3+}$', width=0.6,bottom=z.Ni+z.Pb+z.Cr6+z.Mn+z.As) # stacked bar chart
# plt.bar(Season, z.Ni, color='C4',label='Ni', width=0.6,bottom=z.Pb+z.Cr6+z.Mn+z.As)
# plt.bar(Season, z.Pb, color='C3',label='Pb', width=0.6,bottom=z.Cr6+z.Mn+z.As)
# plt.bar(Season, z.Cr6, color='C2',label='Cr$^{6+}$', width=0.6,bottom=z.Mn+z.As)
# plt.bar(Season, z.Mn, color='C1',label='Mn', width=0.6,bottom=z.As)
# plt.bar(Season, z.As, color='C0',label='As', width=0.6)
# plt.ylabel('Hazard Quotient',fontsize=15)
# plt.xticks(Season,rotation=40, size=10)
# plt.legend(loc="upper left", bbox_to_anchor=(1,1), fontsize=10)
# plt.savefig('Results\\Hazard_Quotient.png')
# plt.show()