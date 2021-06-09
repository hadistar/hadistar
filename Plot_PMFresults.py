import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

df = pd.read_csv('210607recal_ConstrainedErrorEstimationSummary_v5.csv')
df_contri = pd.read_csv('210607recal_ConstrainedErrorEstimationSummary_v5_contribution.csv').dropna()
df_contri['date'] = pd.to_datetime(df_contri['date'], format='%m-%d-%y %H:%M')

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13

Sources_name = ['Sea salts','Soil', 'Combustion for heating',
                'Coal combustion','Secondary sulfate','Industry (smelting)',
                'Secondary Nitrate','Mobile','Biomass burning','Industry (oil)']

data = pd.DataFrame()
data['Species'] = df['Species']

for i, s in enumerate(Sources_name):
    data['Conc_'+s] = df.iloc[:,4*i+1] / df.iloc[:,4*i+1][0]
    data['Min_'+s] = df.iloc[:,4*i+2] / df.iloc[:,4*i+1][0]
    data['Avg_'+s] = df.iloc[:,4*i+3] / df.iloc[:,4*i+1][0]
    data['Max_'+s] = df.iloc[:,4*i+4] / df.iloc[:,4*i+1][0]
    data['EV_'+s] = df.iloc[:,41+i]

data = data.drop(0, axis=0)


Species_name = list(data.Species)
Species_name[:6] = ['NO$_3$$^-$', 'SO$_4$$^{2-}$', 'NH$_4$$^+$', 'K$^+$', 'Na$^+$', 'Cl$^-$']

# Source profile plot

fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(12,24), sharey=True, sharex=True)

for i, s in enumerate(Sources_name):

    error = [(data.loc[:,'Avg_'+s] - data.loc[:,'Min_'+s]),
             (data.loc[:,'Max_'+s] - data.loc[:,'Avg_'+s])] # [(Bottom range), (Upper range)]

    ax1 = axes[i]
    ax2 = ax1.twinx()

    ax1.bar(Species_name, data.loc[:,'Conc_'+s], color='silver',
            yerr=error, capsize=5)
    #ax1.set_ylabel('Concentration ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)')
    ax1.set_yscale('log')
    ax1.set_ylim([1e-4,5])
    ax1.tick_params(axis='y', colors='black')
    ax1.set_yticks([1e-4,1e-3,1e-2,1e-1,1])
    ax1.grid(True, axis='x', linestyle='--')

    ax2.plot(Species_name, data.loc[:,'EV_'+s], 'ro')#, width=0.3)
    ax2.set_ylim([0,100])
    ax2.set_yticks([0,25,50,75,100])
    ax2.yaxis.label.set_color('red')

    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')

    ax1.text(22,0.3, s, size=20, ha='right')

#fig.supylabel('Concentration ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)')
fig.text(0.05, 0.5, 'Fraction of PM$_{2.5}$ ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)', ha='center',va='center',rotation='vertical', fontsize=20)
fig.text(0.95, 0.5, 'Percent of species (%)', ha='center',va='center',rotation=270, fontsize=20, color='red')

fig.autofmt_xdate(rotation=45)
plt.show()


# Contribution plot

fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(12,24), sharex=True)

for i, s in enumerate(Sources_name):
    ax1 = axes[i]
    ax1.plot(df_contri['date'], df_contri[s], 'k-')
    ax1.fill_between(df_contri['date'],0,df_contri[s], color='lightgrey')
    ax1.text(1.0,0.8,
             s+" "+str(df_contri[s].mean().round(2))+" ${\mu}$" +'g/m'+"$^3$",
             size=18, ha='right', transform=ax1.transAxes)
#    ax1.set_ylim(bottom=0)
    max = math.ceil(df_contri[s].max()) + 4 - math.ceil(df_contri[s].max()) % 4
    ax1.set_ylim([0,max])
    ax1.yaxis.set_major_locator(MaxNLocator(4))
    ax1.grid(True, axis='y', linestyle='--')

fig.text(0.05, 0.5, 'Mass Concentration ('+ "${\mu}$" +'g/m'+"$^3$"+")",
         ha='center',va='center',rotation='vertical', fontsize=20)
fig.autofmt_xdate(rotation=45)
plt.show()



#
# # For Backup
#
# fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(12,24), sharey=True, sharex=True)
# ax1 = axes[0]
# #fig, ax1 = plt.subplots(10,1, figsize=(12,4))
# ax2 = ax1.twinx()
#
# ax1.bar(Species_name, data.loc[:,'Conc_Sea salts'], color='silver',
#         yerr=error, capsize=5)
# #ax1.set_ylabel('Concentration ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)')
# ax1.set_yscale('log')
# ax1.set_ylim([1e-4,2])
# ax1.tick_params(axis='y', colors='black')
# ax1.set_yticks([1e-4,1e-3,1e-2,1e-1,1])
#
# ax2.plot(Species_name, data.loc[:,'EV_Sea salts'], 'ro')#, width=0.3)
# ax2.set_ylim([0,100])
# ax2.set_yticks([0,25,50,75,100])
# ax2.yaxis.label.set_color('red')
#
# ax2.spines['right'].set_color('red')
# ax2.tick_params(axis='y', colors='red')
#
# ax1.text(19,0.4, 'Sea salts', size=20, ha='center')
#
# #fig.supylabel('Concentration ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)')
# fig.text(0.05, 0.5, 'Concentration ('+ "${\mu}$" +'g/'+"${\mu}$" +'g)', ha='center',va='center',rotation='vertical', fontsize=20)
# fig.text(0.95, 0.5, 'Explained value', ha='center',va='center',rotation=270, fontsize=20, color='red')
#
# fig.autofmt_xdate(rotation=45)
# plt.show()
