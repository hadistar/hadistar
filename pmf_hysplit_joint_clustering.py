import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Traj for Nov to Mar

traj = pd.read_csv('./data/Nov.toMar.cluster4.csv')

pmf = pd.read_csv('./data/shpmfresulthourly_Rev.csv')

traj['year'] = traj['year'] + 2000
traj['date'] = pd.to_datetime(traj[['year','month','day','hour']])

pmf['date'] = pd.to_datetime(pmf[['year','month','day','hour']])

traj = traj.drop(['year','month','day','hour'], axis=1)
pmf = pmf.drop(['one','year','month','day','hour'], axis=1)

df = pd.merge(pmf, traj, how='left', on='date')

df.to_csv('SH_data_traj_combined.csv', index=False)

cluster1 = df[df['CL# #']==1]
cluster2 = df[df['CL# #']==2]
cluster3 = df[df['CL# #']==3]
cluster4 = df[df['CL# #']==4]

sources = ['sn','ss','cc','bb','heatingc','mobile','soil','seasalts','industrysmelting','industryoil']


# Creating autocpt arguments
def func(pct, allvalues):
    absolute = float(pct / 100. * np.sum(allvalues))
    return "{:.2f}%\n({:.2f})".format(pct, absolute)


data = cluster4[sources].mean()
explode = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.75, 0.5, 0.25)

plt.figure()
plt.pie(data,
        explode=explode,
        autopct=lambda pct: func(pct, data),
        labels=sources)
plt.axis('equal')
#plt.legend(sources)
plt.title('Cluster 4, average PM2.5 = ' + str(round(data.sum(), 2)))
plt.show()


# Traj for Apr to Oct


traj2 = pd.read_csv('./data/SHAprtoOct.cl3.csv')
traj2['date'] = pd.to_datetime(traj2[['year','month','day','hour']])
traj2 = traj2.drop(['year','month','day','hour'], axis=1)

pmf = pd.read_csv('./data/shpmfresulthourly_Rev.csv')
pmf['date'] = pd.to_datetime(pmf[['year','month','day','hour']])
pmf = pmf.drop(['one','year','month','day','hour'], axis=1)
df = pd.merge(pmf, traj2, how='left', on='date')

for i in range(5,8):
    print(i)
    data = df[df['CL# #']==i][sources].mean()

    plt.figure(figsize=(9,9))
    plt.pie(data,
    #        explode=explode,
            autopct=lambda pct: func(pct, data),
            labels=sources)
    #plt.axis('equal')
    #plt.legend(sources)
    plt.title('Cluster '+str(i) +', average PM2.5 = ' + str(round(data.sum(), 2)))
    plt.show()
    plt.close()


traj3 = pd.read_csv('./data/SHAprtoOct.cl5.csv')
traj3['date'] = pd.to_datetime(traj3[['year','month','day','hour']])
traj3 = traj3.drop(['year','month','day','hour'], axis=1)

pmf = pd.read_csv('./data/shpmfresulthourly_Rev.csv')
pmf['date'] = pd.to_datetime(pmf[['year','month','day','hour']])
pmf = pmf.drop(['one','year','month','day','hour'], axis=1)
df = pd.merge(pmf, traj3, how='left', on='date')


for i in range(8,13):
    print(i)
    data = df[df['CL# #']==i][sources].mean()

    plt.figure(figsize=(9,9))
    plt.pie(data,
    #        explode=explode,
            autopct=lambda pct: func(pct, data),
            labels=sources)
    #plt.axis('equal')
    #plt.legend(sources)
    plt.title('Cluster '+str(i) +', average PM2.5 = ' + str(round(data.sum(), 2)))
    plt.show()
    plt.close()

