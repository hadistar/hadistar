import pandas as pd

Seoul = pd.read_excel('data\\집중측정소_to2020_hourly.xlsx', sheet_name='수도권')
Seoul.date = pd.to_datetime(Seoul.date)

# MDL값보다 낮은 비율 계산

MDLs = {'S':0.8, 'K':0.16, 'Ca':0.02, 'Ti':0.012, 'V':0.004, 'Cr':0.002, 'Mn':0.0012, 'Fe':0.0012, 'Ni':0.0008,
       'Cu':0.0008, 'Zn':0.0004, 'As':0.0004, 'Se':0.0012, 'Br':0.0016, 'Pb':0.0012}

# 기간별 자르기

df = Seoul[Seoul.date>'2016-01-01']
df = df[:-1]


#-----------------------------------------------
# 번외: For EDA

from pandas_profiling import ProfileReport


# EDA Report 생성
profile = ProfileReport(df,
            minimal=True,
            explorative=True,
            title='Data Profiling',
            plot={'histogram': {'bins': 8}},
            pool_size=4,
            progress_bar=False)

# Report 결과 경로에 저장
profile.to_file(output_file="data_profiling.html")

#-----------------------------------------------

# Ratio calculation of values below MDLs

for species in MDLs.keys():
    temp = df[df[species]<MDLs[species]].count()[species]
    print(species, round(temp/len(df)*100,2),"%")

#-------------------------------------------------
# Histogram

import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
plt.hist(df['Cr'], bins=200, range=[0,0.004])
plt.show()

#--------------------------------------------------

# Missing ratio calculation

print(round(df.isna().sum()/len(df)*100,2),"%")

#--------------------------------------------------


# MDL 이하값 MDLs*2로 대체..

# Trace elementals

for species in MDLs.keys():
    print(species)
    df[species].loc[df[species]<MDLs[species]] = MDLs[species] * 0.5

# Ions & carbons

MDLs_2 = pd.read_excel('data\\Intensiv_Seoul_MDLs_ions_carbons_2018-19.xlsx')
MDLs_2.date = pd.to_datetime(Seoul.date)

def calcul_mdls(row):

