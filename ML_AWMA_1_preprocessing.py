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
df = df.reset_index(drop=True).copy()

MDLs = pd.read_excel('data\\Intensiv_Seoul_MDLs_2018-19.xlsx')
MDLs.date = pd.to_datetime(MDLs.date)

temp = df.copy()

columns = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+',
           'Ca2+', 'OC', 'EC', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni',
           'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

# MDL 이하값 비율 계산용

def calcul_mdls_ratio(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = 1
        elif row.loc[col] >= MDL.iloc[0][col]:
            temp.loc[col] = 0

    return temp

df3_MDL_bool =  df.apply(calcul_mdls_ratio, axis=1)

# 비율 체크

for species in df3_MDL_bool.columns:
    temp = df3_MDL_bool[df3_MDL_bool[species]==1].count()[species]
    print(species, round(temp/len(df)*100,2),"%")

#-------------------------------------------------------

# MDL 이하값 MDL*0.5 대체용

def calcul_mdls(row):

    MDL = MDLs.loc[(MDLs.date.dt.year == row.date.year) & (MDLs.date.dt.month == row.date.month)]
    temp = row
    for col in columns:
        if row.loc[col] < MDL.iloc[0][col]:
            temp.loc[col] = MDL.iloc[0][col]*0.5

    return temp

df2 = df.apply(calcul_mdls, axis=1)

# Missing ratio calculation

print(round(df2.isna().sum()/len(df2)*100,2),"%")

# df3: drop na values

df3 = df2.dropna()
df3.to_csv('AWMA_input_preprocessed_MDL_Na.csv', index=False)
