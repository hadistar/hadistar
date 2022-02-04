import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np

# 1. Tukey, r2값 이용

df = pd.read_excel('d:\\OneDrive - SNU\\바탕 화면\\ing\\Manuscript_AI\\results__211220.xlsx', sheet_name='results__211220')

group1 = df.loc[(df[0]==4) & (df[2]==1) & (df.model == 'GAIN')& (df.case == 'ocec-elementals')]['r2']
group2 = df.loc[(df[0]==4) & (df[2]==1) & (df.model == 'DNN')& (df.case == 'ocec-elementals')]['r2']
group3 = df.loc[(df[0]==4) & (df[2]==1) & (df.model == 'RF2')& (df.case == 'ocec-elementals')]['r2']
group4 = df.loc[(df[0]==4) & (df[2]==1) & (df.model == 'KNN')& (df.case == 'ocec-elementals')]['r2']

# 예시 데이터 시각화 하기
plot_data = [group1, group2, group3, group4]
ax = plt.boxplot(plot_data)
plt.show()

# data combine
group1 = pd.DataFrame(group1)
group1['type'] = 'GAIN'
group2 = pd.DataFrame(group2)
group2['type'] = 'DNN'
group3 = pd.DataFrame(group3)
group3['type'] = 'RF'
group4 = pd.DataFrame(group4)
group4['type'] = 'KNN'

df2 = group1.append(group2).append(group3).append(group4)

# Tukey test
from statsmodels.stats.multicomp import pairwise_tukeyhsd

posthoc = pairwise_tukeyhsd(df2['r2'], df2['type'], alpha=0.05)
print(posthoc)

fig = posthoc.plot_simultaneous()
plt.savefig('ANOVA_Tukey_r2_Seoul_ID4_PC6.png')
plt.show()
plt.close()



# 2. Tukey, using all values

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams.update({'figure.autolayout': True})

# Predicted loading

pred_dir = "D:\\Results_2nd_final\\"

list = os.listdir(pred_dir)
pred_list = []
pred_cases = []

for i, l in enumerate(list):
    if l[-3:]=='txt':
        pass
    else:
        pred_list.append(l)
        pred_cases.append(l.split(sep='_'))

pred_cases = pd.DataFrame(pred_cases)
pred_cases = pred_cases.rename(columns={1:'type', 3:'location', 5:'seed',6:'model',7:'case'})

plot_cases = pred_cases.loc[(pred_cases[0]==str(4)) & (pred_cases[8]=='1.csv')].copy()

# Answer loading

ans_dir = 'D:\\Dropbox\\패밀리룸\\MVI\\Data\\'

list = os.listdir(ans_dir)
ans_list = []
ans_cases = []

for i, l in enumerate(list):
    ans_list.append(l)
    ans_cases.append(l.split(sep='_'))

ans_cases = pd.DataFrame(ans_cases)
ans_cases = ans_cases.rename(columns={1:'type', 3:'location'})

# Result calculation: R2, RMSE, MAE, MPE
from sklearn.metrics import r2_score
from sklearn import linear_model
import sklearn
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy.stats import gaussian_kde


ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

l = '4_AP+Meteo_1_Seoul_result_777_GAIN_ocec-elementals_1.csv'
i = pred_list.index(l)
case = pred_cases.iloc[i,:]
predicted_total = pd.read_csv(pred_dir+l)
predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
predicted = predicted[elements[elements_name.index(case.case)]]

ans_finding = ans_cases.loc[(ans_cases['type'] == case.type) & (ans_cases['location'] == case.location)].index[0]
answer_total = pd.read_csv(ans_dir+ans_list[ans_finding])
answer = answer_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
answer = answer[elements[elements_name.index(case.case)]]


predicted_GAIN = predicted


l = '4_AP+Meteo_1_Seoul_result_777_DNN_ocec-elementals_2nd_1.csv'
i = pred_list.index(l)
case = pred_cases.iloc[i,:]
predicted_total = pd.read_csv(pred_dir+l)
predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
predicted = predicted[elements[elements_name.index(case.case)]]

predicted_DNN = predicted

l = '4_AP+Meteo_1_Seoul_result_777_RF2_ocec-elementals_1.csv'
i = pred_list.index(l)
case = pred_cases.iloc[i,:]
predicted_total = pd.read_csv(pred_dir+l)
predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
predicted = predicted[elements[elements_name.index(case.case)]]

predicted_RF = predicted


l = '4_AP+Meteo_1_Seoul_result_777_KNN_ocec-elementals_1.csv'
i = pred_list.index(l)
case = pred_cases.iloc[i,:]
predicted_total = pd.read_csv(pred_dir+l)
predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
predicted = predicted[elements[elements_name.index(case.case)]]

predicted_KNN = predicted


l = '4_AP+Meteo_1_Seoul_result_777_Mean_ocec-elementals_1.csv'
i = pred_list.index(l)
case = pred_cases.iloc[i,:]
predicted_total = pd.read_csv(pred_dir+l)
predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
predicted = predicted[elements[elements_name.index(case.case)]]

predicted_Mean = predicted


for species in predicted.columns:
    target = species

#target = 'Cu'

    group0 = answer[target]
    group1 = predicted_GAIN[target]
    group2 = predicted_DNN[target]
    group3 = predicted_RF[target]
    group4 = predicted_KNN[target]
    group5 = predicted_Mean[target]


    # data combine
    group0 = pd.DataFrame(group0)
    group0['type'] = 'Answer'
    group1 = pd.DataFrame(group1)
    group1['type'] = 'GAIN'
    group2 = pd.DataFrame(group2)
    group2['type'] = 'DNN'
    group3 = pd.DataFrame(group3)
    group3['type'] = 'RF'
    group4 = pd.DataFrame(group4)
    group4['type'] = 'KNN'
    group5 = pd.DataFrame(group5)
    group5['type'] = 'Mean'


    df2 = group0.append(group1.append(group2).append(group3).append(group4)).append(group5)

    # Tukey test
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    posthoc = pairwise_tukeyhsd(df2[target], df2['type'], alpha=0.05)
    print(posthoc)

    fig = posthoc.plot_simultaneous()
    plt.title(target)
    plt.savefig('ANOVA_Tukey_total_Seoul_ID4_PC6'+target+'.png')
    plt.show()
    plt.close()




# <2021-12-28> 1개월짜리 예측의 모델결과별 비교 Tukey's

import pandas as pd
import urllib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# 1. Tukey, r2값 이용

df = pd.read_excel('d:\\Dropbox\\hadistar\\results_missing-rate.xlsx', sheet_name='results_missing-rate')

df2 = df.loc[df.rate==2].loc[df.period==0]

group1 = df2.loc[df2.model=='GAIN']
group2 = df2.loc[df2.model=='DNN']
group3 = df2.loc[df2.model=='RF']
group4 = df2.loc[df2.model=='KNN']

df3 = group1[['model', 'r2']].append(group2[['model', 'r2']]).append(group3[['model', 'r2']]).append(group4[['model', 'r2']]).reset_index(drop=True)

posthoc = pairwise_tukeyhsd(df3['r2'], df3['model'], alpha=0.05)
print(posthoc)

fig = posthoc.plot_simultaneous()

plt.show()
plt.close()
