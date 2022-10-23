# importing the packages

import pandas as pd
import numpy as np
import impyute as impy

# Data loading
# 비교 대상 자료: Seoul, ID#4, PC#7, seed number: 777

Data_Name = ['4_AP+Meteo_1_Seoul']
seeds = [777]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions+ocec+elementals]
elements_name = ['ions-ocec-elementals']

iteration = 2

case = Data_Name[0]
s = 0

df = pd.read_csv('D:\\OneDrive - SNU\\hadistar\\MVI_공모전_backup\\Data\\' + case + '_raw.csv')

scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max() - df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min()) / denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

for ele in range(len(elements)):
    for iter in range(iteration):

        name = case + '_result_' + str(seeds[s]) + '_EM_' + str(elements_name[ele]) + '_' + str(iter + 1)
        eraser = df.sample(int(len(df) * 0.2), random_state=seeds[s]).index
        target = elements[ele]

        x_train = data_wodate_scaled.copy()
        x_train.loc[data_wodate_scaled.index[eraser], target] = np.nan

        y_test = np.array(data_wodate_scaled.loc[eraser, target])

        ## impute EM algorithm

        y_predicted_total = impy.em(x_train.values, loops = 1000)

        # rescaling
        # x = x' * (max-min) + min
        # saving scaling factor in [max-min, min, max]

        y_predicted_total = pd.DataFrame(y_predicted_total, columns=data_wodate_scaled.columns)

        for c in y_predicted_total:
            y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

        y_predicted_total.to_csv(name + '.csv', index=False)



#---------------------------------------------------
# Results processing
#---------------------------------------------------

import pandas as pd
import os
import numpy as np
# Predicted loading

name = '4_AP+Meteo_1_Seoul_result_777_EM_ions-ocec-elementals_1'

list = [name]
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

# Answer loading

ans_dir = 'D:\\OneDrive - SNU\\hadistar\\MVI_공모전_backup\\Data\\'

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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

def MPE(y_test, y_pred):
    return np.mean(((y_test - y_pred) / y_test+1e-8) * 100)

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

for i, l in enumerate(pred_list):
    print(l)
    case = pred_cases.iloc[i,:]
    predicted_total = pd.read_csv(l)
    predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
    predicted = predicted[elements[elements_name.index(case.case)]]

    ans_finding = ans_cases.loc[(ans_cases['type'] == case.type) & (ans_cases['location'] == case.location)].index[0]
    answer_total = pd.read_csv(ans_dir+ans_list[ans_finding])
    answer = answer_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
    answer = answer[elements[elements_name.index(case.case)]]

    results_r2 = r2_score(np.array(answer), np.array(predicted))
    results_RMSE = np.sqrt(mean_squared_error(np.array(answer), np.array(predicted)))
    results_MAE = mean_absolute_error(np.array(answer), np.array(predicted))
    results_MAPE = mean_absolute_percentage_error(np.array(answer), np.array(predicted))

    pred_cases.loc[i,'r2'] = results_r2
    pred_cases.loc[i,'RMSE'] = results_RMSE
    pred_cases.loc[i,'MAE'] = results_MAE
    pred_cases.loc[i,'MAPE'] = results_MAPE

    for column in predicted.columns:
        pred_cases.loc[i, 'r2_'+column] = r2_score(np.array(answer[column]), np.array(predicted[column]))

pred_cases.to_csv('results_EM_221023.csv', index=False)

## 1:1 Plot - prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 13
plt.rcParams.update({'figure.autolayout': True})

# Result calculation: R2, RMSE, MAE, MPE
from sklearn.metrics import r2_score
from sklearn import linear_model
import sklearn

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

limits = {'SO42-':40, 'NO3-':80.0, 'Cl-':3.0, 'Na+':1.2, 'NH4+':30,
          'K+':2.0, 'Mg2+':0.3, 'Ca2+':2.4,'OC':15.0, 'EC':6.0,
          'S':20.0, 'K':2.5, 'Ca':2.1, 'Ti':0.25, 'V':0.04,
          'Cr':0.016, 'Mn':0.08, 'Fe':3.0, 'Ni':0.02, 'Cu':0.08,
          'Zn':0.8, 'As':0.4, 'Se':0.015, 'Br':0.07, 'Pb':0.2}

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

for i, l in enumerate(pred_list):

    if l in pred_list:
        print(l)

        case = pred_cases.iloc[i,:]
        predicted_total = pd.read_csv(l)
        predicted = predicted_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
        predicted = predicted[elements[elements_name.index(case.case)]]

        ans_finding = ans_cases.loc[(ans_cases['type'] == case.type) & (ans_cases['location'] == case.location)].index[0]
        answer_total = pd.read_csv(ans_dir+ans_list[ans_finding])
        answer = answer_total.sample(int(len(predicted_total)*0.2), random_state=int(case.seed))
        answer = answer[elements[elements_name.index(case.case)]]

        results_r2 = r2_score(np.array(answer), np.array(predicted))

        for species in predicted.columns:

            x = np.array(predicted[species])
            y = np.array(answer[species])
            print(species, x.max(), y.max())
            # Create linear regression object
            linreg = linear_model.LinearRegression()
            # Fit the linear regression model
            model = linreg.fit(x.reshape(-1, 1), y.reshape(-1, 1))

            # model = linreg.fit(x.to_numpy().reshape(-1, 1), y.to_numpy().reshape(-1, 1))
            # Get the intercept and coefficients
            intercept = model.intercept_
            coef = model.coef_
            result = [intercept, coef]
            predicted_y = x.reshape(-1, 1) * coef + intercept
            r_squared = sklearn.metrics.r2_score(y, predicted_y)

            plt.figure(figsize=(5, 5))
            # plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
            plt.plot(x, y, 'ko', markersize=8, mfc='none')

            plt.plot(x, predicted_y, 'b-', 0.1)
            plt.plot([0, limits[species]], [0, limits[species]], 'k--')

            plt.xlabel('Prediction (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.ylabel('Observation (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.text(limits[species] * 0.5, limits[species] * 0.1,
                     '$R^2$ = %0.2f' % (r_squared), fontsize=15)
            if intercept >= 0:
                plt.text(limits[species] * 0.5, limits[species] * 0.04,
                     'y = %0.4fx + %0.4f'
                     % (coef, intercept))
            else:
                plt.text(limits[species] * 0.5, limits[species] * 0.04,
                         'y = %0.4fx - %0.4f'
                         % (coef, abs(intercept)))

            plt.axis([0, limits[species], 0, limits[species]])
            plt.grid(True, linestyle='--')
            #plt.legend(title=species, loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig('D:\\temp\\' + species + '_predicted_treated.png')
#            plt.show()
            plt.close()

