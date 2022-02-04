import pandas as pd
import os
import numpy as np

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

pred_dir = 'D:\\OneDrive - SNU\\바탕 화면\\ML_AWML_results\\'

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
pred_cases = pred_cases.rename(columns={3:'case', 4:'type', 6:'number'})




ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

col_dic = {'ocec': ocec, 'elementals': elementals, 'ions': ions, 'ocec-elementals': ocec+elementals,
           'ions-ocec': ions+ocec, 'ions-elementals': ions+elementals, 'ions-ocec-elementals': ions+ocec+elementals}

Missing_Col = ['ions','ocec', 'elementals', 'ocec-elementals', 'ions-ocec', 'ions-elementals', 'ions-ocec-elementals']


for i, l in enumerate(pred_list):
    print (l)

    case = pred_cases.iloc[i, :]
    predicted = pd.read_excel(pred_dir + l, sheet_name='predicted')
    predicted = predicted.iloc[:, 1:]
    answer = pd.read_excel(pred_dir + l, sheet_name='answer')
    answer = answer.iloc[:, 1:]

    results_r2 = r2_score(np.array(answer), np.array(predicted))
    results_RMSE = np.sqrt(mean_squared_error(np.array(answer), np.array(predicted)))
    results_MAE = mean_absolute_error(np.array(answer), np.array(predicted))
    results_MAPE = mean_absolute_percentage_error(np.array(answer), np.array(predicted))

    pred_cases.loc[i, 'r2'] = results_r2
    pred_cases.loc[i, 'RMSE'] = results_RMSE
    pred_cases.loc[i, 'MAE'] = results_MAE
    pred_cases.loc[i, 'MAPE'] = results_MAPE

    if pred_cases.iloc[i].case == 'case2':
        for column in predicted.columns:
            if column[-2:] != '.1':
                pred_cases.loc[i, 'r2_'+column] = r2_score(np.array(answer[column]), np.array(predicted[column]))
            else:
                pred_cases.loc[i, 'r2_' + column] = r2_score(np.array(answer[column]), np.array(predicted[column]))

    else:
        for column in predicted.columns:
            pred_cases.loc[i, 'r2_'+column] = r2_score(np.array(answer[column]), np.array(predicted[column]))



pred_cases.to_csv('results_ML_AWMA_220114.csv', index=False)






