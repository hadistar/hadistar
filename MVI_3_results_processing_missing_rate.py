import pandas as pd
import os
import numpy as np
from scipy import stats
# Predicted loading

# pred_dir = 'C:\\Users\\Minjae\\Dropbox\\패밀리룸\\MVI\\Results\\'
pred_dir = 'D:\\Dropbox\\패밀리룸\\MVI\\Results_missing rate_length\\'
#pred_dir = 'D:\\OneDrive - SNU\\바탕 화면\\MVI_results_missing rate\\'
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
pred_cases = pred_cases.rename(columns={1:'type', 3:'location', 5:'seed',6:'model',7:'case', 9:'rate', 11:'period'})

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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
def MPE(y_test, y_pred):
    return np.mean(((y_test - y_pred) / y_test+1e-5) * 100)

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

for i, l in enumerate(pred_list):
    print(l)
    case = pred_cases.iloc[i,:]
    predicted_total = pd.read_csv(pred_dir+l)
    predicted = predicted_total.sample(int(len(predicted_total)*round(int(case.rate)/10, 1)), random_state=int(case.seed))
    predicted = predicted[elements[elements_name.index(case.case)]]

    ans_finding = ans_cases.loc[(ans_cases['type'] == case.type) & (ans_cases['location'] == case.location)].index[0]
    answer_total = pd.read_csv(ans_dir+ans_list[ans_finding])
    if case['period']=='0':
        answer=answer_total.copy()
        n_total = len(answer)

    else:
        if case['model']=='RF' or case['model']=='DNN':
            answer = answer_total[-30 * 24 * int(case['period']):].reset_index(drop=True)
            n_total = len(answer)
        elif case['model']=='GAIN' or case['model']=='KNN':
            if case['period']=='12':
                answer = answer_total[-24 * 365:].reset_index(drop=True)
                n_total = len(answer)

            else:
                answer = answer_total[-30 * 24 * int(case['period']):].reset_index(drop=True)
                n_total = len(answer)



    answer = answer.sample(int(len(predicted_total)*round(int(case.rate)/10, 1)), random_state=int(case.seed))
    answer = answer[elements[elements_name.index(case.case)]]
    n_test = len(answer)
    results_r2 = r2_score(np.array(answer), np.array(predicted))

    results_RMSE = np.sqrt(mean_squared_error(np.array(answer), np.array(predicted)))
    results_MAE = mean_absolute_error(np.array(answer), np.array(predicted))
    # results_MPE = MPE(np.array(answer), np.array(predicted))

    pred_cases.loc[i,'r2'] = results_r2
    pred_cases.loc[i,'RMSE'] = results_RMSE
    pred_cases.loc[i,'MAE'] = results_MAE
    pred_cases.loc[i,'train n'] = n_total-n_test
    pred_cases.loc[i,'test n'] = n_test
    pred_cases.loc[i,'total n'] = n_total


    pred_cases.to_csv('results_missing-rate.csv', index=False)