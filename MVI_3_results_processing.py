import pandas as pd
import os
import numpy as np
# Predicted loading
#test

pred_dir = 'D:\\Dropbox\\패밀리룸\\MVI\\Results\\'

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
    predicted_total = pd.read_csv(pred_dir+l)
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

pred_cases.to_csv('results__211220.csv', index=False)





## 1:1 Plot




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

limits = {'SO42-':40, 'NO3-':80.0, 'Cl-':3.0, 'Na+':1.2, 'NH4+':30,
          'K+':2.0, 'Mg2+':0.3, 'Ca2+':2.4,'OC':15.0, 'EC':6.0,
          'S':20.0, 'K':2.5, 'Ca':2.1, 'Ti':0.25, 'V':0.04,
          'Cr':0.016, 'Mn':0.08, 'Fe':3.0, 'Ni':0.02, 'Cu':0.08,
          'Zn':0.8, 'As':0.4, 'Se':0.015, 'Br':0.07, 'Pb':0.2}

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

for i, l in enumerate(pred_list):

    if l in ['4_AP+Meteo_1_Seoul_result_777_Mean_ions-ocec-elementals_1.csv']:
        print(l)

        case = pred_cases.iloc[i,:]
        predicted_total = pd.read_csv(pred_dir+l)
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

            plt.xlabel('Predicted concentration of sampled filter (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.ylabel('Observed concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.text(limits[species] * 0.5, limits[species] * 0.01,
                     'y = %0.4fx + %0.4f\n$r^2$ = %0.2f (n=%s)'
                     % (coef, intercept, r_squared, format(len(x), ',')))

            plt.axis([0, limits[species], 0, limits[species]])
            plt.grid(True, linestyle='--')
            #plt.legend(title=species, loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig('D:\\temp\\' + species + '_mean_treated.png')
            plt.close()



## with density version
import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
    (0, '#ffffff'),
    (1e-20, '#440053'),
    (0.2, '#404388'),
    (0.4, '#2a788e'),
    (0.6, '#21a784'),
    (0.8, '#78d151'),
    (1, '#fde624'),
], N=3000)

from scipy.stats import gaussian_kde


for i, l in enumerate(pred_list):

    if l in ['4_AP+Meteo_1_Seoul_result_777_DNN_ion-ocec_1.csv','4_AP+Meteo_1_Seoul_result_777_DNN_ions_1.csv','4_AP+Meteo_1_Seoul_result_777_DNN_elementals_1.csv']:
        print(l)

        case = pred_cases.iloc[i,:]
        predicted_total = pd.read_csv(pred_dir+l)
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

            fig = plt.figure(figsize=(5, 5))

            # Calculate the point density
            xy = np.vstack([x, y])
            z = gaussian_kde(xy)(xy)

            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            xx, yy, zz = x[idx], y[idx], z[idx]


            fig, ax = plt.subplots()
            density = ax.scatter(xx, yy, c=zz, s=30, edgecolor=['none'], cmap=white_viridis)
            fig.colorbar(density, label='Density')

            # ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
            # density = ax.scatter_density(x, y, cmap=white_viridis)
            # fig.colorbar(density, label='Number of points per pixel')

            # plt.scatter(x, y, s=40, facecolors='none', edgecolors='k')
            #plt.plot(x, y, 'ro', markersize=1.2)

            ax.plot(x, predicted_y, 'b-', 0.1)
            plt.plot([0, limits[species]], [0, limits[species]], 'k--')

            plt.xlabel('Predicted concentration of sampled filter (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.ylabel('Observed concentration (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
            plt.text(limits[species] * 0.25, limits[species] * 0.85,
                     'y = %0.4fx + %0.4f\n$r^2$ = %0.2f (n=%s)'
                     % (coef, intercept, r_squared, format(len(x), ',')))
            ax.set_xlim(0, limits[species])
            ax.set_ylim(0, limits[species])
            #plt.axis([0, limits[species], 0, limits[species]])
            ax.grid(True, linestyle='--')
            ax.legend(title=species, loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig('D:\\temp\\' + species + '_mean_treated.png')

            plt.close()



BR = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\1_Basic_2_BR_raw.csv')

x = np.array(BR['SO42-'])
y = np.array(BR['S'])

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
plt.plot(x, y, 'ro', markersize=1.2)

plt.plot(x, predicted_y, 'b-', 0.1)
plt.plot([0, 45], [0, 45], 'k--')

plt.xlabel('Concentration SO$_4$$^{2-}$ (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')
plt.ylabel('Concentration of S (' + "${\mu}$" + 'g/m' + r'$^3$' + ')')

plt.text(5, 30,
         'y = %0.2fx + %0.2f\n$r^2$ = %0.2f (n=%s)'
         % (coef, intercept, r_squared, format(len(x), ',')))

plt.axis([0, 45, 0, 45])
plt.grid(True, linestyle='--')
plt.legend(title="SO$_4$$^{2-}$ vs. S", loc='upper left')
plt.tight_layout()
plt.show()
plt.close()

