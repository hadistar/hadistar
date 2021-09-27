import numpy as np
import pandas as pd
import warnings
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')

case = '1_Basic_1_Seoul'

df = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\'+case+'_raw.csv')
scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max()-df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min())/denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

# seeds = [777, 1004, 322, 224, 417]
seeds = [777, 1004, 322]
ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals,ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals','ions-ocec-elementals']

#seeds = [1004, 322]
#elements = [ions+ocec+elementals]
#elements_name = ['ions-ocec-elementals']

iteration = 1

for s in range(len(seeds)):
    for ele in range(len(elements)):
        for iter in range(iteration):

            name = case + '_result_'+ str(seeds[s])+'_KNN_'+str(elements_name[ele])+'_'+str(iter+3)

            eraser = df.sample(int(len(df)*0.2), random_state=seeds[s]).index
            target = elements[ele]

            x_train = data_wodate_scaled.copy()
            x_train.loc[data_wodate_scaled.index[eraser], target] = np.nan

            y_test = np.array(data_wodate_scaled.loc[eraser, target])

            ##하이퍼 파라미터 최적화

            from sklearn.ensemble import RandomForestRegressor

            from sklearn.metrics import mean_squared_error

            imputer = KNNImputer(n_neighbors=3)  # KNN
            y_predicted_total = imputer.fit_transform(x_train)


    # rescaling
    # x = x' * (max-min) + min
    # saving scaling factor in [max-min, min, max]

            y_predicted_total = pd.DataFrame(y_predicted_total, columns=data_wodate_scaled.columns)

            for c in y_predicted_total:
                y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

            y_predicted_total.to_csv(name + '.csv', index=False)

