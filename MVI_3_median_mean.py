import numpy as np
import pandas as pd
import warnings
from sklearn.impute import KNNImputer

warnings.filterwarnings('ignore')

case = '4_AP+Meteo_1_Seoul'

df = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\'+case+'_raw.csv')

data_wodate_scaled = df.iloc[:, 1:]

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

            name = case + '_result_'+ str(seeds[s])+'_Median_'+str(elements_name[ele])+'_'+str(iter+1)

            eraser = df.sample(int(len(df)*0.2), random_state=seeds[s]).index
            target = elements[ele]

            x_train = data_wodate_scaled.copy()
            x_train.loc[data_wodate_scaled.index[eraser], target] = np.nan

            y_test = np.array(data_wodate_scaled.loc[eraser, target])

            y_predicted_total = x_train.fillna(x_train.median())
            y_predicted_total = pd.DataFrame(y_predicted_total, columns=data_wodate_scaled.columns)

            y_predicted_total.to_csv(name + '.csv', index=False)

