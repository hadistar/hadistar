import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

case = '4_AP+Meteo_1_Seoul'

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


seeds = [777]
elements = [ions+elementals, ocec+elementals,ions+ocec+elementals]
elements_name = ['ion-elementals','ocec-elementals','ions-ocec-elementals']

iteration = 1

for s in range(len(seeds)):
    for ele in range(len(elements)):
        for iter in range(iteration):

            name = case + '_result_'+ str(seeds[s])+'_RF_'+str(elements_name[ele])+'_missing rate_'+str(int(missing_ratio*10))+'_'+str(iter+1)

            eraser = df.sample(int(len(df)*0.2), random_state=seeds[s]).index
            target = elements[ele]

            x_train = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=target))
            y_train = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, target]
            y_train = np.array(y_train)

            x_test = np.array(data_wodate_scaled.loc[eraser].drop(columns=target))
            y_test = np.array(data_wodate_scaled.loc[eraser, target])

            ##하이퍼 파라미터 최적화

            from sklearn.ensemble import RandomForestRegressor
            from hyperopt import tpe, hp, Trials
            from hyperopt.fmin import fmin
            from sklearn.metrics import mean_squared_error

            def objective(params):
                est = int(params['n_estimators'])
                md = int(params['max_depth'])
                msl = int(params['min_samples_leaf'])
                mss = int(params['min_samples_split'])
                mf = params['max_features']
                model = RandomForestRegressor(n_estimators=est, max_depth=md, min_samples_leaf=msl,
                                              min_samples_split=mss, max_features=mf, n_jobs=-1)
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                score = mean_squared_error(y_test, pred)
                return score


            def optimize(trial):
                params = {'n_estimators': hp.uniform('n_estimators', 1, 2000),
                          'max_depth': hp.uniform('max_depth', 1, 30),
                          'min_samples_leaf': hp.uniform('min_samples_leaf', 1, 30),
                          'min_samples_split': hp.uniform('min_samples_split', 2, 30),
                          'max_features': hp.choice('max_features', [None,'auto','sqrt'])
                          }
                best = fmin(fn=objective, space=params, algo=tpe.suggest, trials=trial, max_evals=300)
                return best


            trial = Trials()
            best = optimize(trial)


            best_n_estimators = round(best['n_estimators'])
            best_max_depth = round(best['max_depth'])
            best_min_samples_leaf = round(best['min_samples_leaf'])
            best_min_samples_split = round(best['min_samples_split'])
            best_max_features = best['max_features']

            best_model = RandomForestRegressor(n_estimators=best_n_estimators,
                                               max_depth=best_max_depth,
                                               min_samples_leaf=best_min_samples_leaf,
                                               min_samples_split=best_min_samples_split,
                                               max_features=None,
                                               n_jobs=-1)

            best_model.fit(x_train, y_train)

            y_predicted = best_model.predict(x_test)
            evaluation = best_model.score(x_test, y_test)

            f = open(name + '.txt', 'w')
            f.write(f"""
                The hyperparameter search is complete. 
                The best hyperparameters
                 - n_estimators: {best_n_estimators}
                 - max_depth: {best_max_depth}
                 - min_samples_leaf: {best_min_samples_leaf}
                 - min_samples_split: {best_min_samples_split}
                 - max_features: {best_max_features}
                R2 = {evaluation}.
                """)
            f.close()

    # rescaling
    # x = x' * (max-min) + min
    # saving scaling factor in [max-min, min, max]

            y_predicted_total = best_model.predict(np.array(data_wodate_scaled.drop(columns=target)))
            y_predicted_total = pd.DataFrame(y_predicted_total, columns=target)

            for c in y_predicted_total:
                y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

            y_predicted_total.to_csv(name + '.csv', index=False)



