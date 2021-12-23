import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt
import shutil
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)


df = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time.csv')

# Scaling
scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max() - df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min()) / denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

train =data_wodate_scaled.iloc[18:645]
test = data_wodate_scaled.iloc[645:1023]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

col_dic = {'ocec': ocec, 'elementals': elementals, 'ions': ions, 'ocec-elementals': ocec+elementals,
           'ions-ocec': ions+ocec, 'ions-elementals': ions+elementals, 'ions-ocec-elementals': ions+ocec+elementals}
Missing_Col = ['ions','ocec', 'elementals', 'ocec-elementals', 'ions-ocec', 'ions-elementals', 'ions-ocec-elementals']


iteration = 2

# DNN 모델


case = '3day_DNN'

for ele in Missing_Col:
    for iter in range(iteration):

        name = 'result_' + case + '_2019predicttion_' + str(ele) + '_iter_' + str(iter + 1)

        # train x,y 만들기

        temp = np.array(train[col_dic[ele]])

        train_x = []
        train_y = []
        for i in range(0,len(train),3):
            temp = np.array(train)[i] # i번째 행의 모든 자료로 temp 변수 생성
            temp = np.append(temp, np.array(train.drop(col_dic[ele], axis=1))[i+1]) # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
            temp = np.append(temp, np.array(train)[i+2]) # i+2번째 행의 모든 자료 추가

            train_x.append(temp)

            train_y.append(np.array(train[col_dic[ele]])[i+1])

        train_x = np.array(train_x)
        train_y = np.array(train_y)

        # test x, y 만들기

        test_x = []
        test_y = []
        for i in range(0, len(test), 3):
            temp = np.array(test)[i]  # i번째 행의 모든 자료로 temp 변수 생성
            temp = np.append(temp, np.array(test.drop(col_dic[ele], axis=1))[i + 1])  # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
            temp = np.append(temp, np.array(test)[i + 2])  # i+2번째 행의 모든 자료 추가

            test_x.append(temp)

            test_y.append(np.array(test[col_dic[ele]])[i + 1])

        test_x = np.array(test_x)
        test_y = np.array(test_y)


        def model_builder(hp):
            model = keras.Sequential()

            # Tune the number of units in the first Dense layer
            # Choose an optimal value between 32-512

            for i in range(hp.Int('num_layers', 2, 9)):
                model.add(keras.layers.Dense(units=hp.Int('units',
                                                          min_value=32,
                                                          max_value=2048,
                                                          step=32),
                                             activation=LeakyReLU))
                # activation=hp.Choice('activation_function',
                #                      values=[
                #                              'relu',
                #                              'tahn'
                #                             ])))
                model.add(keras.layers.Dropout(hp.Choice('dropout_rate',
                                                         values=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17,
                                                                 0.18, 0.19, 0.2])))

            model.add(keras.layers.Dense(units=train_y.shape[1],
                                         activation=LeakyReLU))

            # activation=hp.Choice('activation_function',
            #                      values=[
            #                              'relu',
            #                              'tahn'
            #                              ])))

            # Tune the learning rate for the optimizer
            # Choose an optimal value from 0.01, 0.001, or 0.0001
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5, 1e-6])

            model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                          loss='mse',
                          metrics=['accuracy'])

            return model


        tuner = kt.Hyperband(model_builder,
                             objective='val_accuracy',
                             max_epochs=200,
                             factor=3,
                             directory='D:/kerastuner',
                             project_name='D:/kerastuner/' + name)


        # tuner = kt.BayesianOptimization(model_builder,
        #                                 objective='val_accuracy',
        #                                 max_trials=100,
        #                                 directory='D:/kerastuner',
        #                                 project_name='D:/kerastuner/'+name)

        class ClearTrainingOutput(tf.keras.callbacks.Callback):
            def on_train_end(*args, **kwargs):
                IPython.display.clear_output(wait=True)


        tuner.search(train_x, train_y, epochs=100, validation_split=0.2, verbose=1,
                     callbacks=[ClearTrainingOutput()])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        tuner.search_space_summary()
        tuner.results_summary()

        # Build the model with the optimal hyperparameters and train it on the data
        model = tuner.hypermodel.build(best_hps)

        history = model.fit(train_x, train_y, epochs=200, validation_split=0.2, verbose=1)

        y_predicted = model.predict(test_x)
        evaluation = model.evaluate(test_x, test_y)

        # For making total results

        print(name + 'done!')
        print(f"""
        The hyperparameter search is complete. 
        The optimal number of layers: {best_hps.get('num_layers')}
        The optimal number of units: {best_hps.get('units')}
        The optimal learning rate: {best_hps.get('learning_rate')}.
        The optimal dropout rate: {best_hps.get('dropout_rate')}.
        R2 = {evaluation[1]}.
        """)

        f = open(name + '.txt', 'w')
        f.write(f"""
        The hyperparameter search is complete. 
        The optimal number of layers: {best_hps.get('num_layers')}
        The optimal number of units: {best_hps.get('units')}
        The optimal learning rate: {best_hps.get('learning_rate')}.
        The optimal dropout rate: {best_hps.get('dropout_rate')}.
        R2 = {evaluation[1]}.
        """)
        f.close()

        # f.write(f"""
        # The hyperparameter search is complete.
        # The optimal number of layers: {best_hps.get('num_layers')}
        # The optimal number of units: {best_hps.get('units')}
        # The optimal learning rate: {best_hps.get('learning_rate')}.
        # The optimal dropout rate: {best_hps.get('dropout_rate')}.
        # The optimal activation function: {best_hps.get('activation_function')}.
        # R2 = {evaluation[1]}.
        # """)
        # f.close()

        # rescaling
        # x = x' * (max-min) + min
        # saving scaling factor in [max-min, min, max]

        for c in y_predicted_total:
            y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

        y_predicted_total.to_csv(name + '.csv', index=False)

        del model
        del history
        shutil.rmtree('D:/kerastuner/' + name)
