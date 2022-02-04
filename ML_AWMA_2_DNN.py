import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt
import shutil
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)

df = pd.read_csv('AWMA_input_preprocessed_MDL_2015_2020_PM25_Meteo_AirKorea_time_case6.csv')
#df=df.drop(columns=['PM25', 'groundtemp', '30cmtemp','dewpoint', 'wd','sunshine','insolation'])

# Scaling
scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max() - df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min()) / denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

# # For case 1
# train =data_wodate_scaled.iloc[:645]
# test = data_wodate_scaled.iloc[645:1023]
#
# # For case 2
#
# train =data_wodate_scaled.iloc[:903]
# test = data_wodate_scaled.iloc[903:1416]

#
# # For case 3
#
# train =data_wodate_scaled.iloc[:786]
# test = data_wodate_scaled.iloc[786:1194]
#
#
# # For case 4
# data_wodate_scaled =data_wodate_scaled.iloc[:759]


# For case 7

data_wodate_scaled =data_wodate_scaled.iloc[:1194]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

col_dic = {'ocec': ocec, 'elementals': elementals, 'ions': ions, 'ocec-elementals': ocec+elementals,
           'ions-ocec': ions+ocec, 'ions-elementals': ions+elementals, 'ions-ocec-elementals': ions+ocec+elementals}

Missing_Col = ['ions','ocec', 'elementals', 'ocec-elementals', 'ions-ocec', 'ions-elementals', 'ions-ocec-elementals']

#Missing_Col = ['elementals']

iteration = 3

# DNN 모델


case = 'DNN_case7_'

for ele in Missing_Col:
    for iter in range(iteration):

        name = 'AWMA_result_' + case + str(ele) + '_iter_' + str(iter + 1)
        #
        # # train x,y 만들기 for case 1
        #
        # temp = np.array(train[col_dic[ele]])
        #
        # train_x = []
        # train_y = []
        # for i in range(0,len(train),3):
        #     temp = np.array(train)[i] # i번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(train.drop(col_dic[ele], axis=1))[i+1]) # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
        #     temp = np.append(temp, np.array(train)[i+2]) # i+2번째 행의 모든 자료 추가
        #
        #     train_x.append(temp)
        #
        #     train_y.append(np.array(train[col_dic[ele]])[i+1])
        #
        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        #
        # # test x, y 만들기 for case 1
        #
        # test_x = []
        # test_y = []
        # for i in range(0, len(test), 3):
        #     temp = np.array(test)[i]  # i번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(test.drop(col_dic[ele], axis=1))[i + 1])  # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
        #     temp = np.append(temp, np.array(test)[i + 2])  # i+2번째 행의 모든 자료 추가
        #
        #     test_x.append(temp)
        #
        #     test_y.append(np.array(test[col_dic[ele]])[i + 1])
        #
        # test_x = np.array(test_x)
        # test_y = np.array(test_y)
        #
        #
        # # train x,y 만들기 for case 2
        #
        # temp = np.array(train[col_dic[ele]])
        #
        # train_x = []
        # train_y = []
        # for i in range(0,len(train),3):
        #     temp = np.array(train)[i+1] # i+1번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(train.drop(col_dic[ele], axis=1))[i]) # i번째 행의 자료 추가, 예측 target은 빼고 추가
        #     temp = np.append(temp, np.array(train.drop(col_dic[ele], axis=1))[i+2]) # i+2번째 행의 자료 추가, 예측 target은 빼고 추가
        #     train_x.append(temp)
        #
        #     temp = np.array(train[col_dic[ele]])[i] # i번째의 행 예측 target 추가
        #     temp = np.append(temp, np.array(train[col_dic[ele]])[i+2]) # i+2번째의 행 예측 target 추가
        #     train_y.append(temp)
        #
        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        #
        # # test x, y 만들기 for case 2
        #
        # test_x = []
        # test_y = []
        # for i in range(0, len(test), 3):
        #     temp = np.array(test)[i + 1]  # i+1번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(test.drop(col_dic[ele], axis=1))[i])  # i번째 행의 자료 추가, 예측 target은 빼고 추가
        #     temp = np.append(temp,
        #                      np.array(test.drop(col_dic[ele], axis=1))[i + 2])  # i+2번째 행의 자료 추가, 예측 target은 빼고 추가
        #     test_x.append(temp)
        #
        #     temp = np.array(test[col_dic[ele]])[i]  # i번째의 행 예측 target 추가
        #     temp = np.append(temp, np.array(test[col_dic[ele]])[i + 2])  # i+2번째의 행 예측 target 추가
        #     test_y.append(temp)
        #
        # test_x = np.array(test_x)
        # test_y = np.array(test_y)


        # # train x,y 만들기 for case 3
        #
        # temp = np.array(train[col_dic[ele]])
        #
        # train_x = []
        # train_y = []
        # for i in range(0,len(train),2):
        #     temp = np.array(train)[i] # i번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(train.drop(col_dic[ele], axis=1))[i+1]) # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
        #
        #     train_x.append(temp)
        #
        #     temp = np.array(train[col_dic[ele]])[i+1] # i+1번째의 행 예측 target 추가
        #     train_y.append(temp)
        #
        # train_x = np.array(train_x)
        # train_y = np.array(train_y)
        #
        # # test x, y 만들기 for case 3
        #
        # test_x = []
        # test_y = []
        # for i in range(0, len(test), 2):
        #     temp = np.array(test)[i]  # i번째 행의 모든 자료로 temp 변수 생성
        #     temp = np.append(temp, np.array(test.drop(col_dic[ele], axis=1))[i+1])  # i+1번째 행의 자료 추가, 예측 target은 빼고 추가
        #     test_x.append(temp)
        #
        #     temp = np.array(test[col_dic[ele]])[i+1]  # i+1번째의 행 예측 target 추가
        #     test_y.append(temp)
        #
        # test_x = np.array(test_x)
        # test_y = np.array(test_y)


        #
        # # train x,y, test x, y 만들기 for case 4
        #
        #
        # eraser = np.random.randint(0, 759, size=int(759*0.2))
        #
        # train_x = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=col_dic[ele]))
        # train_y = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, col_dic[ele]]
        # train_y = np.array(train_y)
        #
        # test_x = np.array(data_wodate_scaled.loc[eraser].drop(columns=col_dic[ele]))
        # test_y = np.array(data_wodate_scaled.loc[eraser, col_dic[ele]])
        #


        # # train x,y, test x, y 만들기 for case 5
        #
        #
        # eraser = np.random.randint(0, 759, size=int(759*0.2))
        #
        # train_x = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=col_dic[ele]))
        # train_y = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, col_dic[ele]]
        # train_y = np.array(train_y)
        #
        # test_x = np.array(data_wodate_scaled.loc[eraser].drop(columns=col_dic[ele]))
        # test_y = np.array(data_wodate_scaled.loc[eraser, col_dic[ele]])


        # # train x,y, test x, y 만들기 for case 6
        #
        # eraser = np.random.randint(0, 759, size=int(759 * 0.2))
        #
        # train_x = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=col_dic[ele]))
        # train_y = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, col_dic[ele]]
        # train_y = np.array(train_y)
        #
        # test_x = np.array(data_wodate_scaled.loc[eraser].drop(columns=col_dic[ele]))
        # test_y = np.array(data_wodate_scaled.loc[eraser, col_dic[ele]])
        #


        # train x,y, test x, y 만들기 for case 7

        eraser = np.random.choice(int(1194/2), int(1194/2 * 0.2), replace=False)*2
        count=0

        test_x = []
        test_y = []
        train_x = []
        train_y = []

        for i in range(0,len(data_wodate_scaled),2):
            if i in eraser:
                temp_test_x = data_wodate_scaled.iloc[i]
                temp = data_wodate_scaled.iloc[i+1]
                test_x.append(temp_test_x.append(temp.drop(col_dic[ele])))
                test_y.append(temp[col_dic[ele]])
            else:
                temp_train_x = data_wodate_scaled.iloc[i]
                temp = data_wodate_scaled.iloc[i+1]
                train_x.append(temp_train_x.append(temp.drop(col_dic[ele])))
                train_y.append(temp[col_dic[ele]])

        test_x = np.array(test_x)
        test_y = np.array(test_y)
        train_x = np.array(train_x)
        train_y = np.array(train_y)


        def model_builder(hp):
            model = keras.Sequential()

            # Tune the number of units in the first Dense layer
            # Choose an optimal value between 32-512

            for i in range(hp.Int('num_layers', 2, 7)):
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
                             factor=2,
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
        training r2 = {model.evaluate(train_x, train_y)[1]}
        test R2 = {evaluation[1]}
        """)

        f = open(name + '.txt', 'w')
        f.write(f"""
        The hyperparameter search is complete. 
        The optimal number of layers: {best_hps.get('num_layers')}
        The optimal number of units: {best_hps.get('units')}
        The optimal learning rate: {best_hps.get('learning_rate')}.
        The optimal dropout rate: {best_hps.get('dropout_rate')}.
        training r2 = {model.evaluate(train_x, train_y)[1]}
        test R2 = {evaluation[1]}
        """)
        f.close()

        # For case 1,...

        y_predicted_pd = pd.DataFrame(y_predicted, columns=col_dic[ele])
        y_predicted_answer = pd.DataFrame(test_y, columns=col_dic[ele])


        # # For case 2 only
        # y_predicted_pd1 = pd.DataFrame(y_predicted[:,:len(col_dic[ele])], columns=col_dic[ele])
        # y_predicted_pd2 = pd.DataFrame(y_predicted[:, len(col_dic[ele]):], columns=col_dic[ele])
        # y_predicted_answer1 = pd.DataFrame(test_y[:,:len(col_dic[ele])], columns=col_dic[ele])
        # y_predicted_answer2 = pd.DataFrame(test_y[:,len(col_dic[ele]):], columns=col_dic[ele])
        #


        # rescaling
        # x = x' * (max-min) + min
        # saving scaling factor in [max-min, min, max]

        # For case 1,...

        for c in y_predicted_pd:
            y_predicted_pd[c] = y_predicted_pd[c] * scalingfactor[c][0] + scalingfactor[c][1]
            y_predicted_answer[c] = y_predicted_answer[c] * scalingfactor[c][0] + scalingfactor[c][1]

        writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
        y_predicted_pd.to_excel(writer, sheet_name='predicted')
        y_predicted_answer.to_excel(writer, sheet_name='answer')
        writer.save()
        writer.close

        # # For case 2 only
        #
        # for c in y_predicted_pd1:
        #     y_predicted_pd1[c] = y_predicted_pd1[c] * scalingfactor[c][0] + scalingfactor[c][1]
        #     y_predicted_answer1[c] = y_predicted_answer1[c] * scalingfactor[c][0] + scalingfactor[c][1]
        # for c in y_predicted_pd2:
        #     y_predicted_pd2[c] = y_predicted_pd2[c] * scalingfactor[c][0] + scalingfactor[c][1]
        #     y_predicted_answer2[c] = y_predicted_answer2[c] * scalingfactor[c][0] + scalingfactor[c][1]
        #
        # y_predicted_pd = pd.concat([y_predicted_pd1, y_predicted_pd2], axis=1)
        # y_predicted_answer = pd.concat([y_predicted_answer1,y_predicted_answer2], axis=1)
        #
        # writer = pd.ExcelWriter(name+'.xlsx', engine='xlsxwriter')
        # y_predicted_pd.to_excel(writer, sheet_name='predicted')
        # y_predicted_answer.to_excel(writer, sheet_name='answer')
        # writer.save()
        # writer.close

        del model
        del history
        shutil.rmtree('D:/kerastuner/' + name)
