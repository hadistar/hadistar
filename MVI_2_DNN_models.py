import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt
import shutil
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)


Data_Name = ['1_Basic_1_Seoul', '1_Basic_2_BR', '1_Basic_3_Ulsan',
             '2_Informed_1_Seoul', '2_Informed_2_BR', '2_Informed_3_Ulsan',
             '3_AP_1_Seoul', '3_AP_2_BR', '3_AP_3_Ulsan',
             '4_AP+Meteo_1_Seoul', '4_AP+Meteo_2_BR', '4_AP+Meteo_3_Ulsan']

Data_Name = ['1_Basic_1_Seoul', '1_Basic_3_Ulsan']


seeds = [777, 1004, 322] #, 224, 417]
seeds = [1004] #, 224, 417]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

elements = [ions+ocec+elementals]
elements_name = ['ions-ocec-elementals']

iteration = 2

for case in Data_Name:

    df = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\' + case + '_raw.csv')
    eraser = df.sample(int(len(df) * 0.2), random_state=seeds[s]).index

    print(case, len(df), len(df)-len(eraser), len(eraser))


    scalingfactor = {}
    data_scaled = df.copy()

    for c in df.columns[1:]:
        denominator = df[c].max() - df[c].min()
        scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
        data_scaled[c] = (df[c] - df[c].min()) / denominator

    data_wodate_scaled = data_scaled.iloc[:, 1:]

    for s in range(len(seeds)):
        for ele in range(len(elements)):
            for iter in range(iteration):

                name = case + '_result_'+ str(seeds[s])+'_DNN_'+str(elements_name[ele])+'_2nd_'+str(iter+1)

                eraser = df.sample(int(len(df)*0.2), random_state=seeds[s]).index
                target = elements[ele]

                x_train = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser], columns=target))
                y_train = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, target]
                y_train = np.array(y_train)

                val_splitrate = 0.2
                vals = np.random.choice(x_train.shape[0], int(x_train.shape[0] * val_splitrate), replace=False)
                vals.sort()

                x_val = x_train[vals]
                y_val = y_train[vals]

                x_train = np.delete(x_train, [vals], axis=0)
                y_train = np.delete(y_train, [vals], axis=0)

                x_test = np.array(data_wodate_scaled.loc[eraser].drop(columns=target))
                y_test = np.array(data_wodate_scaled.loc[eraser, target])


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
                                                     #activation=hp.Choice('activation_function',
                                                     #                      values=[
                                                     #                              'relu',
                                                     #                              'tahn'
                                                     #                             ])))
                        model.add(keras.layers.Dropout(hp.Choice('dropout_rate',
                                                                 values=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])))

                    model.add(keras.layers.Dense(units=y_train.shape[1],
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
                                    max_epochs=100,
                                    factor=3,
                                    directory='D:/kerastuner',
                                    project_name='D:/kerastuner/'+name)



                # tuner = kt.BayesianOptimization(model_builder,
                #                                 objective='val_accuracy',
                #                                 max_trials=100,
                #                                 directory='D:/kerastuner',
                #                                 project_name='D:/kerastuner/'+name)

                class ClearTrainingOutput(tf.keras.callbacks.Callback):
                    def on_train_end(*args, **kwargs):
                        IPython.display.clear_output(wait=True)


                tuner.search(x_train, y_train, epochs=100, validation_data=(x_val, y_val), verbose=1,
                             callbacks=[ClearTrainingOutput()])

                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

                tuner.search_space_summary()
                tuner.results_summary()


                # Build the model with the optimal hyperparameters and train it on the data
                model = tuner.hypermodel.build(best_hps)

                history = model.fit(x_train, y_train, epochs = 200, validation_data = (x_val, y_val), verbose=1)

                y_predicted = model.predict(x_test)
                evaluation = model.evaluate(x_test, y_test)

                # For making total results


                print(name+'done!')
                print(f"""
                The hyperparameter search is complete. 
                The optimal number of layers: {best_hps.get('num_layers')}
                The optimal number of units: {best_hps.get('units')}
                The optimal learning rate: {best_hps.get('learning_rate')}.
                The optimal dropout rate: {best_hps.get('dropout_rate')}.
                R2 = {evaluation[1]}.
                """)


                f = open(name+'.txt', 'w')
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

                y_predicted_total = model.predict(np.array(data_wodate_scaled.drop(columns=target)))
                y_predicted_total = pd.DataFrame(y_predicted_total, columns=target)

                # rescaling
                # x = x' * (max-min) + min
                # saving scaling factor in [max-min, min, max]

                for c in y_predicted_total:
                    y_predicted_total[c] = y_predicted_total[c] * scalingfactor[c][0] + scalingfactor[c][1]

                y_predicted_total.to_csv(name+'.csv', index=False)

                del model
                del history
                shutil.rmtree('D:/kerastuner/'+name)