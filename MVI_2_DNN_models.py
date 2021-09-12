import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import IPython
import keras_tuner as kt


case = '1_Basic_2_BR'


df = pd.read_csv('D:\\Dropbox\\패밀리룸\\MVI\\Data\\'+case+'_raw.csv')
scalingfactor = {}
data_scaled = df.copy()

for c in df.columns[1:]:
    denominator = df[c].max()-df[c].min()
    scalingfactor[c] = [denominator, df[c].min(), df[c].max()]
    data_scaled[c] = (df[c] - df[c].min())/denominator

data_wodate_scaled = data_scaled.iloc[:, 1:]

seeds = [777, 1004, 322] #, 224, 417]
#seeds = [777] #, 224, 417]

ions = ['SO42-', 'NO3-', 'Cl-', 'Na+', 'NH4+', 'K+', 'Mg2+', 'Ca2+']
ocec = ['OC', 'EC']
elementals = ['S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Pb']

elements = [ions,ocec,elementals, ions+ocec, ions+elementals, ocec+elementals, ions+ocec+elementals]
elements_name = ['ions', 'ocec','elementals','ion-ocec','ion-elementals','ocec-elementals', 'ions-ocec-elementals']

#elements = [ions+ocec]
#elements_name = ['ion-ocec']

iteration = 1

for s in range(len(seeds)):
    for ele in range(len(elements)):
        for iter in range(iteration):

            name = case + '_result_'+ str(seeds[s])+'_DNN_'+str(elements_name[ele])+'_'+str(iter+2)

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

                for i in range(hp.Int('num_layers', 2, 15)):
                    model.add(keras.layers.Dense(units=hp.Int('units',
                                                              min_value=256,
                                                              max_value=2048,
                                                              step=32),
                                                 activation=hp.Choice('activation_function',
                                                                      values=[
                                                                              'relu',
                                                                              'sigmoid'])))
                    model.add(keras.layers.Dropout(hp.Choice('dropout_rate',
                                                             values=[0.1, 0.12, 0.14, 0.16, 0.18, 0.2])))

                model.add(keras.layers.Dense(units=y_train.shape[1],
                                             activation=hp.Choice('activation_function',
                                                                  values=[
                                                                          'relu',
                                                                          'tanh',
                                                                          'sigmoid'])))

                # Tune the learning rate for the optimizer
                # Choose an optimal value from 0.01, 0.001, or 0.0001
                hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6])

                model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                              loss='mse',
                              metrics=['accuracy'])

                return model


            #tuner = kt.Hyperband(model_builder,
            #                     objective='val_accuracy',
            #                     max_epochs=200,
            #                     factor=3,
            #                     directory='kerastuner',
            #                     project_name='hyperband_elementals')



            tuner = kt.BayesianOptimization(model_builder,
                                            objective='val_accuracy',
                                            max_trials=30,
                                            directory='D:/kerastuner',
                                            project_name='D:/kerastuner/'+name)

            class ClearTrainingOutput(tf.keras.callbacks.Callback):
                def on_train_end(*args, **kwargs):
                    IPython.display.clear_output(wait=True)


            tuner.search(x_train, y_train, epochs=50, validation_data=(x_val, y_val),
                         callbacks=[ClearTrainingOutput()])

            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

            tuner.search_space_summary()
            tuner.results_summary()


            # Build the model with the optimal hyperparameters and train it on the data
            model = tuner.hypermodel.build(best_hps)

            history = model.fit(x_train, y_train, epochs = 120, validation_data = (x_val, y_val))

            y_predicted = model.predict(x_test)
            evaluation = model.evaluate(x_test, y_test)

            # For making total results



            f = open(name+'.txt', 'w')
            f.write(f"""
            The hyperparameter search is complete. 
            The optimal number of layers: {best_hps.get('num_layers')}
            The optimal number of units: {best_hps.get('units')}
            The optimal learning rate: {best_hps.get('learning_rate')}.
            The optimal dropout rate: {best_hps.get('dropout_rate')}.
            The optimal activation function: {best_hps.get('activation_function')}.
            R2 = {evaluation[1]}.
            """)
            f.close()

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
