import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow import keras

import IPython
import os
import scipy.stats
import keras_tuner as kt


data_na = pd.read_csv('data\\Data_PM25_speciation_Seoul_horuly_withNa.csv')
data = data_na.dropna().reset_index(drop=True)

data['wd_g'] = data['wd_g'] - 180
data['wd_s'] = data['wd_s'] - 180

# EDA
'''
a,b=np.polyfit(data['PM2.5'], data['pm25_jg'],1)
plt.scatter(data['PM2.5'], data['pm25_jg'])
plt.plot(data['PM2.5'], a*data['PM2.5']+b)
plt.show()
print(a,b)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data['PM2.5'], data['pm25_jg'])
print(r_value**2)
'''
# Data normalization
# MinMaxScaler -> x' = (x-min)/(max-min)

# saving scaling factor in [max-min, min, max]

scalingfactor = {}
data_scaled = data.copy()

for c in data.columns[1:]:
    denominator = data[c].max()-data[c].min()
    scalingfactor[c] = [denominator, data[c].min(), data[c].max()]
    data_scaled[c] = (data[c] - data[c].min())/denominator


# Chemical species prediction
# 3 cases
# 1. for ions
# 2. for OC/EC
# 3. for elements

# for ramdom selection

na_rate = 0.2
n = data.shape[0]

eraser = np.random.choice(n,int(n*na_rate), replace=False)
eraser.sort()

# for elementals

elementals = ['Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Ba', 'Pb']
data_elementals = data_scaled.copy()
data_elementals_answers = data_elementals.loc[eraser, elementals]
data_elementals.loc[eraser, elementals] = np.nan

ions = ['SO42.', 'NO3.', 'Cl.', 'Na.', 'NH4.', 'K.', 'Mg2.', 'Ca2.']
data_ions = data_scaled.copy()
data_ions_answers = data_ions.loc[eraser, ions]
data_ions.loc[eraser, ions] = np.nan

# for OC/EC

ocec = ['OC', 'EC']
data_ocec = data_scaled.copy()
data_ocec_answers = data_ocec.loc[eraser, ocec]
data_ocec.loc[eraser, ocec] = np.nan

# 2. DNN


data_wodate_scaled = data_scaled.iloc[:, 1:]
x_train = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser],
                                           columns=elementals))
y_train = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, elementals]
y_train = np.array(y_train)

val_splitrate = 0.15
vals = np.random.choice(x_train.shape[0], int(x_train.shape[0] * val_splitrate), replace=False)
vals.sort()

x_val = x_train[vals]
y_val = y_train[vals]

x_train = np.delete(x_train, [vals], axis=0)
y_train = np.delete(y_train, [vals], axis=0)

x_test = np.array(data_wodate_scaled.loc[eraser].drop(columns=elementals))
y_test = np.array(data_wodate_scaled.loc[eraser, elementals])


def model_builder(hp):
    model = keras.Sequential()

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512

    for i in range(hp.Int('num_layers', 2, 15)):
        model.add(keras.layers.Dense(units=hp.Int('units',
                                                  min_value=512,
                                                  max_value=2048,
                                                  step=32),
                                     activation=hp.Choice('activation_function',
                                                          values=[
                                                                  'relu',
                                                                  'tanh',
                                                                  'sigmoid'])))
        model.add(keras.layers.Dropout(hp.Choice('dropout_rate',
                                                 values=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])))

    model.add(keras.layers.Dense(units=y_train.shape[1],
                                 activation=hp.Choice('activation_function',
                                                      values=[
                                                              'relu',
                                                              'tanh',
                                                              'sigmoid'])))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-5, 1e-5])

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
                                max_trials=100,
                                directory='D:/kerastuner',
                                project_name='D:/kerastuner/Bayesian_elementals')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(x_train, y_train, epochs=150, validation_data=(x_val, y_val),
             callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

tuner.search_space_summary()
tuner.results_summary()


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs = 200, validation_data = (x_val, y_val))


print(f"""
The hyperparameter search is complete. 
The optimal number of layers: {best_hps.get('num_layers')}
The optimal number of units: {best_hps.get('units')}
The optimal learning rate: {best_hps.get('learning_rate')}.
The optimal dropout rate: {best_hps.get('dropout_rate')}.
The optimal activation function: {best_hps.get('activation_function')}.
""")

y_predicted = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)

# rescaling
# x = x' * (max-min) + min
# saving scaling factor in [max-min, min, max]
data_elementals_predicted = data_elementals_answers.copy()
data_elementals_predicted.iloc[:, :] = y_predicted

for c in data_elementals_answers.columns:
    data_elementals_answers[c] = data_elementals_answers[c] * scalingfactor[c][0] + scalingfactor[c][1]
    data_elementals_predicted[c] = data_elementals_predicted[c] * scalingfactor[c][0] + scalingfactor[c][1]

DNN_results_elementals_r2 = []
for i in data_elementals_answers.columns:
    x = data_elementals_answers[i]
    y = data_elementals_predicted[i]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    DNN_results_elementals_r2.append(r_value ** 2)
DNN_results_elementals_r2 = np.array(DNN_results_elementals_r2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy_ions')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss_ions')
plt.legend()
plt.show()

#pd.DataFrame(DNN_results_elementals_r2).to_clipboard(index=False)







### Ions

data_wodate_scaled = data_scaled.iloc[:, 1:]
x_train = np.array(data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser],
                                           columns=ions))
y_train = data_wodate_scaled.drop(index=data_wodate_scaled.index[eraser]).loc[:, ions]
y_train = np.array(y_train)

val_splitrate = 0.15
vals = np.random.choice(x_train.shape[0], int(x_train.shape[0] * val_splitrate), replace=False)
vals.sort()

x_val = x_train[vals]
y_val = y_train[vals]

x_train = np.delete(x_train, [vals], axis=0)
y_train = np.delete(y_train, [vals], axis=0)

x_test = np.array(data_wodate_scaled.loc[eraser].drop(columns=ions))
y_test = np.array(data_wodate_scaled.loc[eraser, ions])


def model_builder(hp):
    model = keras.Sequential()

    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512

    for i in range(hp.Int('num_layers', 2, 15)):
        model.add(keras.layers.Dense(units=hp.Int('units',
                                                  min_value=512,
                                                  max_value=2048,
                                                  step=32),
                                     activation=hp.Choice('activation_function',
                                                          values=[
                                                                  'relu',
                                                                  'tanh',
                                                                  'sigmoid'])))
        model.add(keras.layers.Dropout(hp.Choice('dropout_rate',
                                                 values=[0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2])))

    model.add(keras.layers.Dense(units=y_train.shape[1],
                                 activation=hp.Choice('activation_function',
                                                      values=[
                                                              'relu',
                                                              'tanh',
                                                              'sigmoid'])))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 5e-5, 1e-5])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mse',
                  metrics=['accuracy'])

    return model


#tuner = kt.Hyperband(model_builder,
#                     objective='val_accuracy',
#                     max_epochs=200,
#                     factor=3,
#                     directory='kerastuner',
#                     project_name='hyperband_ions')

tuner = kt.BayesianOptimization(model_builder,
                                objective='val_accuracy',
                                max_trials=100,
                                directory='D:/kerastuner',
                                project_name='D:/kerastuner/Bayesian_ions')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(x_train, y_train, epochs=150, validation_data=(x_val, y_val),
             callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

tuner.search_space_summary()
tuner.results_summary()


# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

history = model.fit(x_train, y_train, epochs = 200, validation_data = (x_val, y_val))


print(f"""
The hyperparameter search is complete. 
The optimal number of layers: {best_hps.get('num_layers')}
The optimal number of units: {best_hps.get('units')}
The optimal learning rate: {best_hps.get('learning_rate')}.
The optimal dropout rate: {best_hps.get('dropout_rate')}.
The optimal activation function: {best_hps.get('activation_function')}.
""")

y_predicted = model.predict(x_test)
evaluation = model.evaluate(x_test, y_test)

# rescaling
# x = x' * (max-min) + min
# saving scaling factor in [max-min, min, max]
data_ions_predicted = data_ions_answers.copy()
data_ions_predicted.iloc[:, :] = y_predicted

for c in data_ions_answers.columns:
    data_ions_answers[c] = data_ions_answers[c] * scalingfactor[c][0] + scalingfactor[c][1]
    data_ions_predicted[c] = data_ions_predicted[c] * scalingfactor[c][0] + scalingfactor[c][1]

DNN_results_ions_r2 = []
for i in data_ions_answers.columns:
    x = data_ions_answers[i]
    y = data_ions_predicted[i]
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    DNN_results_ions_r2.append(r_value ** 2)
DNN_results_ions_r2 = np.array(DNN_results_ions_r2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.figure()
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy_ions')
plt.legend()
plt.show()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss_ions')
plt.legend()
plt.show()

#pd.DataFrame(DNN_results_ions_r2).to_clipboard(index=False)


