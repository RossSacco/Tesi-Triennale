import time
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, BatchNormalization, ReLU, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.interpolate import interp1d
import warnings
import math

# Disable all warnings
warnings.filterwarnings("ignore")

base_path = '/Users/rossana/Desktop/tesi/'

def ridimensiona_righe(data, new_length):
    # Crea un array di valori x
    x = np.arange(len(data))

    # Calcola la nuova lunghezza dell'array x
    new_x = np.linspace(0, len(data) - 1, new_length)

    # Interpolazione dei dati lungo l'asse delle x
    f = interp1d(x, data, axis=0)

    # Valori interpolati per la nuova lunghezza
    new_data = f(new_x)

    return new_data


n_series = 200

for n in range(1, 27):
    #trovo i canali minimi
    file_name = glob(base_path + 'data/subject_' + str(n) + '_MADts_*.csv')[0]
    n_series = n_series if int(file_name[-6:-4]) > n_series else int(file_name[-6:-4])# Numero di serie temporali per gruppo**


def LOSO_function(escluso):

    n_points_glb = 150
    scaler = StandardScaler()

    train_data = np.empty((0, n_points_glb, n_series), dtype=float)
    test_data = np.empty((0, n_points_glb, n_series), dtype=float)

    label_train = np.empty((0,), dtype=int)
    label_test = np.empty((0,), dtype=int)

    for k in range(1, 27):

        file_name = glob(base_path + 'data/subject_' + str(k) + '_MADts_*.csv')[0]

        data = pd.read_csv(file_name).values

        n_samples = int(file_name[-10:-7])

        dim_des = n_samples * n_series
        data = ridimensiona_righe(data, dim_des)
        data = scaler.fit_transform(data)
        data = data.reshape(n_samples, n_points_glb, n_series)

        label = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2)).astype(int)

        if k != escluso:
            label_train = np.concatenate((label_train, label))
            train_data = np.vstack((train_data, data))
        else:
            label_test = label
            test_data = data

    return train_data, test_data, label_train, label_test, n_points_glb, n_series


for i in range(9, 27):  #numero di file csv che dobbiamo valutare, qui i sarà l'escluso
    for j in range(1, 5): #10 prove ciascuno

        X_train, X_test, y_train, y_test, n_points, n_series = LOSO_function(i)

    # Costruzione del modello
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=10,activation='relu', input_shape=(n_points, n_series))) #64 filtri lunghi 10 elementi
        model.add(MaxPooling1D(pool_size=3))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    #sgd = SGD(learning_rate=0.005, momentum=0.8)
    #adam_opt = Adam(learning_rate=0.005)
        for layer in model.layers:
            layer.trainable = True

        sgd = SGD(learning_rate=0.01, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # Training e testing
        model.fit(X_train, y_train, epochs=150, batch_size=8, validation_data=(X_test, y_test), verbose=0)


    #da qui in poi FINETUNING
        file_name = glob(base_path + 'data/subject_' + str(i) + '_MADts_*.csv')[0]
        n_samples = int(file_name[-10:-7])

        test_idx_end = int((n_samples // 2) + int(n_samples * 0.20))
        test_idx_begin = int((n_samples // 2) - int(n_samples * 0.20))

        mask_train = np.ones(X_test.shape[0], dtype=bool)
        mask_train[test_idx_begin:test_idx_end] = False
        X_FT_Train = X_test[mask_train, :, :]
        y_FT_Train = y_test[mask_train]

        mask_test = np.zeros(X_test.shape[0], dtype=bool)
        mask_test[test_idx_begin:test_idx_end] = True
        X_FT_Test = X_test[mask_test, :, :]
        y_FT_Test = y_test[mask_test]


        #CONGELO TUTTI I LAYER TRANNE 6 E 7
        for layer in model.layers[:5]:
            layer.trainable = False

        sgd = SGD(learning_rate=0.01, momentum=0.8)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(X_FT_Train, y_FT_Train, epochs=10, batch_size=8, validation_data=(X_FT_Test, y_FT_Test), verbose=0)
        #VEDO LE PERFORMANCE

        loss, accuracy = model.evaluate(X_FT_Test, y_FT_Test)

        res = 'Subject_test: ' + str(i) + ', accuracy: ' + str(accuracy) + '\n'

        print(res)

        with open(base_path + "results/performanceTS_LOSO_FineTuning_20%.txt", "a",
                    encoding="utf-8") as file_object:
            file_object.write(res)
            file_object.close()
