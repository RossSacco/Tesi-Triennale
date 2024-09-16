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
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from scikeras.wrappers import KerasClassifier
import warnings

# Disable all warnings
warnings.filterwarnings("ignore")

base_path = '/Users/rossana/Desktop/tesi/'

param_grid = {
        'model__learning_rate': [0.01, 0.005, 0.1],
        'model__momentum': [0, 0.8, 0.9]
    }


def create_model(learning_rate, momentum):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=(n_points, n_series))) #64 filtri lunghi 10 elementi
    model.add(MaxPooling1D(pool_size=3))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    sgd = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


for i in range(1, 27):  #numero di file csv che dobbiamo valutare
    file_name = glob(base_path + 'data/subject_' + str(i) + '_MADts_*.csv')[0]
    X = pd.read_csv(file_name).values

    # Normalizzazione e split
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Dimensioni dell'input e reshape dei dati per la CNN 1D, considera che la prima meta' dei samples sono peak
    # e il resto no
    n_series = int(file_name[-6:-4])  # Numero di serie temporali per gruppo
    n_samples = int(file_name[-10:-7])  # Numero di gruppi (samples peak e no-peak)
    n_points = int(file_name[-20:-17])  # Numero di punti temporali per serie

    X = X.reshape(n_samples, n_points, n_series)
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2)).astype(int)

    test_idx_end = int((n_samples // 2) + int(n_samples * 0.1))
    test_idx_begin = int((n_samples // 2) - int(n_samples * 0.1))

    # Create a boolean mask to select elements to keep for training and testing sets
    mask_train = np.ones(X.shape[0], dtype=bool)
    mask_train[test_idx_begin:test_idx_end] = False
    X_train = X[mask_train, :, :]
    y_train = y[mask_train]

    mask_test = np.zeros(X.shape[0], dtype=bool)
    mask_test[test_idx_begin:test_idx_end] = True
    X_test = X[mask_test, :, :]
    y_test = y[mask_test]

    # Costruzione del modello

    model1 = KerasClassifier(model=create_model, epochs=150, batch_size=8, verbose=0)

    grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)

    print("Best_%d: %f using %s" % (i, grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    with open(base_path + "results/gridSearch_IperSGD_2BS_%s.txt" % (time.strftime("%Y%m%d")), "a",
        encoding="utf-8") as file_object:
        file_object.write("\nBest_%d: %f using %s\n" % (i, grid_result.best_score_, grid_result.best_params_))
        # Stampa i punteggi medi, gli scarti standard e i parametri per ciascuna configurazione
        for mean, stdev, param in zip(means, stds, params):
            file_object.write("%f (%f) with: %r\n" % (mean, stdev, param))


