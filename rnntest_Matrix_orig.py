import pandas as pd
import tensorflow.keras as tfa
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data1 = pd.read_csv('nanocav_datafull.csv')  # ! 2391 dados positivos
# data2 = pd.read_csv('nanocav_zeradas.csv')  # ! 4566 dados negativos

# ? Retorna uma quantidade específica de dados negativos
# data2.sample(2391)

'''
Será que seria viável classificar os dados em apenas dois conjuntos ?
Um conjunto onde a saída é positiva e a outra é zerada ?
'''
#dataall = pd.merge(data1, data2, how='outer')
dataall = data1
# ? Aleatoriza o conjunto de dados
# dataall.sample(frac=1).reset_index(drop=True)
# print(dataall)
#!6957 amostras com 2391 positivas


def inoutputs(quantity):
    #-- Inputs
    a = dataall.iloc[:, 0].values/350
    rx = dataall.iloc[:, 1].values
    ry = dataall.iloc[:, 2].values
    a_min = dataall.iloc[:, 3].values/350
    rx_min = dataall.iloc[:, 4].values
    ry_min = dataall.iloc[:, 5].values
    del_a = dataall.iloc[:, 6].values
    del_rx = dataall.iloc[:, 7].values
    del_ry = dataall.iloc[:, 8].values
    del_lc = dataall.iloc[:, 9].values/350

    #-- Outputs
    v = dataall.iloc[:, 10].values
    l_0 = dataall.iloc[:, 12].values
    f_p = dataall.iloc[:, 13].values

    q = np.log10(dataall.iloc[:quantity, 11].values)
    q_test = np.log10(dataall.iloc[quantity:, 11].values)

    # -- Formatação dos dados em formato 4 canais 3x3 ixj
    triad = np.zeros((a.size, 4, 3, 3))
    cont = 0
    for d in range(a.size):
        r = np.diag([a[cont], a_min[cont], del_a[cont]])
        g = np.diag([rx[cont], rx_min[cont], del_rx[cont]])
        b = np.diag([ry[cont], ry_min[cont], del_ry[cont]])
        l = np.diag([del_lc[cont], 1, 1])
        triad[cont, 0, :, :] = r
        triad[cont, 1, :, :] = g
        triad[cont, 2, :, :] = b
        triad[cont, 3, :, :] = l
        cont += 1

    triad_train = triad[:quantity, :, :, :]
    triad_test = triad[quantity:, :, :, :]

    return triad_train, triad_test, q, q_test


triad_train, triad_test, q, q_test = inoutputs(2100)
triad1 = triad_train[0, :, :, :]


model = tfa.models.Sequential([
    tfa.layers.Conv1D(50, 2, input_shape=triad1.shape),
    tfa.layers.Flatten(),
    tfa.layers.Dense(200, activation='relu'),
    tfa.layers.Dropout(0.2),
    tfa.layers.Dense(50, activation='relu'),
    # tfa.layers.Dropout(0.2),
    tfa.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

history = model.fit(triad_train, q, epochs=100,
                    validation_data=(triad_test, q_test))


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()

    plt.subplot(221)
    plt.xlabel('Epoch')
    plt.ylabel('VAL_MAE')
    plt.plot(hist['epoch'], hist['val_mae'],
             color='r')

    plt.subplot(222)
    plt.xlabel('Epoch')
    plt.ylabel('TRAIN_MAE')
    plt.plot(hist['epoch'], hist['mae'])

    plt.subplot(223)
    plt.xlabel('Epoch')
    plt.ylabel('VAL_MSE')
    plt.plot(hist['epoch'], hist['val_loss'],
             color='g')

    plt.subplot(224)
    plt.xlabel('Epoch')
    plt.ylabel('TRAIN_MSE')
    plt.plot(hist['epoch'], hist['loss'],
             color='y')
    plt.tight_layout()
    plt.legend()
    plt.show()

    mae, valmae = min(hist['val_mae']), min(hist['mae'])
    mse, valmse = min(hist['val_loss']), min(hist['loss'])

    print(
        f'mae  = {mae:.4f}, val_mae = {valmae:.4f}, mse = {mse:.4f}, val_mse = {valmse:.4f}')


plot_history(history)
