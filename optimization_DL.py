import pandas as pd
import tensorflow.keras as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

dataall = pd.read_csv('nanocav_datafull.csv')  # ! 2391 dados positivos
# data2 = pd.read_csv('nanocav_zeradas.csv')  # ! 4566 dados negativos

# ? Retorna uma quantidade específica de dados negativos
# data2.sample(2391)

'''
Será que seria viável classificar os dados em apenas dois conjuntos ?
Um conjunto onde a saída é positiva e a outra é zerada ?
'''
# dataall = pd.merge(data1, data2, how='outer')

# ? Aleatoriza o conjunto de dados
# dataall.sample(frac=1).reset_index(drop=True)
# print(dataall)
#!6957 amostras com 2391 positivas

normalization = 320e-9/10e4


def inoutputs(quantity):
    # -- Inputs
    a = dataall.iloc[:, 0].values*normalization
    rx = dataall.iloc[:, 1].values*normalization
    ry = dataall.iloc[:, 2].values*normalization
    a_min = dataall.iloc[:, 3].values*normalization
    rx_min = dataall.iloc[:, 4].values*normalization
    ry_min = dataall.iloc[:, 5].values*normalization
    del_a = dataall.iloc[:, 6].values*normalization
    del_rx = dataall.iloc[:, 7].values*normalization
    del_ry = dataall.iloc[:, 8].values*normalization
    del_lc = dataall.iloc[:, 9].values*normalization

# -- Outputs
    v = dataall.iloc[:, 10].values*normalization
    l_0 = dataall.iloc[:, 12].values*normalization
    f_p = dataall.iloc[:, 13].values*normalization

    q = np.log10(dataall.iloc[:quantity, 11].values)
    q_test = np.log10(dataall.iloc[quantity:, 11].values)

# -- Formatação dos dados em formato 2 canais 5x3 ixj
    triad = np.zeros((a.size, 4, 3, 3))
    cont = 0
    for d in range(a.size):
        r = np.diag([a[cont], a_min[cont], del_a[cont]])
        g = np.diag([rx[cont], rx_min[cont], del_rx[cont]])
        b = np.diag([ry[cont], ry_min[cont], del_ry[cont]])
        l = np.diag([del_lc[cont], 1, 1])
        triad[cont, 0, :, :] = r
        triad[cont, 1, :, :] = g
        triad[cont, 0, :, :] = b
        triad[cont, 1, :, :] = l
        cont += 1

    triad_train = triad[:quantity, :, :, :]
    triad_test = triad[quantity:, :, :, :]

    return triad_train, triad_test, q, q_test


triad_train, triad_test, q, q_test = inoutputs(2200)
triad1 = triad_train[0, :, :, :]

#! Área de rede neural

model = tf.models.Sequential([
    tf.layers.Conv1D(50, 2, input_shape=triad1.shape),
    tf.layers.Flatten(),
    tf.layers.Dense(200, activation='relu'),
    tf.layers.Dropout(0.2),
    tf.layers.Dense(50, activation='relu'),
    tf.layers.Dense(1, activation='relu')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.losses.MeanAbsolutePercentageError(name='MAPE')])

history = model.fit(triad_train, q, epochs=50,
                    validation_data=(triad_test, q_test))

#! Plotagem dos dados analisados pela rede neural


def plot_history(hist):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()

    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.plot(hist['epoch'], hist['val_MAPE'],
             color='r', label='val_mape')
    plt.plot(hist['epoch'], hist['MAPE'],
             label='train_mape')
    plt.legend()

    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'], hist['val_loss'],
             color='g', label='val_mse')
    plt.plot(hist['epoch'], hist['loss'],
             color='y', label='train_mse')

    plt.tight_layout()
    plt.legend()
    plt.show()


plot_history(history)

'''
def valor_min(hist):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    for a in range(len(hist['epoch'])):
        if min(hist['MAPE']) == hist['MA']:
            val = print(hist[a])
        if min(hist['mse']) == hist[a]:
            val1 = print(hist[a])
    return val, val1


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
'''
'''
Existe dados de validação, treinamento e teste.
Validação de quanto a rede neural é boa
E teste para certificar-se que em relação
a outra quantidade a rede neural funciona
'''
