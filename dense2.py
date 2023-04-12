import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfa
import matplotlib.pyplot as plt


def plot_history(history, x, y):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    valmape, mape = min(hist['val_MAPE']), min(hist['MAPE'])
    valmse, mse = min(hist['val_loss']), min(hist['loss'])

    print(f'mape  = {mape:.2f}%, val_mape = {valmape:.2f}%'
          f', mse = {mse:.4f}, val_mse = {valmse:.4f}')

    plt.figure()

    plt.subplot(211)
    plt.xlabel('Epoch')
    plt.ylabel('MAPE')
    plt.plot(hist['epoch'], hist['val_MAPE'],
             color='r', label='val_mape')
    plt.plot(hist['epoch'], hist['MAPE'],
             label='mape')
    plt.legend()

    plt.subplot(212)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.plot(hist['epoch'], hist['val_loss'],
             color='g', label='val_mse')
    plt.plot(hist['epoch'], hist['loss'],
             color='y', label='mse')

    plt.tight_layout()
    plt.legend()
    plt.show()

    error = np.max(y)/np.min(y)
    plt.scatter(x, y, c=y, cmap='gnuplot2')
    plt.plot(np.linspace(min(y) - error, max(y) + error),
             np.linspace(min(y) - error, max(y) + error))
    plt.title(
        f'Regressão da Previsão x Valor calculado R²= {correlacao:.2%}')
    plt.xlabel('Previsões da rede neural')
    plt.ylabel('FDDT log10(V) calculado')
    plt.legend()
    plt.show()


#!6957 amostras com 2391 positivas
dataall = pd.read_csv('nanocav_datafull.csv')

a = dataall.iloc[:, 1].values/350
dataall = dataall.drop(columns=['a'])
dataall.insert(1, 'a', a)

a_min = dataall.iloc[:, 4].values/350
dataall = dataall.drop(columns=['a_min'])
dataall.insert(4, 'a_min', a_min)

del_a = dataall.iloc[:, 7].values/350
dataall = dataall.drop(columns=['del_a'])
dataall.insert(7, 'del_a', del_a)

del_rx = dataall.iloc[:, 8].values/350
dataall = dataall.drop(columns=['del_rx'])
dataall.insert(8, 'del_rx', del_rx)

del_ry = dataall.iloc[:, 9].values/350
dataall = dataall.drop(columns=['del_ry'])
dataall.insert(9, 'del_ry', del_ry)

del_lc = dataall.iloc[:, 10].values/350
dataall = dataall.drop(columns=['del_lc'])
dataall.insert(10, 'del_lc', del_lc)

# q = np.log10(dataall.iloc[:, 12].values)
# dataall = dataall.drop(columns=['Q'])
# dataall.insert(12, 'Q', q)

q = np.log10(dataall.iloc[:, 12].values)
dataall = dataall.drop(columns=['Q'])
dataall.insert(12, 'Q', q)

v = np.log10(dataall.iloc[:, 12].values)
dataall = dataall.drop(columns=['V'])
dataall.insert(12, 'V', v)

# v = dataall.iloc[:, 11].values
# q = dataall.iloc[:, 12].values
# log_q_v = np.log10(q/v)
# dataall['log_q_v'] = log_q_v

quanti = 2000
inputs_train = dataall.iloc[:quanti, 1:11].values
inputs_test = dataall.iloc[quanti:2291, 1:11].values

outputs_train = dataall.iloc[:quanti, 11].values
outputs_test = dataall.iloc[quanti:2291, 11].values

shape = inputs_train[1, :]

model = tfa.models.Sequential([
    tfa.layers.Dense(200, activation='relu',
                     input_shape=shape.shape),
    # tfa.layers.Dropout(0.2),
    tfa.layers.Dense(50, activation='relu'),
    tfa.layers.Dropout(0.2),
    tfa.layers.Dense(1, activation='relu')
])

model.compile(optimizer='Adam',
              loss='mse',
              metrics=[tf.metrics.MeanAbsolutePercentageError(name='MAPE')])

history = model.fit(inputs_train, outputs_train, epochs=1000,
                    validation_data=(inputs_test, outputs_test))

preditcions = model.predict(dataall.iloc[2291:, 1:11].values)
preditcions = np.array(preditcions)
preditcions = preditcions.reshape(-1, )

fddt_log_v = dataall.iloc[2291:, 11]
respostas = pd.DataFrame(
    {'Predictions': preditcions, 'fddt_log_v': fddt_log_v})

correlacao = respostas['fddt_log_v'].corr(respostas['Predictions'])
print(correlacao)

plot_history(history, respostas['Predictions'], respostas['fddt_log_v'])
