import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('nanocav_data.csv')

dados = pd.DataFrame(data, columns=['a', 'rx', 'ry', 'a_min', 'rx_min',
                                    'ry_min', 'del_a', 'del_rx', 'del_ry',
                                    'del_lc', 'V', 'Q', 'l_0', 'F_P'])
indexNames = dados[dados['Q'] == 0].index
dados.drop(indexNames, inplace=True)
print(dados.describe())  # se usar list() podemos ver todos os noemas das colun

# Inputs
a = dados.iloc[:, 0].values
rx = dados.iloc[:, 1].values
ry = dados.iloc[:, 2].values
a_min = dados.iloc[:, 3].values
rx_min = dados.iloc[:, 4].values
ry_min = dados.iloc[:, 5].values
del_a = dados.iloc[:, 6].values
del_rx = dados.iloc[:, 7].values
del_ry = dados.iloc[:, 8].values
del_lc = dados.iloc[:, 9].values
v = dados.iloc[:, 10].values
l_0 = dados.iloc[:, 12].values
f_p = dados.iloc[:, 13].values
q = dados.iloc[:336, 11].values


a = a.reshape(-1, 1)
rx = rx.reshape(-1, 1)
ry = ry.reshape(-1, 1)
a_min = a_min.reshape(-1, 1)
rx_min = rx_min.reshape(-1, 1)
ry_min = ry_min.reshape(-1, 1)
del_a = del_a.reshape(-1, 1)
del_rx = del_rx.reshape(-1, 1)
del_ry = del_ry.reshape(-1, 1)


scaler = StandardScaler()

a = scaler.fit_transform(a)
rx = scaler.fit_transform(rx)
ry = scaler.fit_transform(ry)
a_min = scaler.fit_transform(a_min)
rx_min = scaler.fit_transform(rx_min)
ry_min = scaler.fit_transform(ry_min)
del_a = scaler.fit_transform(del_a)
del_rx = scaler.fit_transform(del_rx)
del_ry = scaler.fit_transform(del_ry)


train_data = dados.iloc[:336, :10].values
test_data = dados.iloc[336:, :10].values

triad1 = train_data[0, :10].shape

print(triad1)

'''
print(max(a), max(rx), max(ry), max(a_min), max(rx_min), max(ry_min),
      max(del_a), max(del_rx), max(del_ry), max(del_lc),
      max(v), max(l_0), max(f_p), max(q)/11e4)

'''

'''
a = 349.29235581288094 rx =0.3498961860412208
ry = 0.3497013972737749 a_min = 339.5738857992393
rx_min = 0.3495404963332031 ry_min = 0.3496492295640105
del_a = 3.494963988065514 del_rx = 3.499384217027917
del_ry = 3.499269633708643 del_lc = 247.21149215115133
v = 1.3862880226622197e-19 l_0 = 9.666367044134983e-07
f_ p = 258.4610370056372 q = 109266.74541285008
'''
