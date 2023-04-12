import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataall = pd.read_csv('nanocav_datafull.csv', index_col=False)

a = dataall.iloc[:, 1].values / 350
q = dataall.iloc[:, 12].values
q = np.log10(q)

plt.plot(q, a, 'bo')
plt.xlabel('Parâmetro a')
plt.ylabel('Fator de qualidade (Q)')
plt.title('Parâmetro a x Fator Q')
plt.tight_layout()
plt.show()

X = np.array(list(zip(q, a))).reshape(len(q), 2)
print(X)
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)

    # distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
    #                                     'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    # mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
    #                                'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping2.items():
    print(f'{key} : {val}')

plt.plot(K, inertias, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('Inertias')
plt.title('Método Elbow usando Inertias')
plt.show()

# for key, val in mapping1.items():
#     print(f'{key} : {val}')

# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Valores de K')
# plt.ylabel('Distortion')
# plt.title('Método Elbow usando Distortion')
# plt.show()

kmeans = KMeans(n_clusters=3,  # numero de clusters
                # algoritmo que define a posição dos clusters de maneira mais assertiva
                init='k-means++', n_init=10,
                max_iter=300)  # numero máximo de iterações
pred_y = kmeans.fit_predict(X)

plt.scatter(q, a, c=pred_y)  # posicionamento dos eixos x e y
plt.xlabel('Parâmetro a ')
plt.ylabel('Fator Q')
plt.title(' Clusters da distribuição')
plt.grid()  # função que desenha a grade no nosso gráfico
# posição de cada centroide no gráfico
# plt.scatter(kmeans.cluster_centers_[:, 1],
#             kmeans.cluster_centers_[:, 0], s=70, c='red')
plt.show()
