from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
d = pd.read_csv('C:/Users/asus/Desktop/datamining/Movieregression.csv')
data = d.to_numpy()
l = len(data[0])
genre = []
col = []
data = np.delete(data , 2 , axis=0)
collection = np.array(data[:,17]).copy()

#normalization
'''mean = []
sd = []
for i in range(18):
    if((i != 12)and(i != 14)and(i != 11)and(i != 17)):
        mean.append(st.mean(data[: , i]))
        sd.append(st.stdev(data[: , i]))
    else:
        mean.append(0)
        sd.append(1)

for i in range(18):
    if((i != 12)and(i != 14)and(i != 11)and(i != 17)):
        for j in range(len(data[: , i])):
            data[j , i] = (data[j , i]-mean[i])/sd[i]
'''
#numericalize Genre column
for row in data:
    if(row[14] not in genre):
        genre.append(row[14])
    col.append([0.0])
genre_col = np.array(col)
for i in range(len(genre)):
    data = np.append(data , genre_col , axis=1)
a = 0.0
for row in data:
    index = genre.index(row[14])
    row[l+index] = 1.0
    a += 1.0
data = np.delete(data , 14 , axis=1)

#delete Time-Taken feature
data = np.delete(data , 12 , axis=1)

#numericalize 3D_dimension column
for row in data:
    row[row == 'YES'] = 1.0
    row[row == 'NO'] = 0.0
    row = row.astype(np.float)

data = data.astype(np.float)
collection = collection.astype(np.float)

two_d_data = data[: , :]

model = DBSCAN(eps = 21500, min_samples = 19)
model = model.fit(two_d_data)

clusters = model.labels_

colors = clusters

plt.figure()

plt.scatter(two_d_data[:, 3], two_d_data[:, 10], c = colors, marker = 'o')
plt.savefig('2dd.png')
print(silhouette_score(two_d_data, clusters))

fig = plt.figure()
ax = Axes3D(fig)
three_d_data = data[:, :]
three_d_data = np.array(three_d_data, dtype=np.float64)
model = model.fit(three_d_data)

plt.scatter(three_d_data[:, 3], three_d_data[:, 10], three_d_data[:, 6], c = model.labels_)
plt.savefig('3dd.png')
'''
a = 0
A = []
B = []
C = []

for i in range(2,30):
    
    for j in np.arange(16500,22500,500):
        model = DBSCAN(eps = j,min_samples = i)
        model = model.fit(data)
        clusters = model.labels_
        if(len(np.unique(clusters)) != 1):

            A.append(i)
            B.append(j)
            C.append(silhouette_score(data, clusters))
fig = plt.figure()
ax = Axes3D(fig)
plt.scatter(C , A , B)
plt.savefig('sil3.png')
'''
