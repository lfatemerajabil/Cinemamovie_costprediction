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

kmeans = KMeans(n_clusters=2).fit(data)

print('clusters:\n', kmeans.labels_)

print('centroids:\n', kmeans.cluster_centers_)

#2D and 3D plotting of data in clustering

two_d_data = data[: , :]
kmeans = kmeans.fit(two_d_data)

plt.scatter(two_d_data[:, 3], two_d_data[:, 10], c = kmeans.labels_, marker = 'o', s=200)
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c = 'k', marker = 'x', s=500)

plt.savefig('2dk.png')

fig = plt.figure()
ax = Axes3D(fig)

three_d_data = data[:, :]
three_d_data = np.array(three_d_data, dtype=np.float64)
kmeans = kmeans.fit(three_d_data)

plt.scatter(three_d_data[:, 3], three_d_data[:, 10], three_d_data[:, 6], c = kmeans.labels_)
plt.savefig('3dk.png')
#Silhouette metric

print('silhouette:\n', silhouette_score(three_d_data, kmeans.labels_))

'''AA = []
BB = []

for i in range(2 , 50):
    AA.append(i)
    kmeans = KMeans(n_clusters=i).fit(data)
    BB.append(silhouette_score(data, kmeans.labels_))

plt.plot(AA , BB)
plt.xlabel('K')
plt.ylabel('silhouette')
plt.savefig('siloknn.png')
'''
