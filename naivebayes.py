from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

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
mean = []
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

#Bias
for row in data:
    row[15] = 1.0

data = data.astype(np.float)
collection = collection.astype(np.float)


#for avoiding random prediction we should define random_state = integer
X_train, X_test, y_train, y_test = train_test_split(data, collection, test_size=0.1, random_state=0)
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train).predict(X_test)
result = pd.DataFrame({'test':y_test , 'NB':gnb})
print(result)
print("error:",mean_squared_error(y_test,gnb))
print("error:",mean_absolute_error(y_test,gnb))
for i in range(len(gnb)):
    if(abs(y_test[i] - gnb[i]) < 1):
        print("yes")
        print(gnb[i],y_test[i])