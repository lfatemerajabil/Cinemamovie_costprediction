import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import statistics as st
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
d = pd.read_csv('C:/Users/asus/Desktop/datamining/Movieregression.csv')
data = d.to_numpy()
l = len(data[0])
genre = []
col = []
data = np.delete(data , 2 , axis=0)
collection = np.array(data[:,17]).copy()
#draw chart
'''leng = []
for i in range(505):
    leng.append(i)
for i in range(18):
    print(len(data[:,i]))
    plt.bar(leng , list(data[:,i]))
    plt.savefig('aks%i.png'%i)
'''
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

#calcute parameter array with LinearAlgebra
parameter = np.linalg.pinv(data)@collection

#predict
r = [492.964 , 91.2 , 0.329 , 3544.9 , 169.7 , 9.24 , 9.39 , 9.235 , 9.365 , 6.96 , 359760 , 1 , 282.099 , 54 , 6.32 , 1 , 0 , 1 , 0 , 0]
r380 = np.array(r)
pred = []
for i in data:
    pred.append(i@parameter)
print("parameter array:\n",parameter)
print("\nexample:predicted collection for row380:",r380@parameter)
print("\nbut real value of this collection is: 26200")
result = pd.DataFrame({'test':collection , 'LR':pred})
print(result)
#error
for i in range(len(pred)):
    if(abs(collection[i] - pred[i]) < 1):
        print("yes")
        print(i)
        print(pred[i])
        print(collection[i])
print("error:",mean_squared_error(pred,collection))
print("error:",mean_absolute_error(pred,collection))