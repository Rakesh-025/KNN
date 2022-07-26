#KNN model for zoo data set
#Loading the packages into python

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Loading the file into Python
zoo = pd.read_csv(r"C:\Users\kaval\OneDrive\Desktop\Assignments\Assignment Datasets\KNN Model\Zoo.csv")
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data set
zoo_n = norm_func(zoo.iloc[:, 1:16])
zoo_n.describe()

X = np.array(zoo_n.iloc[:,:]) # Predictors 
Y = np.array(zoo['type']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 44)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

# Evaluating the Model 
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions']) 


# Calculating the Error on train data
pred_train = knn.predict(X_train)
print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions']) 

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])

#Visualizing the data
import matplotlib.pyplot as plt  

# Accuracy plot of the training data 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# Accuracy plot of the testing data
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")














