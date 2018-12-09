import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

#Data Processing

#Import the data, fill in the NaN, and drop unusable columns
data = pd.read_csv('titanic/train.csv')
data = data.fillna(0)
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

#Convert categorical columns to integer codes
data.Sex = pd.Categorical(data.Sex)
data.Sex = data.Sex.cat.codes
data.Embarked = pd.Categorical(data.Embarked)
data.Embarked = data.Embarked.cat.codes

#Get the passenger ids and drop the column
passID = np.array(data.PassengerId)
data = data.drop(['PassengerId'], axis=1)

#Get the prediction column and drop it
#We will be predicting if a passenger survived or not
Y = np.array(data.Survived)
data = data.drop(['Survived'], axis=1)

#Get the input data
X = np.array(data)

#Normalize Data (Increases speed drastically and Seems to increase accuracy as well)
scaler = MinMaxScaler()  
X = scaler.fit_transform(X)


#Cross Validation Training

#KNN
print ("\n######### KNN CROSS VAL ####################\n")
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
y_train = y_train.ravel()
y_test = y_test.ravel()
for K in range(25):
    K_value = K+1
    neigh = KNeighborsClassifier(n_neighbors = K_value, weights='uniform', algorithm='auto')
    neigh.fit(X_train, y_train) 
    y_pred = neigh.predict(X_test)
    print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for K-Value:",K_value)


#Gaussian SVM
print ("\n######### Gaussian SVM CROSS VAL ####################\n")
for C in range(15):
    C_value = C+1
    clf = svm.SVC(kernel='rbf', C=C_value,gamma ='auto')
    clf.fit(X_train, y_train) 
    y_pred = clf.predict(X_test)
    print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for C Value:",C_value)

#Polynomial SVM
print ("\n######### Polynomial SVM ####################\n")
for d in range (7):
    print ("\nDegree = ", d)
    for C in range(10):
        C_value = C+1
        clf = svm.SVC(kernel='poly',degree = d, C=C_value,gamma ='auto')
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        print ("Accuracy is ", accuracy_score(y_test,y_pred)*100,"% for C Value:",C_value,"and degree: ",d)


