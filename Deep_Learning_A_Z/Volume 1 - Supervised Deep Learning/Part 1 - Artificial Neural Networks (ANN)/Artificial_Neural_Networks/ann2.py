# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 18:12:05 2018

@author: Avneet
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Create your classifier here
# Make an ann Import keras lib and its dependens
import keras
from keras.models import Sequential
from keras.layers import Dense
#Initialize the ann by def it as seq of layers
classifier=Sequential()
# add different layer to the ann
# first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

# add more hiden layers
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

# add the output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

# compiling the whole ann 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# fitting ann to training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)


new_pred=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_pred=(new_pred>0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# K4 cross validation
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
# Keras classifier requires a function as an argumen so w e first make that func
def build_classifier():
        
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

#Wrapper for keras to wrap the sklearn inside the keras library
    # new classifier is trained with k4 
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
#accuracy to store 10 dif accuracies return by cross validation function
mean = accuracies.mean()
variance = accuracies.std()




