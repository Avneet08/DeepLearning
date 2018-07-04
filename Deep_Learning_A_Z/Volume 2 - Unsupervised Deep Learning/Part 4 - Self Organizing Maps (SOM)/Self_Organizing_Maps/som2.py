# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:02:24 2018

@author: Avneet
"""

# importing the librariers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# importing  datasey
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# the som starts learning
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
# inintialize the weights of som randomly
som.random_weights_init(X)
# train som on X
som.train_random(data = X, num_iteration = 100)




# visulizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
# now we will create some red circles and green sq
# red circles are customers who didnt get appproval
# green sq are customers who got approval

markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# finding the frauds
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
    






