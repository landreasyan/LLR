from sklearn.datasets.samples_generator import make_circles
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from numpy.linalg import matrix_power
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import linalg
import pandas as pd
import networkx as nx
import numpy as np
import random
import math
import sys, os
sys.path.append("..")
from llr import LLR


var = 0.5;
n_samples = 100
mu = 1
v = 1
weight_limit = 0
size = 240

n_samples = 100
#beta [0 1]
x_1, y = make_blobs(n_samples=n_samples, n_features=2, centers = [[0., 0.]], cluster_std=1.0)
#beta [1 0]
x_2, y = make_blobs(n_samples=n_samples, n_features=2, centers=[[5., 5.]], cluster_std=1.0)
#beta [1 1]
x_3, y = make_blobs(n_samples=n_samples, n_features=2, centers=[[10., 10.]], cluster_std=1.0)
#X, y = make_circles(n_samples=15, shuffle=True, noise=None, random_state=None, factor=0.8)
X = np.block([[x_1], [x_2], [x_3]])

y_1 = np.zeros(n_samples)
y_2 = np.zeros(n_samples)
y_3 = np.zeros(n_samples)

X_data = np.array(X)
p = len(X[0,:])
n = len(X[:,0])

for i in range(0,n_samples): 
    y_1[i] = 1 + 0 * x_1[i,0]+ x_1[i,1]
    y_2[i] = 1 + 1* x_2[i,0]+0*x_2[i,1]
    y_3[i] = 1 + x_3[i,0]+ x_3[i,1]

y_data= np.concatenate((y_1, y_2, y_3))
#y = np.block([[y_1], [y_2], [y_3]])

fig = plt.figure()
#ax = plt.axes(projection='3d')
    
#plt.plot(X[:, 0], X[:, 1], y,'co')
plt.plot(X_data[:, 0], X_data[:, 1], 'co')
plt.show()
#y_matrix = np.matrix(y)
#ax = plt.axes(projection='3d')


p = len(X_data[0,:])
n = len(X_data[:,0])
x = np.arange(0,size,1)
Theta = np.zeros(3*n)
W = np.zeros((size, size))
W_predict = np.zeros((60, size))
for i in range(0, size):
    for j in range(0, size):
        if i != j:
            W[i,j] = math.exp((-1)*np.linalg.norm(X_data[i]-X_data[j])**2/var**2);

llr_g = LLR()
regression_g = llr_g.fit(X_data[0:size], y_data[0:size], mu, v, perm_size=100, var = var, Graph=W )
Theta_g = regression_g.Theta
Y_g = regression_g.Y

for i in range(240, 300):
    for j in range(0, size):
        W_predict[i-240,j] = math.exp((-1)*np.linalg.norm(X[i]-X[j])**2/var**2); 
                  
Y_test_g = regression_g.predict(X[240:300], W_predict)

llr_no_g = LLR()
regression_no_g = llr_no_g.fit(X_data[0:size], y_data[0:size], mu, v, perm_size=100,  var = var, Graph=None)
Theta_no_g = regression_no_g.Theta
Y_no_g = regression_no_g.Y
Y_test_no_g = regression_no_g.predict(X[240:300])
row_storchastic = W_predict/W_predict.sum(axis=1)[:,None]

test_entry  = math.exp((-1)*np.linalg.norm(X_data[5]-X_data[4])**2/var**2);
diff_g = regression_g.W_small
diff_no_g = regression_no_g.W_small
W_asynm_no_g = regression_no_g.weight
W_asym_g = regression_g.weight

MSE_g = mean_squared_error(y_data[0:size], Y_g);
MSE_no_g = mean_squared_error(y_data[0:size], Y_no_g);
MSE_test_g = mean_squared_error(y_data[240:300], Y_test_g);
MSE_test_no_g = mean_squared_error(y_data[240:300], Y_test_no_g);
print("MSE fit G =", MSE_g)
print("MSE fit without G =", MSE_no_g)
print("MSE predict G = ", MSE_test_g)
print("MSE predict without G = ", MSE_test_no_g)
#plt.figure()
#plt.scatter(y_data[0:240], Y_g)

#plt.show() 

