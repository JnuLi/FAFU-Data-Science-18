import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

df = pd.read_csv('boston.csv')

x_data = df[['RM']].values
y_data = df[['MEDV']].values

# 标准化
from sklearn import preprocessing

stdsc = preprocessing.StandardScaler()
x_data = stdsc.fit_transform(x_data)

import random
# ydata = b + w * xdata 
b = 0 # initial b
w = 0 # initial w
lr = .00001 # learning rate
iteration = 100000
# Store initial values for plotting.
b_history = [b]
w_history = [w]

for i in range(iteration):
    
    b_grad = 0.0
    w_grad = 0.0
    n = random.randint(0,len(y_data)-1)
    b_grad = b_grad  - 2.0*(y_data[n] - b - w*x_data[n])*1.0
    w_grad = w_grad  - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    
    # Update parameters.
    b = b - lr* b_grad 
    w = w - lr* w_grad
    # Store parameters for plotting
    b_history.append(b)
    w_history.append(w)

#error = [np.sum((x_data * w_history[i] + b_history[i]-y_data)**2 / 2) for i in range(len(w_history))]

#plt.plot(range(1,10001), error[:10000])
#plt.show()

x = np.arange(-2.5,27.5,.1) #bias
y = np.arange(-9,21,0.1) #weight
Z =  np.zeros((len(x), len(y)))
X, Y = np.meshgrid(x, y)
Z = np.array([np.sum((Y[i][j]*x_data +X[i][j] - y_data)**2) for i in range(300) for j in range(300)]).reshape(300, 300)

fig = plt.figure()
plt.contourf(x,y,Z, 20)
plt.colorbar()
plt.axis(aspect='image')
plt.plot([22.5], [6.38], 'x', ms=12, markeredgewidth=3, color='orange')
plt.plot(b_history, w_history, 'X-', ms=3, lw=1.5, color='red')

plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()