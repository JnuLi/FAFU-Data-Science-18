import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('boston.csv')

class LinearRegressionGD(object):

    def __init__(self, eta=0.00001, n_iter=10000):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + 1)
        self.cost_ = []
        self.w_history = []
        self.b_history = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_history.append(self.w_[1])
            self.b_history.append(self.w_[0])
            self.w_[1] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)

x_data = df['RM'].values
y_data = df['MEDV'].values

from sklearn import preprocessing

stdsc = preprocessing.StandardScaler()
x_data = stdsc.fit_transform(x_data.reshape(-1, 1)).reshape(506)

lr = LinearRegressionGD()
lr.fit(x_data, y_data)

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
plt.plot(lr.b_history, lr.w_history, 'X-', ms=3, lw=1.5, color='red')

plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()