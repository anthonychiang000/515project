import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Sum, Product


def generate_data():
    data = np.loadtxt('lda.csv', delimiter=",")
    sample = np.random.choice(data.shape[0], size = 32, replace=False)
    samp_data = data[sample]
    X = samp_data[:, :-1]
    y = samp_data[:, -1]
    return X, y

def fit_gaussian(X, y):
    kernel = Sum(Product(C(1, (1e-20, 1e20)), RBF(1, (1e-15, 1e15))), C(0, (1e-16, 1e16)))

    # Set a non-zero prior mean (e.g., the mean of your data)
    
    model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha = 0.001**2)
    model.fit(X, y)
    print(model.kernel_.get_params())

def fit_gaussian_2(X, y):
    kernel = C(1, (1e-20, 1e20)) * RBF(1, (1e-15, 1e15)) + C(0, (1e-20, 1e16))

    # Set a non-zero prior mean (e.g., the mean of your data)
    
    model2 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha = 0.001**2)
    model2.fit(X, y)
    print(model2.kernel_.get_params())

X, y = generate_data()
fit_gaussian(X, y)
fit_gaussian_2(X, y)