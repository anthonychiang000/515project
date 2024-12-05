import numpy as np
import math
import GPy
import matplotlib.pyplot as plt
#GPy

def generate_data():
    data = np.loadtxt('svm.csv', delimiter=",")
    sample = np.random.choice(data.shape[0], size = 32, replace=False)
    samp_data = data[sample]
    X = samp_data[:, :-1]
    y = samp_data[:, -1].reshape(-1, 1)
    return X, y

def fit_gaussian_gpy(X, y):
    k = GPy.kern.RBF(input_dim=X.shape[1], variance=1.0, lengthscale=1.0)  # No need for separate constant kernel in the product
    mean = GPy.mappings.Constant(input_dim = X.shape[1], output_dim = 1, value = np.mean(y))
    model = GPy.models.GPRegression(X, y, k, noise_var=0.001**2, mean_function=mean)

    model.Gaussian_noise.variance.fix()
    #model.kern.lengthscale.constrain_bounded(1e-15, 1e15)
    #model.kern.variance.constrain_bounded(1e-20, 1e20)  # Constrain variance of RBF

    model.optimize_restarts(20, verbose=False)  

    print(model)  
    print(model.log_likelihood())
    #print("RBF variance:", model.kern.variance)
    #print("RBF lengthscale:", model.kern.lengthscale)
    #print("Constant Mean:", model.mean_function)
    print("BIC:", 3 * math.log(32) - 2 * model.log_likelihood())

X, y = generate_data()

fit_gaussian_gpy(X, y)

