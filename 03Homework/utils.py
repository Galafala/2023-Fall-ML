from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def fit_model(x, y, d, show=False):
    poly = PolynomialFeatures(degree=d)
    xtrain_poly = poly.fit_transform(x)

    polynomial_model = LinearRegression(fit_intercept=True)
    polynomial_model.fit(xtrain_poly, y)

    quad_r2 = polynomial_model.score(xtrain_poly, y)

    if show:
        print(f'Dinmension: {d}')
        print(f'coef: {polynomial_model.coef_}')
        print(f'inter: {polynomial_model.intercept_}')
        print(f'r^2: {quad_r2}')


    return polynomial_model

def predict(model, x, d):
    poly = PolynomialFeatures(degree=d)
    x_poly = poly.fit_transform(x)
    
    return model.predict(x_poly)

def generate_data(variance=0.07, mean=0, num_samples=30):
    import numpy as np
    sigma = np.sqrt(variance)
    Normal = np.random.normal(mean, sigma, num_samples)
    g = lambda x: np.power(np.sin(2*np.pi*x), 2)
    Xtrain = np.linspace(0, 1, num_samples).reshape(-1,1)
    Ytrain = [g(x)+Normal[i] for i, x in enumerate(Xtrain)]
    
    return Xtrain, Ytrain