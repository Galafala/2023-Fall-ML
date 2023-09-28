from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent
M = 3
N = 2

def fit_model(x, y, d, show=True):
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

    return (polynomial_model, quad_r2)

def predict(model, x, d):
    poly = PolynomialFeatures(degree=d)
    x_poly = poly.fit_transform(x)
    
    return model.predict(x_poly)


def draw_model_line(models, x, y, d, color='darkorange', alpha=1.0, label=''):
    model = models[0]
    q2 = models[1]

    xfit = np.linspace(0, 1, 20).reshape(-1,1)
    
    yfit = predict(model=model, x=xfit, d=d)
    
    plt.subplot(M,N,1)
    plt.plot(xfit, yfit, c=color, alpha=alpha, lw=2, label=f'{label}, $R^2={q2:.2f}')
    plt.scatter(x, y, c='gray')

    plt.xlim([0, 1])
    plt.ylim([-2, 12])

    y_pd = predict(model=model, x=x, d=d)
    residuals = y-y_pd
    
    # Residual Plot
    plt.subplot(M,N,2)
    plt.scatter(x, residuals, c='gray')
    plt.axhline(0, c='red', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([-2, 2])
    plt.xlabel('x')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot (Degree {d})')

    #linearity
    plt.subplot(M,N,3)
    plt.scatter(y_pd, residuals, c='gray')
    plt.axhline(max(residuals)+0.1, c='red', linestyle='--')
    plt.axhline(min(residuals)-0.1, c='red', linestyle='--')
    plt.ylim([-2, 2])
    plt.xlabel('y predict')
    plt.ylabel('Residuals')
    plt.title(f'Linearity (Degree {d})')

    #linearity
    plt.subplot(M,N,4)
    plt.scatter(y_pd, residuals, c='gray')
    plt.axhline(0, c='red', linestyle='--')
    plt.ylim([-2, 2])
    plt.xlabel('y predict')
    plt.ylabel('Residuals')
    plt.title(f'Linearity (Degree {d})')

    # normal distribution
    plt.subplot(M,N,5)
    plt.hist(residuals, bins='auto', edgecolor='black')
    plt.ylim([0, 10])
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'Normal Distribution (Degree {d})')

    # Independence
    plt.subplot(M,N,6)
    plt.acorr(residuals.ravel())
    plt.xlim([-12, 12])
    plt.ylim([-0.75, 1.25])
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Independence (Degree {d})')


def main():   
    x_path = f"{ROOT}/02HW1_Xtrain"
    y_path = f"{ROOT}/02HW1_Ytrain"

    Xtrain = np.loadtxt(x_path)
    Ytrain = np.loadtxt(y_path)

    Xtrain = Xtrain.reshape(-1, 1)
    Ytrain = Ytrain.reshape(-1, 1)

    plt.figure(figsize=(18,20))

    models =[]

    for d in [3]:
        models.append(fit_model(Xtrain, Ytrain, d))

    for d, model in enumerate(models, 3):
        draw_model_line(model, Xtrain, Ytrain, d, color='darkorange', alpha=1.0, label='line')

    plt.tight_layout()
    plt.savefig(f"{ROOT}/hw2-1.jpg")


if __name__ == "__main__":
    main()