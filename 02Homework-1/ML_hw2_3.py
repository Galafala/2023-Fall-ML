import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from ML_hw2_1 import fit_model, predict

ROOT = Path(__file__).parent
M = 3
N = 2

def scatter_plot(x, y, position, title='', xlabel='', ylabel=''):
    plt.subplot(M,N,position)
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def main():
    maturity = [1, 3, 6, 9, 12, 15, 18, 21, 24, 30, 36, 48, 60, 72, 84, 96, 108, 120]
    yields = [7.571, 7.931, 7.977, 7.996,  
              8.126, 8.247, 8.298, 8.304,  
              8.311, 8.327, 8.369, 8.462, 
              8.487, 8.492, 8.479, 8.510, 
              8.507, 8.404]
    
    maturity = np.array(maturity)
    yields = np.array(yields)

    Xtrain = maturity.reshape(-1,1)
    Ytrain = yields.reshape(-1,1)

    plt.figure(figsize=(12,12*1.5))

    scatter_plot(maturity, yields, 1, xlabel="Maturity", ylabel="Yields")

    models = []
    for d in [1, 2, 3, 4, 5, 6]:
        models.append(fit_model(Xtrain, Ytrain, d))

    r2s = []
    for _, r2 in models:
        r2s.append(r2)

    scatter_plot(np.linspace(1,6,6), r2s, 2, xlabel="Dimension", ylabel="R^2")

    residuals = []

    for d, model in enumerate(models, 1):
        y_pd = predict(model=model[0], x=Xtrain, d=d)
        residual = Ytrain-y_pd
        residuals.append(residual)
    
    scatter_plot(Xtrain, residuals[3], 3, xlabel="Maturity", ylabel="4D's Residuals")

    plt.subplot(M, N, 4)
    plt.hist(np.array(residuals).ravel())
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    ax = plt.subplot(M, N, 5)
    residuals = np.array(residuals).ravel()
    sm.qqplot(residuals, line='45', fit=True, ax=ax)

    plt.tight_layout()
    plt.savefig(f"{ROOT}/hw2-3.jpg")


if __name__ == "__main__":
    main()