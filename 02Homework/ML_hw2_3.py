import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pathlib import Path

from ML_hw2_1 import fit_model, predict

ROOT = Path(__file__).parent
M = 4
N = 2

def scatter_plot(x, y, position, title='', xlabel='', ylabel=''):
    plt.subplot(M,N,position)
    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def calcualte_residual(train, pd):
    return train-pd

def main():
    maturity = [1, 3, 6, 9, 
                12, 15, 18, 21, 
                24, 30, 36, 48, 
                60, 72, 84, 96, 
                108, 120]
    yields = [7.571, 7.931, 7.977, 7.996,  
              8.126, 8.247, 8.298, 8.304,  
              8.311, 8.327, 8.369, 8.462, 
              8.487, 8.492, 8.479, 8.510, 
              8.507, 8.404]
    
    maturity = np.array(maturity)
    yields = np.array(yields)

    Xtrain = maturity.reshape(-1,1)
    Ytrain = yields.reshape(-1,1)

    plt.figure(figsize=(12,20))

    scatter_plot(maturity, yields, (1,2), title="(a)", xlabel="Maturity", ylabel="Yields")

    models = []
    for d in [1, 2, 3, 4, 5, 6]:
        models.append(fit_model(Xtrain, Ytrain, d))

    r2s = []
    for _, r2 in models:
        r2s.append(r2)

    scatter_plot(np.linspace(1,6,6), r2s, (3,4), title="(b)", xlabel="Order", ylabel="adjust-R-square")

    residuals = []

    for d, model in enumerate(models, 1):
        y_pd = predict(model=model[0], x=Xtrain, d=d)
        residual = calcualte_residual(Ytrain, y_pd)
        residuals.append(residual)
    
    residual = residuals[3] # 4D model

    scatter_plot(Xtrain, residual, (5,6), title="(c)", xlabel="Maturity", ylabel="4D's Residuals")
    plt.axhline(0, c='red', linestyle='--')

    residual = np.array(residual).ravel()

    plt.subplot(M, N, 7)
    plt.hist(residual, bins=6, edgecolor='black')
    plt.title("(d) Histogram")
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")

    ax = plt.subplot(M, N, 8)
    plt.title("(d) qq Plot")
    sm.qqplot(residual, line='45', fit=True, ax=ax)
    # plt.ylim([-1.5,1.5])

    plt.tight_layout()
    plt.savefig(f"{ROOT}/hw2-3.jpg")


if __name__ == "__main__":
    main()