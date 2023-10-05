from utils import predict, fit_model, generate_data
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
ROOT = Path(__file__).resolve().parent

def main():
    Xtrain, Ytrain = generate_data(num_samples=30)
    dimesions = np.linspace(1, 18, 18).astype(int)

    models = {}
    for d in dimesions:
        models[d] = fit_model(Xtrain, Ytrain, d)

    Xtrain, Ytrain = generate_data(num_samples=1000)
    mse_list = []
    for d, model in models.items():
        y_pred = predict(model=model, x=Xtrain, d=d)
        mse = mean_squared_error(Ytrain, y_pred)
        mse_list.append(mse)

    plt.figure(figsize=(12,15))
    plt.plot(dimesions, mse_list)
    plt.scatter(dimesions, mse_list)
    plt.xlabel('Dimension', fontsize=20)
    plt.ylabel('MSE', fontsize=20)
    plt.xticks(dimesions, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('MSE vs. Dimension', fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{ROOT}/hw3-1-c.jpg")

if __name__ == "__main__":
    main()