import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from utils import predict, fit_model, generate_data


ROOT = Path(__file__).parent
M = 3
N = 2

def draw_model_line(model, x, y, d, index,color='darkorange', alpha=1.0, label=''):
    xfit = np.linspace(0, 1, 30).reshape(-1,1)
    
    yfit = predict(model=model, x=xfit, d=d)
    
    plt.subplot(M,N,index)
    plt.plot(xfit, yfit, c=color, alpha=alpha, lw=2)
    plt.scatter(x, y, c='gray')
    plt.title(f"Dimension {d}")

    plt.xlim([0, 1])


def main():
    Xtrain, Ytrain = generate_data()
    
    plt.figure(figsize=(18,20))

    models = {}
    for d in [2, 5, 10, 14, 18]:
        models[d] = fit_model(Xtrain, Ytrain, d)

    index = 0
    for d, model in models.items():
        index += 1
        draw_model_line(model, Xtrain, Ytrain, d, index, color='darkorange', alpha=1.0, label='line')

    plt.tight_layout()
    plt.savefig(f"{ROOT}/hw3-1-a.jpg")


if __name__ == "__main__":
    main()