from utils import fit_model, predict
import matplotlib.pyplot as plt
import numpy as np


def log_likelihood(model, Xtrain, Ytrain, d):
    y_pd = predict(model=model, x=Xtrain, d=d)
    RSS = np.sum(np.power(Ytrain-y_pd, 2))
    N = len(Xtrain)

    AIC = N*np.log(RSS/N) + 2*d
    BIC = N*np.log(RSS/N) + np.log(N)*d

    return AIC, BIC

def main():
    Xtrain = np.array([0.2, 0.3, 0.6, 0.9, 1.1, 1.3, 1.4, 1.6]).reshape(-1,1)
    Ytrain = [0.050446, 0.098426, 0.33277, 0.7266, 1.0972, 1.5697, 1.8487, 2.5015]
    dimensions = np.array([1, 2, 3])

    models = {}
    for d in dimensions:
        models[d] = fit_model(Xtrain, Ytrain, d)

    AICs = []
    BICs = []
    for d, model in models.items():
        AIC, BIC = log_likelihood(model, Xtrain, Ytrain, d)
        AICs.append(AIC)
        BICs.append(BIC)

    plt.figure(figsize=(8, 10))

    plt.subplot(3,2,1)
    plt.scatter(Xtrain, Ytrain, label='Data')
    plt.title('Original Data')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(3,2,5)
    plt.scatter(Xtrain, Ytrain, label='Data')
    plt.xlabel('x')
    plt.ylabel('y')

    for d, model in models.items():
        plt.subplot(3,2,d+1)
        y_pd = predict(model, Xtrain, d)
        plt.title(f'D={d}')
        plt.scatter(Xtrain, y_pd, label=f'D={d}')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(3,2,5)
        plt.plot(Xtrain, y_pd, label=f'D={d}')

    plt.legend()

    plt.subplot(3,2,6)
    plt.bar(dimensions-0.1, AICs, label='AIC', width=0.2)
    plt.bar(dimensions+0.1, BICs, label='BIC', width=0.2)
    plt.xticks(dimensions)
    plt.ylim([-120, 0])
    plt.xlabel('Dimension')
    plt.ylabel('AIC/BIC')

    plt.tight_layout()
    plt.legend()
    plt.savefig('03Homework/hw3-2.jpg')

if __name__ == "__main__":
    main()



