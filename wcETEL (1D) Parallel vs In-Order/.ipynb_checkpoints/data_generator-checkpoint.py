import numpy as np

N = 300
n_datasets = 50

datasets = []

for i in range(n_datasets):
    sig = np.random.beta(2, 2, size=N)
    X = np.sort(sig)
    datasets.append(X)
