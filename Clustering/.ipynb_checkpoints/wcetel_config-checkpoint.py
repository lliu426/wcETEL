import numpy as np
from data_generator_GM import generate_GM_data

nrep = 1
N = 100
nIter = 100
d = 1
burn_in = nIter // 2
lamda = 3
Alpha = 0

X_data = generate_GM_data(N=N, seed=0)
