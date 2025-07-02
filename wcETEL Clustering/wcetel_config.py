import numpy as np
from data_generator_GM import generate_GM_data

nrep = 1
N = 100
nIter = 100
d = 1
burn_in = nIter // 2
lamda = 3
Alpha = 0

# X_data = np.random.normal(loc=0, scale=1, size=N)

X_data = generate_GM_data(N=N, seed=42)

# X_data = np.random.standard_t(df=3, size=100)

# X_data = np.random.exponential(scale=1, size=100)

# a = np.random.normal(-3,1, size=50)
# b = np.random.normal(2,0.5, size=50)
# X_data = np.sort(np.concatenate([a,b]))
