import numpy as np



def generate_GM_data(N=100, seed=0):
    np.random.seed(seed)
    sig = np.random.normal(0, 1, size=N-14)
    noi1 = np.random.normal(100, 1, size=7)
    noi2 = np.random.normal(100, 1, size=7)
    X_data = np.sort(np.concatenate([sig, noi1, noi2]))
    return X_data

# Test for less contaminated data
# def generate_GM_data(N=100, seed=0):
#     np.random.seed(seed)
    
#     sig = np.random.normal(0, 1, size=N-6)
#     noi1 = np.random.normal(3, 3, size=3)
#     noi2 = np.random.normal(3, 3, size=3)
    
#     X_data = np.sort(np.concatenate([sig, noi1, noi2]))
#     return X_data
