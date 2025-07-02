import numpy as np

def generate_datasets(mode="heavy", N=300, n_datasets=50):
    """
    Generate datasets for wcETEL under different contamination levels.

    Parameters:
        mode (str): 'clean', 'mild', or 'heavy'
        N (int): total sample size
        n_datasets (int): number of datasets to generate

    Returns:
        datasets (list): list of generated datasets
    """
    datasets = []
    
    for i in range(n_datasets):
        np.random.seed(1000 + i)  # fixed seed for reproducibility

        if mode == "clean":
            sig = np.random.beta(2, 2, size=N)
            X = np.sort(sig)
        
        elif mode == "mild":
            sig = np.random.beta(2, 2, size=N - 14)
            noi_1 = np.random.beta(1, 100, size=7)
            noi_2 = np.random.beta(100, 1, size=7)
            X = np.sort(np.concatenate([sig, noi_1, noi_2]))

        elif mode == "heavy":
            sig = np.random.beta(2, 2, size=N - 60)
            noi_1 = np.random.beta(1, 100, size=30)
            noi_2 = np.random.beta(100, 1, size=30)
            X = np.sort(np.concatenate([sig, noi_1, noi_2]))

        else:
            raise ValueError("Invalid mode. Use 'clean', 'mild', or 'heavy'.")
        
        datasets.append(X)
    
    return datasets