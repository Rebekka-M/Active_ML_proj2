import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import gaussian_filter1d
from typing import Optional, Iterable

sigma = 2
probs = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

eps = [f"$\epsilon$={round(i - i / 10, 2)}" for i in probs]
eps[0] = "No cheap data"

# Get all files in pkl? ending with .pkl

def get_results(filename):
    files = glob.glob(f"{filename}/*.pkl")
    results = []
    for file in files:
        with open(file, 'rb') as f:
            result = pickle.load(f)
        results.append(result)
    return np.mean(results,axis = 0)

results_zero = get_results('pkl3-baseline')
results_nonzero = get_results('pkl3')

results = np.array([results_nonzero[0], results_zero[0], *results_nonzero[1:]])

# # Turn results list into ndarray and take the mean over the different experiments
# results = np.array(results)
# # results = np.median(results, axis=0)
# results = np.mean(results, axis=0)


def plot(plot_idx: Iterable, ylim: list, saveas: str, yticks: Optional = None, filter: bool = True):
    fig = plt.figure(figsize=(10, 10))
    for i in plot_idx:
        if filter:
            results[i, :, 1] = gaussian_filter1d(results[i, :, 1], sigma=sigma)
        plt.plot(results[i, :, 0] + 20, results[i, :, 1], label=eps[i], linewidth=3)

    if ylim is not None:
        plt.ylim(ylim)
    plt.xlim([15, 405])
    if yticks is not None:
        plt.yticks(yticks, fontsize=15)
    else:
        plt.yticks([i / 10 for i in range(3, 11)], fontsize=15)
    plt.xticks([j for j in range(20, 401, 20)], fontsize=13)
    plt.legend(fontsize=14)
    plt.title(r"Accuracy vs. Number of expensive data points $n_{ex}$", fontsize=20)
    plt.xlabel(r"$n_{ex}$", fontsize=16)
    plt.ylabel(r"accuracy", fontsize=16)
    plt.grid()
    plt.savefig(f'{saveas}.png', bbox_inches='tight')


plot(plot_idx=range(11), ylim=[0.35, 1], saveas='non_filtered_results', filter=False)  # Raw data
plot(plot_idx=[0, 1, 2, 6, 7, 8, 9, 10], ylim=[0.80, 1], saveas='allaccs', filter=True)  # All accs
plot(plot_idx=[1, 2, 3, 4, 5, 6], ylim=None, saveas='top5accs', yticks=[0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91],
     filter=True)
