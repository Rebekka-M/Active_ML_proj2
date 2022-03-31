import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import gaussian_filter1d

files = glob.glob("pkl3/*.pkl")
results_nonzero = []
results_zero = []

found_zero = False

for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
    if len(result) == 11:
        results_nonzero.append(result)
        print('lidt sejt')
    elif len(result) == 1:
        print('pænt sejt')
        found_zero = True
        results_zero.append(result)
    else:
        raise ValueError('Something\'s wrong with the shapes')

probs = [-1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

eps = [f"$\epsilon$={round(i - i / 10, 2)}" for i in probs]
eps[0] = "No cheap data"

results_nonzero = np.mean(results_nonzero, axis = 0)

if found_zero:
    results_zero = np.mean(results_zero, axis = 0)
    results = np.array([results_nonzero[0], results_zero[0], *results_nonzero])
else:
    results = np.array(results_nonzero)
    probs.pop(1)


fig = plt.figure(figsize=(10, 10))

for i in range(12):
    plt.plot(results[i, :, 0] + 20, results[i, :, 1], label=eps[i], linewidth=3)

plt.ylim([0.35, 1])
plt.xlim([15, 405])
plt.yticks([i / 10 for i in range(3, 11)], fontsize=15)
plt.xticks([j for j in range(20, 401, 20)], fontsize=13)
plt.legend(fontsize=14)
plt.title(r"Accuracy vs. Number of expensive data points $n_{ex}$", fontsize=20)
plt.xlabel(r"$n_{ex}$", fontsize=16)
plt.ylabel(r"accuracy", fontsize=16)
plt.grid()
plt.savefig('non_filtered_results.png', bbox_inches='tight')

fig = plt.figure(figsize=(10, 10))

for i in [0, 1, 2, 6, 7, 8, 9, 10, 11]:
    results[i, :, 1] = gaussian_filter1d(results[i, :, 1], 8)
    plt.plot(results[i, :, 0] + 20, results[i, :, 1], label=eps[i], linewidth=3)

plt.ylim([0.35, 1])
plt.xlim([15, 405])
plt.yticks([i / 10 for i in range(3, 11)], fontsize=15)
plt.xticks([j for j in range(20, 401, 20)], fontsize=13)
plt.legend(fontsize=14)
plt.title(r"Accuracy vs. Number of expensive data points $n_{ex}$", fontsize=20)
plt.xlabel(r"$n_{ex}$", fontsize=16)
plt.ylabel(r"accuracy", fontsize=16)
plt.grid()
plt.savefig('allaccs.png', bbox_inches='tight')

fig = plt.figure(figsize=(10, 10))

for i in [1, 2, 3, 4, 5, 6]:
    results[i, :, 1] = gaussian_filter1d(results[i, :, 1], 8)
    plt.plot(results[i, :, 0] + 20, results[i, :, 1], label=eps[i], linewidth=3)

plt.xlim([15, 405])
plt.yticks([0.86, 0.87, 0.88, 0.89, 0.9, 0.91], fontsize=15)
plt.xticks([j for j in range(20, 401, 20)], fontsize=13)
plt.legend(fontsize=14)
plt.title(r"Accuracy vs. Number of expensive data points $n_{ex}$", fontsize=20)
plt.xlabel(r"$n_{ex}$", fontsize=16)
plt.ylabel(r"accuracy", fontsize=16)
plt.grid()
plt.savefig('top5accs.png', bbox_inches='tight')
