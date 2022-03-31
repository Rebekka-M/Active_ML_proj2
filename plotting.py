import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob
from scipy.ndimage import gaussian_filter1d

files = glob.glob("*.pkl")
results = []
for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
    results.append(result)

probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

results = np.array(results)
# results = np.median(results, axis=0)
results = np.mean(results, axis=0)

fig = plt.figure(figsize=(10, 10))

for i in range(10):
    results[i, :, 1] = gaussian_filter1d(results[i, :, 1], 16)
    plt.plot(results[i, 20:, 0], results[i, 20:, 1], label=f"$p={probs[i]}$", linewidth=3)

plt.ylim([0.8, 1])
plt.xlim([15, 385])
# plt.yticks([i / 10 for i in range(11)], fontsize=19)
plt.xticks([j for j in range(20, 381, 20)], fontsize=12)
plt.legend()
plt.show()
