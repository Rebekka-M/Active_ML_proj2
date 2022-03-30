import pickle
import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("*.pkl")
results = []
for file in files:
    with open(file, 'rb') as f:
        result = pickle.load(f)
    results.append(result)

probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

results = np.array(results)
experiments = results.shape[0]
results = results.sum(axis=0)
results = results / experiments

fig = plt.figure(figsize=(10, 10))

for i in range(3):
    plt.plot(results[i, :, 0], results[i, :, 1], label=f"$p={probs[i]}$")

plt.ylim([0, 1])
plt.yticks([i / 10 for i in range(11)], fontsize=19)
plt.xticks([j for j in range(0, 381, 20)], fontsize=12)
plt.legend()
plt.show()
