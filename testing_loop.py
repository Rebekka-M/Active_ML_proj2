from learning_loop import learning_loop_multiple
from sklearn.linear_model import LogisticRegression
from model_wrapper import ModelWrapper
from model import CNN_class
from sklearn.datasets import make_classification
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from torchvision import datasets
import torch
from sklearn.model_selection import train_test_split
import os

class LogRegWrapper(LogisticRegression):
    def __init__(self, penalty="l2", *, dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, seed=None):
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio)
    
    def fit(self, X, y, sample_weight=None):
        return super().fit(X, y.argmax(axis=1))

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)


seed_mine = 69
seed_everything(seed_mine)
seed = 69

n_classes = 10
n_queries = 380
n_samples = 2000
#split_idx = 400
# Using sklearn generated random dataset
# X, y_good = make_classification(n_samples=n_samples,
#                            n_features=25,
#                            n_informative = 15,
#                            n_redundant = 10,
#                            n_classes=n_classes,
#                            n_clusters_per_class = 1,
#                            weights=None,
#                            flip_y=0,
#                            random_state=seed)


training_set = datasets.MNIST(root="./data", train=True,  download=True)#, transform=transform)
ds = torch.utils.data.Subset(training_set, range(n_samples))
X = np.array([np.array(i[0], dtype=np.float32)/255 for i in ds])
y_good = np.array([np.array(i[1]) for i in ds])


train_idx, test_idx = train_test_split(np.arange(len(y_good)), test_size=0.5, random_state=seed, stratify=y_good, shuffle=True)
X_train = X[train_idx]
X_test = X[test_idx]
y_train_good = y_good[train_idx]
y_test_good = y_good[test_idx]

y_lies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
# y_cheap = y_train_good.copy()
# mask = np.random.uniform(0,1,len(y_cheap)) < y_lie
# y_cheap[mask] = np.random.uniform(0,n_classes, np.sum(mask))

y_cheap = np.zeros((len(y_lies), len(y_train_good)))

for i, lie in enumerate(y_lies):
    y_cheap[i, :] = y_train_good.copy()
    mask = np.random.uniform(0,1,len(y_train_good)) < lie
    y_cheap[i, mask] = np.random.uniform(0, n_classes, np.sum(mask))

results = learning_loop_multiple(
    Estimator=ModelWrapper,
    X=X_train,
    y_good=y_train_good,
    y_cheaps=y_cheap,
    y_lies=y_lies,
    n_classes=n_classes,
    good_pool_size=0.02,
    n_queries=n_queries,
    X_test=X_test,
    y_test=y_test_good,
    seed=seed
)

with open(f'testing_loop_baseline.pkl', 'wb') as handle:
    pickle.dump(results, handle)

#%%
with open(f'testing_loop_baseline.pkl', 'rb') as handle:
    results = pickle.load(handle)

scores = np.zeros((len(results), n_queries))
for i in range(len(results)):
    _, scores[i,:] = zip(*results[i])

for i in range(len(scores)):
    plt.plot(scores[i,:])
plt.show()

# %%
