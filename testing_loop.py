from learning_loop import learning_loop_multiple
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import numpy as np
import random

class LogRegWrapper(LogisticRegression):
    def __init__(self, penalty="l2", *, dual=False, tol=0.0001, C=1, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver="lbfgs", max_iter=100, multi_class="auto", verbose=0, warm_start=False, n_jobs=None, l1_ratio=None, seed=None):
        super().__init__(penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs, l1_ratio)
    
    def fit(self, X, y, sample_weight=None):
        return self.fit(X, y.argmax(axis=1))

seed = 1
n_classes = 10
n_samples = 100
# Using sklearn generated random dataset
X, y_good = make_classification(n_samples=n_samples,
                           n_features=25,
                           n_informative = 15,
                           n_redundant = 10,
                           n_classes=n_classes,
                           n_clusters_per_class = 1,
                           weights=None,
                           flip_y=0,
                           random_state=seed)
#Data lie
split_idx = 40
X_train = X[:split_idx]
X_test = X[split_idx:]
y_train_good = y_good[:split_idx]
y_test_good = y_good[split_idx:]

y_lie = 0.2
y_cheap = y_good.copy()
mask = np.random.uniform(0,1,len(X)) < y_lie
y_cheap[mask] = np.random.uniform(0,n_classes, np.sum(mask))

results = learning_loop_multiple(
    Estimator=LogRegWrapper,
    X=X,
    y_good=y_train_good,
    y_cheap=y_cheap,
    y_lie=y_lie,
    n_classes=n_classes,
    cheap_pool_sizes=[0.1, 0.2, 0.3],
    n_queries=40,
    X_test=X_test,
    y_test=y_test_good,
    seed=seed
)


for result in results:
    print(result)
#Estimator, X, y_good, y_cheap, y_lie, n_classes, cheap_pool_sizes, n_queries, X_test, y_test, seed