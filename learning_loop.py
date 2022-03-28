import numpy as np
from joblib import Parallel, delayed
import itertools as it
from modAL.disagreement import vote_entropy_sampling
from modAL.uncertainty import entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.model_selection import train_test_split
from collections import namedtuple
from tqdm.notebook import tqdm, trange
ResultsRecord = namedtuple('ResultsRecord', ['query_id', 'score'])
# in case repetitions are desired
#n_repeats = 5
#permutations=[np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

def labels_to_probs(labels, p_correct, n_classes):
    probs = np.ones((len(labels), n_classes)) * (1-p_correct) / (n_classes - 1)
    probs[np.arange(len(labels)), labels] = p_correct
    
    return probs

def pool_splits(y_good, y_cheap, y_lie, n_classes, good_pool_size, seed):
    if good_pool_size > 0:
        pool_idx, good_idx = train_test_split(np.arange(len(y_good)), test_size=good_pool_size, random_state=seed, stratify=y_good, shuffle=True)
    else:
        pool_idx, good_idx = np.arange(len(y_good)), np.zeros(0, dtype=int)
    
    # Create probability distributions for targets
    y_train = labels_to_probs(y_cheap, 1-y_lie, n_classes)
    y_train[good_idx] = labels_to_probs(y_good[good_idx], 1, n_classes)

    return y_train, pool_idx, good_idx


def pool_update(y_train, labels, labels_idx, pool_idx, good_idx):
    # WARN: Function mutates y_train
    
    # Update y_train probabilities to oracle labels
    # y_train[labels_idx] = labels_to_probs(labels, 1, n_classes=y_train.shape[1])
    labels_prob = np.zeros((len(labels), y_train.shape[1]))
    labels_prob[np.arange(len(labels)), labels] = 1
    y_train[labels_idx] = labels_prob

    # Update pool and good indexes to remove new oracle samples
    pool_idx = np.setdiff1d(pool_idx, labels_idx)
    good_idx = np.append(good_idx, labels_idx)

    return y_train, pool_idx, good_idx


def learning_loop(Estimator, X, y_good, y_cheap, y_lie, n_classes, good_pool_size, n_queries, X_test, y_test, seed):
    # Set random seeds and initialize estimator
    rng = np.random.default_rng(seed)
    estimator = Estimator(seed=seed)
    
    # Ensure labels are integers
    y_good = y_good.astype(int)
    y_cheap = y_cheap.astype(int)

    # Prepare pool
    y_train, pool_idx, good_idx = pool_splits(y_good, y_cheap, y_lie, n_classes, good_pool_size, seed)
    
    #Store results
    results = []

    #Initialize learner
    #start_indices = permutations[i_repeat][:1]

    # TODO: Do we use vote entropy sampling?
    learner = ActiveLearner(estimator=estimator,
                            query_strategy=entropy_sampling,
                            X_training=X,
                            y_training=y_train)

    for i_query in range(n_queries):
        #get learner uncertainty query
        y_new_idx, query_inst = learner.query(X[pool_idx])
        y_new_idx = pool_idx[y_new_idx]

        #obtaining new labels from the Oracle
        y_new = y_good[y_new_idx]

        # supply label for queried instance
        # learner.teach(X[y_new_idx], labels_to_probs(y_new, 1, n_classes))
        learner.y_training[y_new_idx] = labels_to_probs(y_new, 1, n_classes)
        learner._fit_to_known(bootstrap=False)

        # learner._set_classes() #this is needed to update for unknown class labels

        # Update pool
        y_train, pool_idx, good_idx = pool_update(y_train, y_new, y_new_idx, pool_idx, good_idx)

        
        # Test model
        score = learner.score(X_test, y_test)
        print(f"{y_lie}: {i_query}/{n_queries} accuracy: {score}", flush=True)

        # Track scores
        results.append(ResultsRecord(
            i_query+1,
            score
        ))


    return results


def learning_loop_multiple(Estimator, X, y_good, y_cheaps, y_lies, n_classes, good_pool_size, n_queries, X_test, y_test, seed):
    #TODO: Increase amount of parallel jobs. Set to -1 to use all available resources.
    return Parallel(n_jobs=5, batch_size="auto", verbose=5)(
        delayed(learning_loop)(
            Estimator, 
            X, y_good, y_cheaps[i], 
            y_lies[i], n_classes,
            good_pool_size, n_queries,
            X_test, y_test, 
            seed
        )
        for i in range(len(y_lies))
    )