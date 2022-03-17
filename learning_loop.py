import numpy as np
from joblib import Parallel, delayed
import itertools as it
from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.model_selection import train_test_split
from collections import namedtuple
from tqdm.notebook import tqdm, trange
ResultsRecord = namedtuple('ResultsRecord', ['query_id', 'score'])
# in case repetitions are desired
#n_repeats = 5
#permutations=[np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

def pool_splits(X, y_good, y_cheap, cheap_pool_size, seed):
    pool_idx, good_idx = train_test_split(X, test_size=cheap_pool_size, random_state=seed, stratisfy=y_good, shuffle=True)
    
    X_train = X
    y_train = y_cheap.copy()
    y_train[good_idx] = y_good[good_idx]

    X_pool = X_train[pool_idx]
    y_pool = y_good[pool_idx]

    return X_train, X_pool, y_train, y_pool, pool_idx, good_idx

def pool_update(y_train, labels, labels_idx, pool_idx, good_idx):
    # Update y_train probabilities to oracle labels
    labels_prob = np.zeros((len(labels), y_train.shape[1]))
    labels_prob[np.arange(len(labels)), labels] = 1
    y_train[labels_idx] = labels_prob

    # Update pool and good indexes to remove new oracle samples
    pool_idx = np.setdiff1d(pool_idx, labels_idx)
    good_idx = np.append((good_idx, labels_idx))

    return y_train, pool_idx, good_idx



def learning_loop(Estimator, X, y_good, X_test, y_test, y_cheap, cheap_pool_size, seed, n_queries):
    # Set random seeds and initialize estimator
    rng = np.random.default_rng(seed)
    estimator = Estimator(seed=seed)

    # Prepare pool
    X_train, X_pool, y_train, y_pool, pool_idx, good_idx = pool_splits(X, y_good, y_cheap, cheap_pool_size, seed)
    
    #Store results
    results = []
    print('')

    #Initialize learner
    #start_indices = permutations[i_repeat][:1]

    # TODO: Do we use vote entropy sampling?
    learner = ActiveLearner(estimator=estimator,
                            query_strategy=vote_entropy_sampling,
                            X_training=X_train,
                            y_training=y_train)

    for i_query in tqdm(range(1, n_queries), desc=f'XXX', leave=False):
        #get learner uncertainty query
        query_idx, query_inst = learner.query(X_pool)

        #obtaining new labels from the Oracle
        y_new = y_good[query_idx]

        # supply label for queried instance
        learner.teach(X_pool[query_idx], y_new)

        learner._set_classes() #this is needed to update for unknown class labels

        # Update pool
        y_train, pool_idx, good_idx = pool_update(y_train, y_new, query_idx, pool_idx, good_idx)

        score = learner.score(X_test, y_test)

        results.append(ResultsRecord(
            i_query,
            score))

    return results

# We need to call this in main for parallel programming
result = Parallel(n_jobs=-1)(delayed(train_committee)(i,i_members,X_train,y_train)
                    for i, i_members in it.product(range(n_repeats), n_members))
committee_results=[r for rs in result for r in rs]