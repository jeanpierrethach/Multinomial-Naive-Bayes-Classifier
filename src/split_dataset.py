import random
from operator import itemgetter

def split_dataset(data, N, random_seed=1000):
    """
    This function shuffles the indices of the dataset and 
    splits the dataset into training set and validation set.

    Parameters:
        data:   tuple = (N, M) where    N = samples
                                        M = features
        N:  number of samples to use for model training
        random_seed:  value of the random seed for the random number generator
    Returns tuple of train_set, train_labels, valid_set, valid_labels
    """
    # Comment to have random (non-deterministic) results
    random.seed(random_seed)

    inds = list(range(len(data[0])))
    random.shuffle(inds)

    train_inds = inds[:N]
    valid_inds = inds[N:]

    train_set = itemgetter(*train_inds)(data[0])
    valid_set = itemgetter(*valid_inds)(data[0])
    train_labels = itemgetter(*train_inds)(data[1])
    valid_labels = itemgetter(*valid_inds)(data[1])

    return (train_set, train_labels, valid_set, valid_labels)
