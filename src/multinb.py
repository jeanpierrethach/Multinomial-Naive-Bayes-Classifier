import numpy as np
import scipy.sparse as sp

class MultinomialNaiveBayes():
    def __init__(self, alpha=1.0, verbose=False):
        self.alpha = alpha
        self.verbose = verbose
        if self.verbose:
            np.set_printoptions(threshold=np.inf)    

    def fit(self, X, y):
        """
        This method fit the classifier according to X and y.

        Parameters:
            X:  Training set vectorized
                shape = (N, M) where    N = number of samples
                                        M = number of features
                type = scipy.sparse.csr.csr_matrix
            y:  Target label values
                shape = (N, )
        Returns self
        """
        self.label_classes, counts = np.unique(y, return_counts=True)
        total_docs, total_words = X.shape

        # Calculates the priors of each class
        self.priors = dict(zip(self.label_classes, 1.0 * counts / total_docs))
        self.likelihood = sp.csr_matrix(([], ([], [])), shape=(0, total_words))

        # Calculates the likelihood probability of words given each class with Laplace smoothing
        for label_c in self.label_classes:
            label_c_idx = np.where(y == label_c)
            self.likelihood = sp.vstack((self.likelihood, sp.csr_matrix((X[label_c_idx].sum(axis=0) + self.alpha) / (X[label_c_idx].sum() + (self.alpha * total_words)))))
                  
        return self

    def predict(self, X):
        """
        This method predicts the class for each sample in X using the
        maximum log-likelihood approach.

        Parameters:
            X:  shape = (N, M) where    N = number of samples
                                        M = number of features 
                type = scipy.sparse.csr.csr_matrix
        Returns the predictions of class labels
                shape = (N, )
                type = np.array
        """
        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, X.shape[0]))

        for c_idx, c_likelihood in enumerate(self.likelihood):
            
            p_sample = np.zeros(X.shape[0], dtype=np.float64)
            for idx, sample in enumerate(X):
                nz_sample_idx = np.nonzero(sample)
                
                # Calculates the probability of the word given the class with the word frequency in the sample
                p_word_given_class = np.power(c_likelihood.A[nz_sample_idx], sample[nz_sample_idx])

                # Removes any zero probability
                p_word_given_class = p_word_given_class[np.nonzero(p_word_given_class)]

                # Calculates the log likelihood for each sample given the class
                p_sample[idx] = np.log(self.priors[self.label_classes[c_idx]]) + np.sum(np.log(p_word_given_class))

            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(p_sample)))

        if self.verbose:
            print(c_matrix.argmax(axis=0).A)

        preds = c_matrix.argmax(axis=0).A.flatten() 
        classes_preds = self._pred_classes(preds)
        
        return classes_preds

    def _pred_classes(self, predictions):
        """
        This method creates the predictions with the class label names 
        based on the index values.

        Parameter:
            predictions:    shape = (N, )   where N = number of samples
                            type = np.array
        Returns np.array of class labels
        """
        return np.array([self.label_classes[pred] for pred in predictions])

