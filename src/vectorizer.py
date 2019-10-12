import numpy as np
import scipy.sparse as sp
from preprocess import preprocess

class CountVectorizer():
    def __init__(self):
        pass

    def _get_vocabulary(self, doc):
        words = []
        for sentence in doc:
            for word in sentence.split(" "):
                words.append(word)
        
        vocabulary = list(set(words))

        # Create a dictionary with words as keys and indices as values
        word_dict = {word: idx for idx, word in enumerate(vocabulary)}

        return word_dict, len(vocabulary)

    def fit_transform(self, doc):
        """
        This method transforms the document into a matrix representation
        using word count as scores and creates the vocabulary of the document.

        Parameter:
            doc:  list = (N, ) where   N = number of samples
        Returns c_matrix (scipy.sparse.csr.csr_matrix)
            shape (N, M) where      N = number of samples
                                    M = number of features
        """
        self.word_dict, self.n_vocab = self._get_vocabulary(doc)
       
        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))

        for sentence in doc:
            bag_vector = np.zeros(self.n_vocab, dtype=np.int64)
            for word in sentence.split(" "):
                bag_vector[self.word_dict[word]] += 1
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(bag_vector)))
        return c_matrix
      
    def transform(self, doc):
        """
        This method transforms the document into a matrix representation
        using word count as scores. Out of vocabulary words aren't taken
        in account.
        
        Parameter:
            doc:    shape = (N, ) where   N = number of samples
                    type = np.array
        Returns c_matrix (scipy.sparse.csr.csr_matrix) of shape doc
        """
        doc = preprocess(doc)

        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))
        
        for sentence in doc:
            bag_vector = np.zeros(self.n_vocab, dtype=np.int64)
            for word in sentence.split(" "):
                if word not in self.word_dict:
                    continue
                bag_vector[self.word_dict[word]] += 1
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(bag_vector)))
        return c_matrix
