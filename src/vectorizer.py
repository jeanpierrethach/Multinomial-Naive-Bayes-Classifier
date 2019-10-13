import numpy as np
import scipy.sparse as sp
from preprocess import preprocess

from itertools import chain
from collections import Counter

class TfidfVectorizer():
    def __init__(self):
        pass

    def _get_vocabulary(self, corpus):
        words = []
        for doc in corpus:
            for word in doc.split(" "):
                words.append(word)

        vocabulary = list(set(words))

        # Create a dictionary with words as keys and indices as values
        word_dict = {value: i for i, value in enumerate(vocabulary)}

        return word_dict, len(vocabulary)

    def count_word_appearance(self, corpus):
        """
        This method returns a dictionary containing all unique words and their
        occurrences from the corpus
        """
        return dict(Counter(chain.from_iterable(set(doc.split(" ")) for doc in corpus)))

    def _tf(self, corpus):
        """
        This method returns a np.array of term frequency for every word in the
        vocabulary adjusted with the document length
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            bag_vector[self.word_dict[word]] += 1

        bag_vector /= len(doc)
        return bag_vector

    def _tf_raw_count(self, corpus):
        """
        This method returns a np.array of term frequency for every word in the
        vocabulary
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            bag_vector[self.word_dict[word]] += 1

        return bag_vector

    def _tf_log_normalization(self, corpus):
        """
        This method returns a np.array of the log normalization of term frequency 
        for every word in the vocabulary
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            bag_vector[self.word_dict[word]] += 1

        bag_vector = np.log(bag_vector + 1)
        return bag_vector
    
    def _idf_smooth(self, corpus):
        """
        This method returns a np.array of the inverse document frequency 
        with smoothing
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        for word, count in self.count_word_appearance(corpus).items():
            bag_vector[self.word_dict[word]] += count

        bag_vector = np.log(corpus.shape[0] / (bag_vector + 1))
        return bag_vector
    
    def fit_transform(self, corpus):
        """
        This method transforms the corpus into a matrix representation
        using the term frequency and inverse document frequency as scores
        and creates the vocabulary of the corpus.

        Parameter:
            corpus: shape = (N, ) where   N = number of samples
                    type = np.array
        Returns c_matrix (scipy.sparse.csr.csr_matrix)
                shape (N, M) where  N = number of samples
                                    M = number of features
        """
        self.word_dict, self.n_vocab = self._get_vocabulary(corpus)

        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))

        idf_vec = self._idf_smooth(corpus)
        for doc in corpus:
            tf_vec = self._tf(doc)    
            tfidf_vec = tf_vec * idf_vec
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(tfidf_vec)))
        return c_matrix


    def _tf_oov(self, corpus):
        """
        This method returns a np.array of term frequency for every word in the
        vocabulary adjusted with the document length. Out of vocabulary words 
        aren't taken in account.
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            if word not in self.word_dict:
                continue
            bag_vector[self.word_dict[word]] += 1

        bag_vector /= len(doc)
        return bag_vector

    def _tf_raw_count_oov(self, corpus):
        """
        This method returns a np.array of term frequency for every word in the
        vocabulary. Out of vocabulary words aren't taken in account.
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            if word not in self.word_dict:
                continue
            bag_vector[self.word_dict[word]] += 1

        return bag_vector

    def _tf_log_normalization_oov(self, corpus):
        """
        This method returns a np.array of the log normalization of term frequency 
        for every word in the vocabulary. Out of vocabulary words aren't taken in account.
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        doc = corpus.split(" ")
        for word in doc:
            if word not in self.word_dict:
                continue
            bag_vector[self.word_dict[word]] += 1

        bag_vector = np.log(bag_vector + 1)
        return bag_vector

    def _idf_smooth_oov(self, corpus):
        """
        This method returns a np.array of the inverse document frequency 
        with smoothing. Out of vocabulary words aren't taken in account.
        """
        bag_vector = np.zeros(self.n_vocab, dtype=np.float64)

        for word, count in self.count_word_appearance(corpus).items():
            if word not in self.word_dict:
                continue
            bag_vector[self.word_dict[word]] += count

        bag_vector = np.log(corpus.shape[0] / (bag_vector + 1))
        return bag_vector
 
    def transform(self, corpus):
        """
        This method transforms the corpus into a matrix representation
        using the term frequency and inverse document frequency as scores. 
        Out of vocabulary words aren't taken in account.

        Parameter:
            corpus: shape = (N, ) where   N = number of samples
                    type = np.array
        Returns c_matrix (scipy.sparse.csr.csr_matrix) of shape corpus

        Note: The corpus will be preprocessed before creating the matrix.
        """
        corpus = preprocess(corpus)

        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))

        idf_vec = self._idf_smooth_oov(corpus)
        for doc in corpus:
            tf_vec = self._tf_oov(doc)    
            tfidf_vec = tf_vec * idf_vec
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(tfidf_vec)))
        return c_matrix


class CountVectorizer():
    def __init__(self):
        pass

    def _get_vocabulary(self, corpus):
        words = []
        for doc in corpus:
            for word in doc.split(" "):
                words.append(word)
        
        vocabulary = list(set(words))

        # Create a dictionary with words as keys and indices as values
        word_dict = {word: idx for idx, word in enumerate(vocabulary)}

        return word_dict, len(vocabulary)

    def fit_transform(self, corpus):
        """
        This method transforms the corpus into a matrix representation
        using word count as scores and creates the vocabulary of the corpus.

        Parameter:
            corpus: shape = (N, ) where   N = number of samples
                    type = np.array
        Returns c_matrix (scipy.sparse.csr.csr_matrix)
                shape (N, M) where  N = number of samples
                                    M = number of features
        """
        self.word_dict, self.n_vocab = self._get_vocabulary(corpus)
       
        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))

        for doc in corpus:
            bag_vector = np.zeros(self.n_vocab, dtype=np.int64)
            for word in doc.split(" "):
                bag_vector[self.word_dict[word]] += 1
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(bag_vector)))
        return c_matrix
      
    def transform(self, corpus):
        """
        This method transforms the corpus into a matrix representation
        using word count as scores. Out of vocabulary words aren't taken
        in account.
        
        Parameter:
            corpus: shape = (N, ) where   N = number of samples
                    type = np.array
        Returns c_matrix (scipy.sparse.csr.csr_matrix) of shape corpus

        Note: The corpus will be preprocessed before creating the matrix.
        """
        corpus = preprocess(corpus)

        c_matrix = sp.csr_matrix(([], ([], [])), shape=(0, self.n_vocab))
        
        for doc in corpus:
            bag_vector = np.zeros(self.n_vocab, dtype=np.int64)
            for word in doc.split(" "):
                if word not in self.word_dict:
                    continue
                bag_vector[self.word_dict[word]] += 1
            c_matrix = sp.vstack((c_matrix, sp.csr_matrix(bag_vector)))
        return c_matrix
