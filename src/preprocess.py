import re
import numpy as np
from utils import load_nltk_stopwords

def preprocess(X):
    """
    This function preprocess a collection of documents.

    Parameter:
        X:  Dataset
            tuple = (N, ) where     N = samples
    Returns the corpus of documents preprocessed
            type = np.array
    """
    corpus = []
    nltk_stopwords = load_nltk_stopwords()

    for doc in X:
        # Remove blank lines, lowercase and join into full string
        sentence = " ".join([line.lower() for line in doc.strip().splitlines() if line.strip()])

        # Only keep alphanumeric
        sentence = re.sub("[^0-9a-zA-Z]+", ' ', sentence).strip()

        words_list = [w for w in sentence.split(' ') if w not in nltk_stopwords]

        final_words = " ".join(words_list)
        corpus.append(final_words)

    return np.array(corpus)