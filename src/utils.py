stoplist_file = 'nltk_stopwords.txt'

def load_nltk_stopwords():
    with open(stoplist_file) as f:
        nltk_stopwords = f.readlines()
        return [w.strip() for w in nltk_stopwords]
    

