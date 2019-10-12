import numpy as np
from split_dataset import split_dataset
from preprocess import preprocess

from vectorizer import CountVectorizer
from multinb import MultinomialNaiveBayes

data = np.load("../data_train.pkl", allow_pickle=True)
test_data = np.load("../data_test.pkl", allow_pickle=True)

train_set, train_labels, valid_set, valid_labels = split_dataset(data, N=49000, random_seed=3395)

corpus = preprocess(train_set)

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(corpus)
y = np.array(train_labels)

naive = MultinomialNaiveBayes(alpha=0.35, verbose=True)
classifier = naive.fit(X, y)

predict_vec = classifier.predict(vectorizer.transform(np.array(valid_set)))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(list(valid_labels), predict_vec)

accuracy = cm.trace()/cm.sum()
print(accuracy)
