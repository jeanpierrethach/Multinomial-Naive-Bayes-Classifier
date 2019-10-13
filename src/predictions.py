import numpy as np
from preprocess import preprocess

import csv

from vectorizer import CountVectorizer, TfidfVectorizer
from multinb import MultinomialNaiveBayes

data = np.load("../data_train.pkl", allow_pickle=True)
test_data = np.load("../data_test.pkl", allow_pickle=True)

train_set = data[0]
train_labels = data[1]

corpus = preprocess(train_set)

#vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(corpus)
y = np.array(train_labels)

naive = MultinomialNaiveBayes(alpha=0.52)
classifier = naive.fit(X, y)


with open("predictions.csv", 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(["Id", "Category"])
    
    predictions = classifier.predict(vectorizer.transform(np.array(test_data)))
    for i, pred in enumerate(predictions):
        wr.writerow((i,pred))