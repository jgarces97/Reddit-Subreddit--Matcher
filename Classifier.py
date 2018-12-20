#!/usr/bin/env python3

import nltk
nltk.download('popular')

import json
import random
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt


stemmer = SnowballStemmer("english", ignore_stopwords=True)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer()


def supportVectorClass(data, target, testing_data, testing_target):

    text_clf_svc = Pipeline([('vect', CountVectorizer()),
                         ('clf-svc', SVC(gamma='auto'))
                         ])

    text_clf_svc = text_clf_svc.fit(data, target)
    return text_clf_svc.score(testing_data, testing_target)


def randomForestClass(data, target, testing_data, testing_target):

    text_clf_rfc = Pipeline([('vect', CountVectorizer()),
                             ('clf', RandomForestClassifier(n_estimators=100))
                             ])
    text_clf_rfc = text_clf_rfc.fit(data, target)
    return text_clf_rfc.score(testing_data, testing_target)


def supportVectorClass_stemming(data, target, testing_data, testing_target):

    text_clf_svc = Pipeline([('vect', stemmed_count_vect),
                         ('clf-svc', SVC(gamma='auto'))
                         ])

    text_clf_svc = text_clf_svc.fit(data, target)
    return text_clf_svc.score(testing_data, testing_target)


def randomForestClass_stemming(data, target, testing_data, testing_target):

    text_clf_rfc = Pipeline([('vect', stemmed_count_vect),
                             ('clf', RandomForestClassifier(n_estimators=100))
                             ])
    text_clf_rfc = text_clf_rfc.fit(data, target)
    return text_clf_rfc.score(testing_data, testing_target)


def randomAmountList(data, target, amount):

    c = list(zip(data, target))
    random.shuffle(c)
    data, target = zip(*c)
    return data[:amount], target[:amount]


def plotRFCLearningCurve(data, target, test_data, test_target):

    X = list()
    Y = list()
    x = 10
    while x < len(data):
        ran_data, ran_traget = randomAmountList(data, target, x)
        X.append(x)
        Y.append(randomForestClass_stemming(ran_data, ran_traget, test_data, test_target) * 100)
        x += int(len(data)/25)

    plt.plot(X, Y)
    plt.axis([0, len(data), 0, 100])
    plt.title('Amount of Training Data vs. Mean Accuracy on Random Forest Classifier')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Amount of Training Data')
    plt.show()


def plotSVCLearningCurve(data, target, test_data, test_target):
    X = list()
    Y = list()
    x = 10
    while x < len(data):
        ran_data, ran_traget = randomAmountList(data, target, x)
        X.append(x)
        Y.append(supportVectorClass_stemming(ran_data, ran_traget, test_data, test_target) * 100)
        x += int(len(data)/25)

    plt.plot(X, Y)
    plt.axis([0, len(data), 0, 100])
    plt.title('Amount of Training Data vs. Mean Accuracy on Support Vector Classifier')
    plt.ylabel('Mean Accuracy')
    plt.xlabel('Amount of Training Data')
    plt.show()


if __name__ == '__main__':

    if len(sys.argv) is not 3:
        print('Usage ./Classifier <training json file> <testing json file>')

    train = json.load(open(sys.argv[1], 'r'))
    train_data = train[0]
    train_target = train[1]

    development = json.load(open(sys.argv[2], 'r'))
    dev_data = development[0]
    dev_target = development[1]

    print('Support Vector Classifier Mean Accuracy(No Stemming): ' +
          str(supportVectorClass(train_data, train_target, dev_data, dev_target) * 100) + '%')

    print('Random Forest Classifier Mean Accuracy(No Stemming): ' +
          str(randomForestClass(train_data, train_target, dev_data, dev_target) * 100) + '%')

    print('Support Vector Classifier Mean Accuracy(With Stemming): ' +
          str(supportVectorClass_stemming(train_data, train_target, dev_data, dev_target) * 100) + '%')

    print('Random Forest Classifier Mean Accuracy(With Stemming): ' +
          str(randomForestClass_stemming(train_data, train_target, dev_data, dev_target) * 100) + '%')

    plotRFCLearningCurve(train_data, train_target, dev_data, dev_target)
    plotSVCLearningCurve(train_data, train_target, dev_data, dev_target)


