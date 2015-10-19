import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


word_features10k_f = open("GMO_Hanzhe/word_features10k.pickle", "rb")
word_features = pickle.load(word_features10k_f)
word_features10k_f.close()


def find_features(document):
#    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in document)
#        print(features)

    return features



open_file = open("GMO_Hanzhe/BasedNaiveClassifier10k.pickle", "rb")
BasedB_classifier = pickle.load(open_file)
open_file.close()


open_file = open("GMO_Hanzhe/LogisticRegression_classifier10k.pickle", "rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()



open_file = open("GMO_Hanzhe/LinearSVC_classifier10k.pickle", "rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()


voted_classifier = VoteClassifier(
                                  BasedB_classifier,
                                  LogisticRegression_classifier,
                                  LinearSVC_classifier
                                  )


def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
