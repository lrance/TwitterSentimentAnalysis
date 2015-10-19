import nltk
import random
from sklearn import cross_validation
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist, ConditionalFreqDist
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import re, math
import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.corpus import stopwords
import numpy as np
from scipy import interp
import pylab as pl
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline




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
    
short_pos = open("hanzhe_GMO_DataSet/SearchAPI_hanzheData_high_low_pro.txt","r").read()
short_neg = open("hanzhe_GMO_DataSet/SearchAPI_hanzheData_high_low_con.txt","r").read()



# move this up here
all_words = []
# pretend as iris.target
documents = []

#  j is adjective, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

#build frequency distibution of all words and
#then frequency distributions of words within positive and negative labels



posFeatures = []
negFeatures = []
    
training = []
#process positive dataset "processed_pro_GMO.txt"
for i in short_pos.split('\n'):
    posWords = word_tokenize(i)
    posWords_tag = [feature_select(posWords),"pos"]
    #post each word as "pos" in positive dataset
    posFeatures.append(posWords_tag)


                
#process negative dataset "processed_anti_GMO.txt"
for i in short_neg.split('\n'):
    negWords = word_tokenize(i)
    negWords_tag = [feature_select(negWords),"neg"]
    negFeatures.append(negWords_tag)


                    
#get 6-Fold cross validation for Accuracy,Recall,Prediction
num_folds = 6
training = posFeatures + negFeatures
cv = cross_validation.KFold(len(training),n_folds=6, shuffle=True, random_state=None)

Naive_Accu = 0
neg_Precision = 0
neg_recall = 0
pos_Precision = 0
pos_recall = 0

SVC_Accu = 0
Regression_Accu = 0
testFeatures = []

precision = dict()
recall = dict()
average_Precision = dict()

for traincv, testcv in cv:
    #BasedNaiveClassifier
    BasedNaiveClassifier = NaiveBayesClassifier.train(training[traincv[0]:traincv[len(traincv)-1]])
    accuracy = (nltk.classify.util.accuracy(BasedNaiveClassifier, training[testcv[0]:testcv[len(testcv)-1]]))*100
    Naive_Accu += accuracy
    BasedNaiveClassifier.show_most_informative_features(10)


    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

     parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
     
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(traincv, testcv)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = grid_search.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)
    
   


