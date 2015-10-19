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



# pos-tag data
documents = []

all_words = []

#  j is adjective, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

#build frequency distibution of all words and
#then frequency distributions of words within positive and negative labels

def evaluate_features(feature_select):

    posFeatures = []
    negFeatures = []
    
    training = []
    #process positive dataset "processed_pro_GMO.txt"
    for i in short_pos.split('\n'):
        posWords = word_tokenize(i)
        posWords_tag = [feature_select(posWords),"pos"]
        #post each word as "pos" in positive dataset
        posFeatures.append(posWords_tag)

        #tag word type for each word like, "NN","J", j is adjective, r is adverb, and v is verb
        pos = nltk.pos_tag(posWords)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
                documents.append("pos")

                
    #process negative dataset "processed_anti_GMO.txt"
    for i in short_neg.split('\n'):
        negWords = word_tokenize(i)
        negWords_tag = [feature_select(negWords),"neg"]
        negFeatures.append(negWords_tag)

        #tag word type for each word like, "NN","J", j is adjective, r is adverb, and v is verb
        pos = nltk.pos_tag(negWords)
        for w in pos:
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
                documents.append("neg")

                
    
    #get 10-Fold cross validation for Accuracy,Recall,Prediction
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
        BasedNaiveClassifier.show_most_informative_features(5)

        
        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)

##        #LogisticRegression
##        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
##        LogisticRegression_classifier.train(training[traincv[0]:traincv[len(traincv)-1]])
##        Regression_Accuracy = (nltk.classify.util.accuracy(LogisticRegression_classifier, training[testcv[0]:testcv[len(testcv)-1]]))*100
##        Regression_Accu += Regression_Accuracy
##
##
##        #LinearSVC
##        LinearSVC_classifier = SklearnClassifier(LinearSVC())
##        LinearSVC_classifier.train(training[traincv[0]:traincv[len(traincv)-1]])
##        SVC_Accuracy = (nltk.classify.util.accuracy(LinearSVC_classifier, training[testcv[0]:testcv[len(testcv)-1]]))*100
##        SVC_Accu += SVC_Accuracy
        
        

        for idx in testcv:
            testFeatures.append(training[idx])
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = BasedNaiveClassifier.classify(features)
            testSets[predicted].add(i)  
#7/5/2015        
        pos_Precision += (nltk.metrics.precision(referenceSets["pos"], testSets["pos"]))*100     
        pos_recall += (nltk.metrics.recall(referenceSets["pos"], testSets["pos"]))*100 
        neg_Precision += (nltk.metrics.precision(referenceSets["neg"], testSets["neg"]))*100
        neg_recall += (nltk.metrics.recall(referenceSets["neg"], testSets["neg"]))*100

        precision["pos"] = nltk.metrics.precision(referenceSets["pos"], testSets["pos"])     
        recall["pos"] = nltk.metrics.recall(referenceSets["pos"], testSets["pos"]) 
        precision["neg"] = nltk.metrics.precision(referenceSets["neg"], testSets["neg"])
        recall["neg"] = nltk.metrics.recall(referenceSets["neg"], testSets["neg"])

#get Average score for Accuracy, Precision and Recall
    accu = Naive_Accu/num_folds
#7/5/2015
    pos_Precision = pos_Precision/num_folds
    pos_recall = pos_recall/num_folds
    neg_Precision = neg_Precision/num_folds
    neg_recall = neg_recall/num_folds
    print("Average Naive Bayes Accuracy is:", accu)
#7/5/2015
    print("Average Positive Precision is:", pos_Precision)
    print("Average Positive Recall is:", pos_recall)
    print("Average Negative Precision is:", neg_Precision)
    print("Average Negative Recall is:", neg_recall)

##    Regression_Accu = Regression_Accu/num_folds
##    print("LogisticRegression_classifier accuracy percent:", Regression_Accu)
##
##    SVC_Accu = SVC_Accu/num_folds
##    print("LinearSVC_classifier accuracy percent:", SVC_Accu)



#Unigrams//BaseLine Bag of Words Feature Extraction 
def word_feats(words):
    return dict([(word, True) for word in words])
#Unigrams// Remove Stopwords Feature Extraction
def word_stop(words):
    return dict([word, True] for word in words if not word in stopwords.words('english'))

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words,bigrams)])

#evaluate_features(word_stop)
evaluate_features(bigram_word_feats)
##################

#all_words = nltk.FreqDist(all_words)
#word_features = list(all_words.keys())[:5000]

#def find_features(document):
#    words = word_tokenize(document)
#    features = {}
#    for w in word_features:
#        features[w] = (w in words)

#    return features
#featuresets = [(find_features(rev), category) for (rev, category) in documents]

#random.shuffle(featuresets)
#print(len(featuresets))

#testing_set = featuresets[10000:]
#training_set = featuresets[:10000]






