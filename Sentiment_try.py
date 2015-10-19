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
    
short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()
#short_pos = open('short_reviews/pro_Movie_GMO','r',errors='ignore').read()
#short_neg = open('short_reviews/anti_Movie_GMO','r',errors='ignore').read()

# move this up here
all_words = []
# pretend as iris.target
documents = []

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
        BasedNaiveClassifier.show_most_informative_features(50)

        
        #initiates referenceSets and testSets
        referenceSets = collections.defaultdict(set)
        testSets = collections.defaultdict(set)

        #LogisticRegression
        LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
        LogisticRegression_classifier.train(training[traincv[0]:traincv[len(traincv)-1]])
        Regression_Accuracy = (nltk.classify.util.accuracy(LogisticRegression_classifier, training[testcv[0]:testcv[len(testcv)-1]]))*100
        Regression_Accu += Regression_Accuracy

        save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
        pickle.dump(LogisticRegression_classifier, save_classifier)
        save_classifier.close()

        #LinearSVC
        LinearSVC_classifier = SklearnClassifier(LinearSVC())
        LinearSVC_classifier.train(training[traincv[0]:traincv[len(traincv)-1]])
        SVC_Accuracy = (nltk.classify.util.accuracy(LinearSVC_classifier, training[testcv[0]:testcv[len(testcv)-1]]))*100
        SVC_Accu += SVC_Accuracy
        
        save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
        pickle.dump(LinearSVC_classifier, save_classifier)
        save_classifier.close()

        

        for idx in testcv:
            testFeatures.append(training[idx])
        #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
        for i, (features, label) in enumerate(testFeatures):
            referenceSets[label].add(i)
            predicted = BasedNaiveClassifier.classify(features)
            testSets[predicted].add(i)  
#7/5/2015        
##        pos_Precision += (nltk.metrics.precision(referenceSets["pos"], testSets["pos"]))*100     
##        pos_recall += (nltk.metrics.recall(referenceSets["pos"], testSets["pos"]))*100 
##        neg_Precision += (nltk.metrics.precision(referenceSets["neg"], testSets["neg"]))*100
##        neg_recall += (nltk.metrics.recall(referenceSets["neg"], testSets["neg"]))*100
##
##        precision["pos"] = nltk.metrics.precision(referenceSets["pos"], testSets["pos"])     
##        recall["pos"] = nltk.metrics.recall(referenceSets["pos"], testSets["pos"]) 
##        precision["neg"] = nltk.metrics.precision(referenceSets["neg"], testSets["neg"])
##        recall["neg"] = nltk.metrics.recall(referenceSets["neg"], testSets["neg"])
##
##        save_classifier = open("GMOHedging/BasedNaiveClassifier.pickle","wb")
##        pickle.dump(BasedNaiveClassifier, save_classifier)
##        save_classifier.close()
###    average_precision["pos"] = precision["pos"]
##

    #get Average score for Accuracy, Precision and Recall
    accu = Naive_Accu/num_folds
#7/5/2015
##    pos_Precision = pos_Precision/num_folds
##    pos_recall = pos_recall/num_folds
##    neg_Precision = neg_Precision/num_folds
##    neg_recall = neg_recall/num_folds
    print("Average Naive Bayes Accuracy is:", accu)
#7/5/2015
##    print("Average Positive Precision is:", pos_Precision)
##    print("Average Positive Recall is:", pos_recall)
##    print("Average Negative Precision is:", neg_Precision)
##    print("Average Negative Recall is:", neg_recall)

    Regression_Accu = Regression_Accu/num_folds
    print("LogisticRegression_classifier accuracy percent:", Regression_Accu)

    SVC_Accu = SVC_Accu/num_folds
    print("LinearSVC_classifier accuracy percent:", SVC_Accu)

    

#    selects 3/4 of the features to be used for training and 1/4 to be used for testing
##    posCutoff = int(math.floor(len(posFeatures)*3/4))
##    negCutoff = int(math.floor(len(negFeatures)*3/4))
##    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
##    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

##   trains a Naive Bayes Classifier
##    classifier = NaiveBayesClassifier.train(trainFeatures)
##   trains a LinearSVC Classifier
##    classifier = SklearnClassifier(LinearSVC())
##    classifier.train(trainFeatures)
##   
##    accuracy = (nltk.classify.util.accuracy(classifier, testFeatures))*100
##    print("Accuracy for LinearSVC again: ", accuracy)
    

##    print("overAll Accuracy is: ", accuracy)
##    save_classifier = open("GMOHedging/classifier.pickle","wb")
##    pickle.dump(classifier, save_classifier)
##    save_classifier.close()

#    referenceSets = collections.defaultdict(set)
#    testSets = collections.defaultdict(set)	
#   puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
#    for i, (features, label) in enumerate(testFeatures):
#        referenceSets[label].add(i)
#        predicted = classifier.classify(features)
#        testSets[predicted].add(i)	

    #prints metrics to show how well the feature selection did
##    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
##    print ('accuracy:', (nltk.classify.util.accuracy(classifier, testFeatures))*100)
##    print ('pos precision:', nltk.metrics.precision(referenceSets["pos"], testSets["pos"]))
##    print ('pos recall:', nltk.metrics.recall(referenceSets["pos"], testSets["pos"]))
##    print ('neg precision:', nltk.metrics.precision(referenceSets["neg"], testSets["neg"]))
##    print ('neg recall:', nltk.metrics.recall(referenceSets["neg"], testSets["neg"]))


#pickled_StreamHecker/originalnaivebayes.pickle
##    save_classifier = open("GMOHedging/originalnaivebayes.pickle","wb")
##    pickle.dump(classifier, save_classifier)
##    save_classifier.close()

##    MNB_classifier = SklearnClassifier(MultinomialNB())
##    MNB_classifier.train(trainFeatures)
##    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testFeatures))*100)
##
##    save_classifier = open("GMOHedging/MNB_classifier.pickle","wb")
##    pickle.dump(MNB_classifier, save_classifier)
##    save_classifier.close()

##    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
##    BernoulliNB_classifier.train(trainFeatures)
##    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testFeatures))*100)

##    save_classifier = open("GMOHedging/BernoulliNB_classifier.pickle","wb")
##    pickle.dump(BernoulliNB_classifier, save_classifier)
##    save_classifier.close()

#builds dictionary of word scores based on chi-squared test
def create_word_scores():

    posWord_score = []
    negWord_score = []

    for i in short_pos.split('\n'):
        posWords = word_tokenize(i)
        posWord_score.append(posWords)

    for i in short_neg.split('\n'):
        negWords = word_tokenize(i)
        negWord_score.append(negWords)
        
    word_scores = {}

    posWord_score = list(itertools.chain(*posWord_score))
    negWord_score = list(itertools.chain(*negWord_score))

    #build frequency distibution of all words and then frequency distributions of words within positive and negative labels
    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()
    
    for word in posWord_score:
        word_fd[word.lower()] += 1
        cond_word_fd["pos"][word.lower()] += 1
    for word in negWord_score:
        word_fd[word.lower()] += 1
        cond_word_fd["neg"][word.lower()] += 1
        
    #finds the number of positive and negative words, as well as the total number of words
    pos_word_count = cond_word_fd["pos"].N()
    neg_word_count = cond_word_fd["neg"].N()
    total_word_count = pos_word_count + neg_word_count

    #Chi-Squared Informative Gain
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd["pos"][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd["neg"][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score
        
    return word_scores



#finds word scores
word_scores = create_word_scores()

#Bigram Collocations
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words,bigrams)])

#BaseLine Bag of Words Feature Extraction
def word_feats(words):
    return dict([(word, True) for word in words])
#Remove Stopwords Feature Extraction
def word_stop(words):
    return dict([word, True] for word in words if not word in stopwords.words('english'))

#evaluate_features(word_stop)

#finds the best 'number' words based on word scores
def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda ws: ws[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])

    save_word_features = open("GMOHedging/word_features10k.pickle","wb")
    pickle.dump(best_words, save_word_features)
    save_word_features.close()
    return best_words

#creates feature selection mechanism that only uses best words
def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])

#numbers of features to select
#numbers_to_test = [10, 100, 1000, 10000, 15000]
#tries the best_word_features mechanism with each of the numbers_to_test of features
#for num in numbers_to_test:
#    print ('evaluating best %d word features' % (num))
#    best_words = find_best_words(word_scores, num)
#    evaluate_features(best_word_features)

#number of features set as 10000 default
best_words = find_best_words(word_scores, 10000)
evaluate_features(best_word_features)
###############
#    classifier = nltk.NaiveBayesClassifier.train(trainFeatures)
#    print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testFeatures))*100)
#    classifier.show_most_informative_features(15)

#    save_classifier = open("pickled_StreamHecker/originalnaivebayes5k.pickle","wb")
#    pickle.dump(classifier, save_classifier)
#    save_classifier.close()

#    MNB_classifier = SklearnClassifier(MultinomialNB())
#    MNB_classifier.train(trainFeatures)
#    print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testFeatures))*100)

#    save_classifier = open("pickled_StreamHecker/MNB_classifier5k.pickle","wb")
#    pickle.dump(MNB_classifier, save_classifier)
#    save_classifier.close()

#    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#    BernoulliNB_classifier.train(trainFeatures)
#    print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testFeatures))*100)

#    save_classifier = open("pickled_StreamHecker/BernoulliNB_classifier5k.pickle","wb")
#    pickle.dump(BernoulliNB_classifier, save_classifier)
#    save_classifier.close()

   

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

#builds dictionary of word scores based on TF-IDF test (undeveloped)
def create_word_scores_TFIDF():
    tfidf = TfidfVectorizer(tokenizer=all_words, stop_words='english')
    tfs = tfidf.fit_transform(token_dict.values())
    return tfidf






