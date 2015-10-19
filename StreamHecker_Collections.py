import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re, math
from nltk.probability import FreqDist, ConditionalFreqDist
from nltk.corpus import stopwords
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

# move this up here
all_words = []
documents = []

#  j is adjective, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

#build frequency distibution of all words and
#then frequency distributions of words within positive and negative labels

word_fd = FreqDist()
cond_word_fd = ConditionalFreqDist()

stopset = set(stopwords.words('english'))

def evaluate_features(feature_select):

    posFeatures =[]
    negFeatures = []
    for i in short_pos.split('\n'):
        #remove punctuation
        posWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        posWords = [feature_select(posWords),'pos']
#        documents.append( (i, "pos") )
        documents.append(posWords)
        posFeatures.append(posWords)
        words = word_tokenize(i)
        pos = nltk.pos_tag(words)
        for w in pos:
            word_fd[w[0].lower()] += 1
            cond_word_fd['pos'][w[0].lower()] += 1
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())

    
    for i in short_neg.split('\n'):
#        documents.append( (p, "neg") )
        #remove punctuation
        negWords = re.findall(r"[\w']+|[.,!?;]", i.rstrip())
        negWords = [feature_select(negWords),'neg']
        documents.append(negWords)
        posFeatures.append(negWords)
        
        words = word_tokenize(i)
        pos = nltk.pos_tag(words)
        for w in pos:
            word_fd[w[0].lower()] += 1
            cond_word_fd['neg'][w[0].lower()] += 1
            if w[1][0] in allowed_word_types:
                all_words.append(w[0].lower())
    #selects 3/4 of the features to be used for training and 1/4 to be used for testing
    posCutoff = int(math.floor(len(posFeatures)*3/4))
    negCutoff = int(math.floor(len(negFeatures)*3/4))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    #trains a Naive Bayes Classifier
    classifier = NaiveBayesClassifier.train(trainFeatures)
    #initiates referenceSets and testSets
    referenceSets = collections.defaultdict(set)
    testSets = collections.defaultdict(set)	

    #puts correctly labeled sentences in referenceSets and the predictively labeled version in testsets
    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)
    #prints metrics to show how well the feature selection did
    print ('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print ('accuracy:', nltk.classify.util.accuracy(classifier, testFeatures))
    print ('pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos']))
    print ('pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos']))
    print ('neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg']))
    print ('neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg']))
    classifier.show_most_informative_features(10)

#BaseLine Bag of Words Feature Extraction
def word_feats(words):
    return dict([(word, True) for word in words])
#Stopword Filtering
def stopword_filtered_word_feats(words):
    return dict([(word, True) for word in words if word not in stopset])
#Bigram Collocations
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words,bigrams)])

#evaluate_features(word_feats)
#evaluate_features(stopword_filtered_word_feats)
evaluate_features(bigram_word_feats)
