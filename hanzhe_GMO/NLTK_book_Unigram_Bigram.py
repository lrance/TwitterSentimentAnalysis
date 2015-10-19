import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import itertools
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
 
def evaluate_classifier(featx):
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')
 
    negfeats = [(featx(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(featx(movie_reviews.words(fileids=[f])), 'pos') for f in posids]
 
    negcutoff = int(len(negfeats)*3/4)
    poscutoff = int(len(posfeats)*3/4)
 
    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
 
    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)
 
    for i, (feats, label) in enumerate(testfeats):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
    print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))
    print ('pos precision:', nltk.metrics.precision(refsets['pos'], testsets['pos']))
    print ('pos recall:', nltk.metrics.recall(refsets['pos'], testsets['pos']))
    print ('neg precision:', nltk.metrics.precision(refsets['neg'], testsets['neg']))
    print ('neg recall:', nltk.metrics.recall(refsets['neg'], testsets['neg']))
    classifier.show_most_informative_features()

def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
 
evaluate_classifier(bigram_word_feats)
