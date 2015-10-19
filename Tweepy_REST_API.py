from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import csv
from csv import *
import operator
from collections import Counter
from nltk.corpus import stopwords
import string
from nltk import bigrams
from nltk import trigrams
from nltk.tokenize import word_tokenize
import re
import sys
import vincent




# Authentication details. To  obtain these visit dev.twitter.com
consumer_key = 'SQs0N4i8MLiZZf5j4KdjLkFNt'
consumer_secret = 'oNMf3qExMdxR9BASmGmEVQuPfA4K1j1RAhaT3CkB1Vfx3bwGi6'
access_token = '3286257269-2GxMXtboCXCUYBufiuY75FXqEfP44XmGU2IpjV4'
access_token_secret = 'zSjx5KhP7HOqEq77xAc2JUDwldDzLBcMDD93fcpHnA41I'

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
 
def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt','via']

# This is the listener, resposible for receiving data
class StdOutListener(StreamListener):
    
    def on_data(self, data):
        # Twitter returns data in JSON format - we need to decode it first
        tweet = json.loads(data)
        output = open('Baanyan_REST.txt','a',encoding='utf-8')
                  
        # Pull out various data from the tweets
        # tweet['text'].encode('utf-8','ignore')
        if(tweet['user']['lang'] == "en"):
            terms_all = [term for term in preprocess(tweet['text'])
                         if term not in stop and
                         not term.startswith(('#','@'))]
            #consider sequences of two terms
#            terms_bigram = bigrams(terms_all)
            output.write('%s,%s,%s\n' % (tweet['user']['screen_name'],tweet['user']['name'], terms_all,))
            output.write('\n')
#            csvWriter = csv.writer(output)
        output.close()  

        # Also, we convert UTF-8 to ASCII ignoring all bad characters sent by users
        #print ('@%s: %s' % (decoded['user']['screen_name'], decoded['text'].encode('ascii', 'ignore')))
        #print ('')
        return True

    def on_error(self, status):
        print (status)

#if __name__ == '__main__':
#    l = StdOutListener()
#    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#    auth.set_access_token(access_token, access_token_secret)

#    print ("Showing all new tweets for #programming:")

    # There are different kinds of streams: public stream, user stream, multi-user streams
    # In this example follow #programming tag
    # For more details refer to https://dev.twitter.com/docs/streaming-apis
#    stream = tweepy.Stream(auth, l)
#    stream.filter(track=['programming'])

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

twitterStream = Stream(auth, StdOutListener())
twitterStream.filter(track=['gmo','GMO'])
