import json
import Sentiment_mod_2_classification as s
import operator
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
import re
import sys
import vincent

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
fname = 'GMO_Hanzhe_1.txt'

# this is where the fun actually starts :)
with open(fname,'r',encoding='utf8') as f:

    for line in f:
##        try:  
##            try:
##                tweet = json.loads(line)
##            except:
##                pass
##            
##            texts = [term for term in preprocess(tweet['text'])
##                     if term not in stop and
##                     not term.startswith(('#','@'))]
            texts = [term for term in preprocess(line)
                     if term not in stop and
                     not term.startswith(('#','@'))]

#            print("the texts is: ",texts)
            
            sentiment_value, confidence = s.sentiment(texts)
            print(texts,sentiment_value,confidence)

            if confidence*100 >= 80:
                output = open('GMO_Hanzhe_1_test.txt','a',encoding='utf-8')
                output.write('%s,%s,%s\n' %(texts, sentiment_value, confidence))
                output.close()

##
##        except:
##            pass
          
     


