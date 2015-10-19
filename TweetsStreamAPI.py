from tweepy import Stream
import csv
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import urllib
import json
import sys
from csv import writer
#import sentiment_mod as s

#consumer key, consumer secret, access token, access secret.
ckey="SQs0N4i8MLiZZf5j4KdjLkFNt"
csecret="oNMf3qExMdxR9BASmGmEVQuPfA4K1j1RAhaT3CkB1Vfx3bwGi6"
atoken="3286257269-2GxMXtboCXCUYBufiuY75FXqEfP44XmGU2IpjV4"
asecret="zSjx5KhP7HOqEq77xAc2JUDwldDzLBcMDD93fcpHnA41I"

#from twitterapistuff import *

sentdexAuth=''

def sentimentAnalysis(text):
    encoded_text = urllib.quote(text)
    API_Call = 'http://sentdex.com/api/api.php?text='+encoded_text+'$auth='+sentdexAuth
    output = urllib.urlopen(API_Call).read()  
    return output


class listener(StreamListener):

    def on_data(self, data):
#        tweet = data.split(',"text":"')[1].split('","source')[0]
#test:        tweet = data.split(',"text":"')[0]
#test:        tweet = data.split('","source')[0]
#test:        print(data)
#version2:        all_data = json.loads(data)
#version2:        tweet = all_data["text"]
#        sentimentRating = sentimentAnalysis(tweet)
#version2:        sentiment_value, confidence = s.sentiment(tweet)
#version2:        print(tweet, sentiment_value, confidence)

#        saveMe = tweet+'::' + sentimentRating +'\n'
        output = open('Baanyan.txt','a')
#        output.write(saveMe)
        output.write(data)
        csvWriter = csv.writer(output)
        output.close()
#version2:        if confidence*100 >= 80:
#            output = open("twitter-out.txt","a")
#            output.write(sentiment_value)
#            output.write('\n')
#            output.close()
        
        
        return(True)

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["gmo","GMO"])
