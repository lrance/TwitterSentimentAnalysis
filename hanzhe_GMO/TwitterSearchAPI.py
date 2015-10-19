from TwitterSearch import *
import csv
from csv import *
#from csv import writer
#import json
#import sys
import re
try:
    tso = TwitterSearchOrder() # create a TwitterSearchOrder object
# let's define all words we would like to have a look for
    tso.set_keywords(['gmo', 'GMO',"gmo labeling", "gmo risk"])
# we want to see English tweets only
    tso.set_language('en')
# and don't give us all those entity information
    tso.set_include_entities(True)

# it's about time to create a TwitterSearch object with our secret tokens
    ts = TwitterSearch(
        consumer_key = 'SQs0N4i8MLiZZf5j4KdjLkFNt',
        consumer_secret = 'oNMf3qExMdxR9BASmGmEVQuPfA4K1j1RAhaT3CkB1Vfx3bwGi6',
        access_token = '3286257269-2GxMXtboCXCUYBufiuY75FXqEfP44XmGU2IpjV4',
        access_token_secret = 'zSjx5KhP7HOqEq77xAc2JUDwldDzLBcMDD93fcpHnA41I'
     )
    
# this is where the function actually starts :)
    for tweet in ts.search_tweets_iterable(tso):     
          
          output = open('GMO_Hanzhe_1.txt','a',encoding='utf-8')        
          tweetRemove = tweet['text']
          tweetRemove = ' '.join(re.sub("(RT)|(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweetRemove).split())
          output.write('%s\n' % (tweetRemove,))
          output.close()       

except TwitterSearchException as e: # take care of all those ugly errors if there are some
    print(e)
