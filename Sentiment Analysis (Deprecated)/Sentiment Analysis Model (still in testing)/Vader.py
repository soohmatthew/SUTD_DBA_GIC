# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 03:29:53 2018

@author: Sp_ceinvader
"""
import nltk
nltk.download('vader_lexicon')
import pandas as pd
excel = pd.ExcelFile('BestBuy Output.xlsx')
df = excel.parse('Review Data')
dataset = df.iloc[:,2]
reviewRating = df.iloc[:,4]

print (dataset)

def nltk_sentiment(sentence):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score

datasetlist = []
for i in range(len(dataset)):
    datasetlist.append(dataset[i])
   
reviewratinglist = []
for i in range(len(reviewRating)):
    reviewratinglist.append(reviewRating[i])
    
nltk_results = [nltk_sentiment(datasetlist[x]) for x in range(len(dataset))]
results_df = pd.DataFrame(nltk_results)
text_df = pd.DataFrame(datasetlist, columns = ['Reviews'])
rating_df = pd.DataFrame(reviewratinglist, columns = ['Corresponding Ratings'])
nltk_df = text_df.join(results_df)
nltkfinal_df=nltk_df.join(rating_df)

writer = pd.ExcelWriter('Sentiment.xlsx')
nltkfinal_df.to_excel(writer,'BestBuy Sentiment')
writer.save()