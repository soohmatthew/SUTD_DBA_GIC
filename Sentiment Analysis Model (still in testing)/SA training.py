# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:42:05 2018

@author: clare
"""
'''
import tarfile

tar = tarfile.open("unprocessed.tar.gz", "r:gz")
tar.extractall()
tar.close()
print(tar)
'''
import pandas as pd
import spacy
nlp = spacy.load('en') 
from spacy import displacy 
from spacy.lang.en.stop_words import STOP_WORDS
'''
import tarfile
import numpy as np 

df_amz = tarfile.open("unprocessed.tar.gz", "r:gz")
for member in df_amz.getmembers():
     f = df_amz.extractfile(member)
     if f:
         content = f.read()
         Data = np.loadtxt(content)
print(df_amz)         
'''
# Loading our dataset from a textfile from the internet
df_amz = pd.read_table('amazon_cells_labelled.txt')

# Concatenate our Datasets
frames = [df_amz]
# Renaming Column Headers
for colname in frames:
    colname.columns = ["Message","Target"]
review = (df_amz['Message'])
# print(review)
# length of data set
length = df_amz.shape[0] - 1
list = []
for i in range (0, length):
    list.append(review[i])
list = "".join(list)
reviews = nlp(list)
# print(reviews)
'''
# Lemmatizing of tokens
for word in reviews:
    if word.lemma_ != "-PRON-":
        print(word.lemma_.lower().strip())
'''
# dictionary of stop words
stopwords = (STOP_WORDS)
# print(stopwords)

# Filtering out Stopwords and Punctuations
list2 = []
for word in reviews:
    if word.is_stop == False and not word.is_punct:
#     if word.is_stop != True and not word.is_punct:
        list2.append(word)
# Use the punctuations of string module
import string
punctuations = string.punctuation
# Creating a Spacy Parser
from spacy.lang.en import English
parser = English()  
def spacy_tokenizer(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    return mytokens

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import accuracy_score 
from sklearn.base import TransformerMixin 
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

#Custom transformer using spaCy 
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}

# Basic function to clean the text 
def clean_text(text):     
    return text.strip().lower()

# Vectorization
vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1)) 
classifier = LinearSVC()

'''
# Using Tfidf
tfvectorizer = TfidfVectorizer(tokenizer = spacy_tokenizer)
'''

# Splitting Data Set
from sklearn.model_selection import train_test_split

# Features and Labels
X = df_amz['Message']
ylabels = df_amz['Target']
X_train, X_test, y_train, y_test = train_test_split(X, ylabels, test_size=0.2, random_state=42)

# Create the  pipeline to clean, tokenize, vectorize, and classify 
pipe = Pipeline([("cleaner", predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])
# Fit our data
pipe.fit(X_train,y_train)

our_data = pd.read_excel('Customer Reviews of coffee machine.xlsx')
reviews = (our_data['User Comment'])

# Predicting with a test dataset
sample_prediction = pipe.predict(X_test)

# Prediction Results
# 1 = Positive review
# 0 = Negative review
for (sample,pred) in zip(X_test,sample_prediction):
    print(sample,"Prediction=>",pred)

# Accuracy
# print("Accuracy: ",pipe.score(X_test,y_test))
# print("Accuracy: ",pipe.score(X_test,sample_prediction))


# Accuracy
# print("Accuracy: ",pipe.score(X_train,y_train))

# group by product
# group by rating
# perform SA
# get results like most appeared key words => positive or negative
# what kind of ideas can we derive from the key words 
# years with rating 5 would mean long lasting?
'''
print (reviews[1])
print (pipe.predict(reviews))
for i in reviews:
    score= (pipe.predict(i))
    

for i in reviews:
    i = spacy_tokenizer(i)
    ' '.join(i)
    print(i)
count_of_names = our_data.groupby(['Name']).size().reset_index(name='count')
count_of_ratings = our_data.groupby(['Rating']).size().reset_index(name='count')
# print(count_of_names)
# print(count_of_ratings)
counts_only = count_of_names['count']
list = []
for i in counts_only:
    list.append(i)
# print(list)

def add_one_by_one(l):
    new_l = []
    cumsum = 0
    for elt in l:
        cumsum += elt
        new_l.append(cumsum)
    return new_l    

new_list = add_one_by_one(list)
new_list.insert(0, 0)

score_list=[]
for j in range(len(new_list)):
    for i in range(new_list[j], new_list[j+1]):
        score = 0
        score += pipe.predict(reviews[i])
        score_list.append(score)
print(score_list)


# print(new_list)
# print(pipe.predict(reviews))




example = ["I do enjoy my job",
 "What a poor product!,I will have to get a new one",
 "I feel amazing!"]

print(pipe.predict(example))
'''
