#Standard library imports
import string
import pickle
from collections import OrderedDict
import os
import itertools

#Third party imports
import pandas as pd
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
import numpy as np
import spacy
import gensim
import openpyxl
import nltk
nltk.download('wordnet')

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def construct_stopwords(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    list_of_brands = DF["Brand"].unique()

    # If the word is in the list of common words, word and its synonyms will be added to list of stop words to be removed. 
    for word in LIST_OF_COMMON_WORDS:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                LIST_OF_ADDITIONAL_STOP_WORDS.append(l.name())

    #Remove brand name
    list_of_brands = [brand.lower() for brand in list_of_brands]
    LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS + list_of_brands

    list_of_stop_words = set(stopwords.words('english'))
    for additional_word in LIST_OF_ADDITIONAL_STOP_WORDS:
        list_of_stop_words.add(additional_word)
    return list_of_stop_words

def Preprocessing(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    print("Processing raw text...")
    stop_words = construct_stopwords(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)
    processed_data_by_quarter_by_brand = OrderedDict()

    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    df['Y-Quarter'] = df['Date'].dt.to_period("Q")
    list_of_quarters = df['Y-Quarter'].unique()
    list_of_brands = df["Brand"].unique()
    
    df_positive = df[(df['Rating'] == 5) | (df['Rating'] == 4)]
    df_negative = df[(df['Rating'] == 1) | (df['Rating'] == 2)]

    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")

    for type_of_review in ['positive', 'negative']:
        if type_of_review == 'positive':
            df = df_positive
        elif type_of_review == 'negative':
            df = df_negative
        processed_data_by_quarter_by_brand[type_of_review] = OrderedDict()
    
        for quarter in list_of_quarters:
            print("Processing {}...".format(quarter))
            processed_data_by_quarter_by_brand[type_of_review][str(quarter)] = OrderedDict()

            for brand in list_of_brands:
                doc_of_quarter_by_brand = df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)]["User Comment"].tolist()
                if doc_of_quarter_by_brand == []:
                    continue
                else:
                    doc_of_quarter_by_brand_token = list(sent_to_words(doc_of_quarter_by_brand))

                    bigram_brand = gensim.models.phrases.Phrases(doc_of_quarter_by_brand_token, min_count=3, threshold=10)
            
                    doc_of_quarter_by_brand_token = [bigram_brand[line] for line in doc_of_quarter_by_brand_token]

                    #Lemmatize words
                    doc_of_quarter_by_brand_lemma = lemmatization(doc_of_quarter_by_brand_token, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

                    #Remove stopwords, digits.
                    doc_of_quarter_by_brand_stop = []
                    for sentence in doc_of_quarter_by_brand_lemma:
                        sentence = ' '.join(word for word in sentence.split() if word not in stop_words and not word.isdigit())
                        doc_of_quarter_by_brand_stop.append(sentence)

                    processed_data_by_quarter_by_brand[type_of_review][str(quarter)][brand] = doc_of_quarter_by_brand_stop

        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand'), 'wb') as handle:
            pickle.dump(processed_data_by_quarter_by_brand, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return processed_data_by_quarter_by_brand