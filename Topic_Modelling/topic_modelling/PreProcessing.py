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

"""
PreProcessing.py contains functions that will be called by "HDP.py" and "LDA_GridSearch.py", in particular, the function 'Preprocessing'.

The main purpose of PreProcessing is to clean up raw textual data, and output a textual data that is more suitable to be fitted into the topic modelling algorithms.

Please ensure that you have installed spacy's language model "en_core_web_sm". Run "python -m spacy download en_core_web_sm" to install the language model.

"""

# We use NLTK's wordnet library for the removal of common stopwords. 
nltk.download('wordnet')

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """
    Function:
    ---------
        (1) Function conducts lemmatization on a sentence ("texts").

        (2) It will identify the Part Of Sentence tag, and only store the lemmmatized words of the sentence that are in the "allowed_postags"

    Args:
    -----
        (1) texts (list): A list where each element is a list with a single word inside. 
            (a) E.g. [['this'], ['is'], [], ['coffee'], ['review'], ['and'], [], ['feel'], ['like'], ['the'], ['coffee'], ['is'], ['very'], ['bad']] 

        (2) allowed_postags (list): A list of parts of speech tags, if a (lemmatized) word's POS tag is not in this list of POS tag, it will not be recorded in the output.  

    Returns:
    --------
        texts_out (list): A list of words that have been lemmatized and processed

    References:
    -----------
        [1] API Reference:
        https://spacy.io/usage/linguistic-features#pos-tagging 
        https://spacy.io/api/annotation#lemmatization
    """
    nlp = spacy.load('en_core_web_sm')
    texts_out = []
    for sent in texts:
        # Strips the list of one word into just a string
        strip_list = " ".join(sent)
        doc = nlp(strip_list)
        # If the token.lemma_ is not "-PRON-" (lemma for all personal pronouns), and if the token's POS tag is in the list of allowed_postags, in will be appended to texts_out
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def sent_to_words(sentences):
    """
    Function:
    ---------
        (1) Converts sentences into individual words using gensim's "gensim.utils.simple_preprocess"
    Args:
    -----
        (1) sentences (list): List of User review (str)

    Returns:
    --------
        str – Tokens from text
    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/utils.html#gensim.utils.simple_preprocess

    """
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  

def construct_stopwords(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    """
    Function:
    ---------
        (1) Constructs a list of stop words, consisting of: 
            (a) Brands of the coffee machine
            
            (b) Synonyms of "LIST_OF_COMMON_WORDS"

            (c) Words from "LIST_OF_ADDITIONAL_STOP_WORDS"

            (d) Words from NLTK's list of stopwords
    Args:
    -----
        (1) DF (pandas DataFrame): DataFrame of user reviews

        (2) LIST_OF_ADDITIONAL_STOP_WORDS (list): List of additional stop words

        (3) LIST_OF_COMMON_WORDS (list): List of common words, that you would want to not only remove the word from the review, but synonyms of the word as well.

    Returns:
    --------
        list_of_stop_words (list): List of Stopwords compiled by function

    References:
    -----------
        [1] API Reference:
        http://www.nltk.org/howto/wordnet.html

    """

    list_of_brands = DF["Brand"].unique()

    # If the word is in the list of common words, word and its synonyms will be added to list of stop words to be removed. 
    for word in LIST_OF_COMMON_WORDS:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                LIST_OF_ADDITIONAL_STOP_WORDS.append(l.name())

    #Remove brand name
    list_of_brands = [brand.lower() for brand in list_of_brands]
    LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS + list_of_brands

    # Add in stopwords from NLTK's set of English stopwords
    list_of_stop_words = set(stopwords.words('english'))
    for additional_word in LIST_OF_ADDITIONAL_STOP_WORDS:
        list_of_stop_words.add(additional_word)
    return list_of_stop_words

def Preprocessing(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    """
    Function:
    ---------
        (1) Function consolidates the entire preprocessing pipeline. It calls various smaller functions as part of the preprocessing pipeline, namely:

            (a) "construct_stopwords"

            (b) "sent_to_words"

            (c) "lemmatization"

        (2) The preprocessing pipeline consists of (in order):

            (a) Tokenization of sentence

            (b) Identifying bigrams (using gensim.models.phrases.Phrases)

            (c) Lemmatization of words

            (d) Removal of stopwords (including LIST_OF_ADDITIONAL_STOP_WORDS given by the user and LIST_OF_COMMON_WORDS where synonyms of the element of this list will be removed as well)

    Args:
    -----
        (1) df (pandas DataFrame): A pandas DataFrame of the raw data scraped from the web.

        (2) LIST_OF_ADDITIONAL_STOP_WORDS (list): A list of strings, where each element is an additional stop word that should be removed from the raw dataset as well
        
        (3) LIST_OF_COMMON_WORDS (list): A list of strings, containing common words, these common words, as well as their synonyms, will be added to the list of stop words.

    Returns:
    --------
        (1) processed_data_by_quarter_by_brand (OrderedDict): A nested Ordered Dictionary in the following format:
            {'positive':
                {'Quarter1':
                    {'Brand1':
                        [processed_user_review_1, processed_user_review_2, ...],
                        'Brand2':
                        ...},
                    'Quarter2':
                    {'Brand1':
                        [processed_user_review_A, processed_user_review_B, ...],
                        'Brand2':
                        ...},
                    ...},
                ...},
            'negative':
                ...}

            (a) First Layer: Type of Review (postive/negative)

            (b) Second Layer: List of Quarters

            (c) Third Layer: List of Brands of the Quarter

    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/phrases.html#gensim.models.phrases.Phrases
    """

    print("Processing raw text...")

    # Creates complete list of stop words by calling the function "construct_stopwords"
    stop_words = construct_stopwords(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    # Initialize empty ordered dictionary
    processed_data_by_quarter_by_brand = OrderedDict()

    # Create a new column "Y-Quarter", which determines the quarter that the user comment falls under
    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    df['Y-Quarter'] = df['Date'].dt.to_period("Q")
    list_of_quarters = df['Y-Quarter'].unique()
    list_of_brands = df["Brand"].unique()
    
    # Split the dataset into positive and negative comments, based on the rating
    df_positive = df[(df['Rating'] == 5) | (df['Rating'] == 4)]
    df_negative = df[(df['Rating'] == 1) | (df['Rating'] == 2)]

    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")

    # Iterate through the different types of reviews, quarters and brands
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
                # doc_of_quarter_by_brand is a list of user comments, where each element is an individual string of user comments
                doc_of_quarter_by_brand = df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)]["User Comment"].tolist()
                if doc_of_quarter_by_brand == []:
                    continue
                else:
                    # Convert user comments from sentences to words
                    doc_of_quarter_by_brand_token = list(sent_to_words(doc_of_quarter_by_brand))

                    # "gensim.models.phrases.Phrases" detects phrases based on collocation counts, combines words into phrases if it detects that it is a phrase.
                    bigram_brand = gensim.models.phrases.Phrases(doc_of_quarter_by_brand_token, min_count=3, threshold=10)
                    doc_of_quarter_by_brand_token = [bigram_brand[line] for line in doc_of_quarter_by_brand_token]

                    #Lemmatize words using "lemmatization"
                    doc_of_quarter_by_brand_lemma = lemmatization(doc_of_quarter_by_brand_token, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

                    #Remove stopwords that are found in the list "stop_words" and remove digits 
                    doc_of_quarter_by_brand_stop = []
                    for sentence in doc_of_quarter_by_brand_lemma:
                        sentence = ' '.join(word for word in sentence.split() if word not in stop_words and not word.isdigit())
                        doc_of_quarter_by_brand_stop.append(sentence)

                    processed_data_by_quarter_by_brand[type_of_review][str(quarter)][brand] = doc_of_quarter_by_brand_stop

        # Save nested dictionary as a pickle file
        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand'), 'wb') as handle:
            pickle.dump(processed_data_by_quarter_by_brand, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return processed_data_by_quarter_by_brand