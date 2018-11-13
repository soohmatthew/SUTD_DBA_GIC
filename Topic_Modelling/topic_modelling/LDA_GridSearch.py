#Standard library imports
import string
import pickle
from collections import OrderedDict
import os
import itertools
import sys

#Third party imports
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
import spacy
import gensim
import openpyxl

#Python File Imports
sys.path.append(os.getcwd())
from Topic_Modelling.topic_modelling.PreProcessing import Preprocessing

# Corpus Download necessary to run text processing
import nltk
nltk.download('wordnet')

class LDAUsingPerplexityScorer(LatentDirichletAllocation):
    def score(self, X, y=None):
        score = super().perplexity(X, sub_sampling=False)
        # Perplexity is lower for better, negative scoring to simulate that.
        return -1*score

def build_single_LDA_model(dict_of_clean_doc, quarter, brand, type_of_review, number_of_topics_range):
    try:
        vectorizer = CountVectorizer(analyzer='word',       
                                min_df= 3,                        # minimum required occurence of a word 
                                stop_words='english',             # remove stop words
                                lowercase=True,                   # convert all words to lowercase
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )
        print("Building LDA model for ... {}, {} ".format(str(quarter), brand))
        doc_clean = dict_of_clean_doc[type_of_review][str(quarter)][brand]

        data_vectorized = vectorizer.fit_transform(doc_clean)

        lda_model = LDAUsingPerplexityScorer(max_iter=10,               # Max learning iterations
                                            learning_method='online',   
                                            random_state=100,          # Random state
                                            batch_size=128,            # n docs in each learning iter
                                            evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                            n_jobs = -1,               # Use all available CPUs
                                            )

        search_params = {'n_components': number_of_topics_range, 'learning_decay': [.5, .7, .9]}
        grid_search_model = GridSearchCV(lda_model, param_grid=search_params)
        grid_search_model.fit(data_vectorized)

        best_lda_model = grid_search_model.best_estimator_

        #Creates a dictionary, where the key is the nth topic, 
        #and the corresponding value is a dictionary, with a 'keyword' key, and 
        #list of tuples of the top 10 keyword and the associated weight as the value

        def show_topics(vectorizer, lda_model, n_words=10):
            keywords = np.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in lda_model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]

                # Get top 10 keywords and the associated weight
                top_keyword_weight = sorted(-topic_weights, key=float)[:n_words]
                top_keyword_list = keywords.take(top_keyword_locs).tolist()
                topic_keywords.append([(top_keyword_list[i], -top_keyword_weight[i]) for i in range(len(top_keyword_weight))])

            topic_keywords_dict = OrderedDict()
            for topic in topic_keywords:
                topic_keywords_dict["topic " + str(topic_keywords.index(topic)+1)] = {'keywords':topic}
            return topic_keywords_dict

        topic_keywords_dict = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

        #Creates a pandas dataframe to count the frequency of the nth topic being the main topic of a document.
        #Count the number of time topic n is the main topic

        lda_output = best_lda_model.transform(data_vectorized)
        # column names
        topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
        # index names
        docnames = ["Doc" + str(i) for i in range(len(doc_clean))]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1).tolist()

        #Count of dominant topic of each document, and add it to the topic_keywords_dict constructed earlier
        for topic_number in range(best_lda_model.n_components):
            topic_keywords_dict['topic ' + str(topic_number+1)]['frequency'] = dominant_topic.count(topic_number)

        topic_model_df = pd.DataFrame()
        for topic in topic_keywords_dict:
                for keyword in topic_keywords_dict[topic]['keywords']:
                    keyword_dict = OrderedDict()
                    keyword_dict['Quarter'] = str(quarter)
                    keyword_dict['Brand'] = brand
                    keyword_dict['Type of Review'] = type_of_review.capitalize()
                    keyword_dict['Topic'] = topic
                    keyword_dict['Topic Frequency'] = topic_keywords_dict[topic]['frequency']
                    keyword_dict['Keyword'] = keyword[0]
                    keyword_dict['Keyword Weight'] = keyword[1]
                    topic_model_df = topic_model_df.append(keyword_dict, ignore_index=True)
        return topic_model_df
    except ValueError:
        return pd.DataFrame()

def LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, number_of_topics_range):
    #Read in processed documents from cache, or process new document
    if os.path.isfile('pickle_files/processed_data_by_quarter_by_brand.pickle'): 
        with open('pickle_files/processed_data_by_quarter_by_brand.pickle', 'rb') as handle_2:
            dict_of_clean_doc_by_quarter_by_brand = pickle.load(handle_2)
    else:
        dict_of_clean_doc_by_quarter_by_brand = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    #Generate list of quarters
    DF['Date'] = pd.to_datetime(DF['Date'],infer_datetime_format=True)
    DF['Y-Quarter'] = DF['Date'].dt.to_period("Q")
    list_of_quarters = DF['Y-Quarter'].unique()
    
    #Limit quarters to those in 2016, 2017, 2018
    list_of_quarters = [quarter for quarter in list_of_quarters if any(year in str(quarter) for year in LIST_OF_YEARS_TO_INCLUDE)]
    
    combination_of_brands = []
    for type_of_review in ['positive', 'negative']:
        for quarter in list_of_quarters:
            combination_of_brands += list(itertools.product([str(quarter)], dict_of_clean_doc_by_quarter_by_brand[type_of_review][str(quarter)].keys(), [type_of_review]))

    from multiprocessing import Pool, cpu_count, Manager
    print("{} products found... ".format(str(len(combination_of_brands))))
    list_of_arguments = [(dict_of_clean_doc_by_quarter_by_brand, str(quarter_brand[0]), quarter_brand[1], quarter_brand[2], number_of_topics_range) for quarter_brand in combination_of_brands]

    output_df = Manager().list()

    with Pool(processes= cpu_count() * 2) as pool:
        review_df = pool.starmap(build_single_LDA_model, list_of_arguments)

    pool.terminate()
    pool.join()
    
    output_df = pd.concat(review_df, ignore_index = True)    
    
    writer = pd.ExcelWriter('Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand.xlsx')
    output_df.to_excel(writer,'Topic Model by Quarter by Brand')
    writer.save()
    writer.close()
    return