#Standard library imports
import pickle
from collections import OrderedDict
import os
import sys

#Third party imports
from gensim.corpora import Dictionary
import openpyxl
from gensim.models import HdpModel, CoherenceModel
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

#Python File Imports
sys.path.append(os.getcwd())
nltk.download('wordnet')
from PreProcessing import Preprocessing

"""
HDP.py contains the functions needed to prepare and genereate HDP models to get the best topics of a document. There are a few assumptions made during the implementation of this, which will be outlined during the individual functions.

The main function of interest is 'HDP_topic_modeller_by_quarter_by_brand', multiprocessing is NOT used, please expect it to take a longer time than LDA_GridSearch.py 
"""

def model_fitting(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE):
    """
    Function:
    ---------
        (1) Prepares the "dict_of_clean_doc_by_quarter_by_brand" by calling PreProcessing.py, or retrieving from cache. "dict_of_clean_doc_by_quarter_by_brand" is a nested Ordered Dictionary with the following three layers:

            (a) Type of Review
            
            (b) Quarter

            (c) Brand

                (i) Each brand will have a list of processed user reviews (str)

        (2) Filter reviews such that only if there are 10 reviews and above for a certain brand and quarter, will it get reviewed.

        (3) Use gensim's HdpModel to generate a HDP model

        (4) Use gensim's CoherenceModel to generate a coherence model to evaluate the coherence of the HDP model

    Args:
    -----
        (1) DF (pandas DataFrame): DataFrame of user reviews

        (2) LIST_OF_ADDITIONAL_STOP_WORDS (list): List of additional stop words (str)

        (3) LIST_OF_COMMON_WORDS (list): List of common words (str), that you would want to not only remove the word from the review, but synonyms of the word as well.

        (4) LIST_OF_YEARS_TO_INCLUDE (list): List of years (str) to include for topic modelling. User comments will be removed if their date of comment is not in the list of years.

    Returns:
    --------
        processed_model_by_brand (OrderedDict): A nested Ordered Dictionary with a similar structure to "dict_of_clean_doc_by_quarter_by_brand", with the following three layers:

            (a) Type of Review
            
            (b) Quarter

            (c) Brand

                (i) Each brand will have a list of consisting of the elements:

                    1. dictionary - gensim.corpora.Dictionary object fitted with user reviews for the specific brand, quarter and type of review
                    
                    2. corpus - Bag-of-Words representation of the user review for the specific brand, quarter and type of review
                    
                    3. hdpmodel - HdpModel fitted with the dictionary and corpus generated for the specific brand, quarter and type of review 
                    
                    4. topic_list - List of all the topics (150) as (weight, word) pairs.
                    
                    5. hdp_coherence - CoherenceModel Object that allows for building and maintaining a model for topic coherence
                   
                    6. raw_text - Text provided by filtered_dict

    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/hdpmodel.html
        https://radimrehurek.com/gensim/models/coherencemodel.html
    """

    # Checks if there is a cached version of "dict_of_clean_doc_by_quarter_by_brand"
    if os.path.isfile('pickle_files/processed_data_by_quarter_by_brand.pickle'): 
        with open('pickle_files/processed_data_by_quarter_by_brand.pickle', 'rb') as handle_2:
            dict_of_clean_doc_by_quarter_by_brand = pickle.load(handle_2)
    else:
        dict_of_clean_doc_by_quarter_by_brand = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    filtered_dict = OrderedDict()

    # Creates a filtered dictionary that contains user reviews if there are more than 10 reviews per brand per quarter
    # User reviews are split into a list of individual words
    for type_of_review in dict_of_clean_doc_by_quarter_by_brand:
        filtered_dict[type_of_review] = OrderedDict()
        for quarter in dict_of_clean_doc_by_quarter_by_brand[type_of_review]:
            if any(year in quarter for year in LIST_OF_YEARS_TO_INCLUDE):
                filtered_dict[type_of_review][quarter] = OrderedDict()
                for brand in dict_of_clean_doc_by_quarter_by_brand[type_of_review][quarter]:
                    #We only choose brands that have more than 10 reviews.
                    if len(brand) < 10:
                        continue
                    else:
                        filtered_dict[type_of_review][quarter][brand] = []
                        for review in dict_of_clean_doc_by_quarter_by_brand[type_of_review][quarter][brand]:
                            filtered_dict[type_of_review][quarter][brand].append(review.split())
            else:
                continue

    processed_model_by_brand = OrderedDict()

    # Iterate through every type of review, quarter and brand found in "filtered_dict"
    for type_of_review in filtered_dict:
        processed_model_by_brand[type_of_review] = OrderedDict()
        for quarter in filtered_dict[type_of_review]:
            processed_model_by_brand[type_of_review][quarter] = OrderedDict()
            for brand in filtered_dict[type_of_review][quarter]:

                # Creates dictionary, corpus objects required for HdpModel
                dictionary = Dictionary(filtered_dict[type_of_review][quarter][brand])
                corpus = [dictionary.doc2bow(text) for text in filtered_dict[type_of_review][quarter][brand]]

                # Fit HdpModel
                hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)

                # Save all topics generated from hdpmodel. 
                # "num_topics = -1" gives us the maximum number of topics generated. 
                # "formatted = false" get us the topics as lists of (weight, word) pairs.
                topic_list = hdpmodel.show_topics(num_topics= -1, num_words = 10, formatted= False)

                # Create CoherenceModel object using the hdpmodel, and corpus generated earlier. 
                # coherence = 'c_v' is the Coherence measure to be used.
                hdp_coherence = CoherenceModel(model = hdpmodel, texts = filtered_dict[type_of_review][quarter][brand],corpus = corpus, coherence='c_v')
                raw_text = filtered_dict[type_of_review][quarter][brand]

                processed_model_by_brand[type_of_review][quarter][brand] = [dictionary, corpus, hdpmodel, topic_list, hdp_coherence, raw_text]
    return processed_model_by_brand


def return_coherence(coherence_model):
    # Returns a list of coherence values for each topic
    list_of_coherence = coherence_model.get_coherence_per_topic()
    return list_of_coherence

def HDP_topic_modeller_by_quarter_by_brand(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, SEARCH_TERM):
    """
    Function:
    ---------
        (1) Generates a nested OrderedDict that contains the fitted HdpModel and CoherenceModel, calling "model_fitting". Nested Dictionary has the following 3 layers:

            (a) Type of Review
            
            (b) Quarter

            (c) Brand

                (i) Each brand will have a list of consisting of the elements:

                    1. dictionary - gensim.corpora.Dictionary object fitted with user reviews for the specific brand, quarter and type of review
                    
                    2. corpus - Bag-of-Words representation of the user review for the specific brand, quarter and type of review
                    
                    3. hdpmodel - HdpModel fitted with the dictionary and corpus generated for the specific brand, quarter and type of review 
                    
                    4. topic_list - List of all the topics (150) as (weight, word) pairs.
                    
                    5. hdp_coherence - CoherenceModel Object that allows for building and maintaining a model for topic coherence
                   
                    6. raw_text - Text provided by filtered_dict

        (2) Filter reviews such that only if there are 10 reviews and above for a certain brand and quarter, will it get reviewed.

        (3) Use gensim's HdpModel to generate a HDP model

        (4) Use gensim's CoherenceModel to generate a coherence model to evaluate the coherence of the HDP model

    Args:
    -----
        (1) DF (pandas DataFrame): DataFrame of user reviews

        (2) LIST_OF_ADDITIONAL_STOP_WORDS (list): List of additional stop words (str)

        (3) LIST_OF_COMMON_WORDS (list): List of common words (str), that you would want to not only remove the word from the review, but synonyms of the word as well.

        (4) LIST_OF_YEARS_TO_INCLUDE (list): List of years (str) to include for topic modelling. User comments will be removed if their date of comment is not in the list of years.

        (5) SEARCH_TERM (str): Term searched by the user.

    Returns:
    --------
        topic_df (pandas DataFrame): A pandas DataFrame containing the following columns:
            (a) Brand

            (b) Keyword

            (c) Keyword TF-IDF Weight

            (d) Quarter

            (e) Topic
            
            (f) Type of Review
            
            (g) Coherence Level
            
    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/hdpmodel.html
        https://radimrehurek.com/gensim/models/coherencemodel.html
    """

    # Fit HdpModel and CoherenceModel
    processed_model_by_brand = model_fitting(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE)
    
    MAX_NUMBER_OF_TOPICS = 5
    THRESHOLD_WEIGHT_OF_TOPIC = 0.25
    
    topic_df = pd.DataFrame()
    count = 0
    total_count = 0
    
    # Get total count
    for type_of_review in processed_model_by_brand:
        for quarter in processed_model_by_brand[type_of_review]:
            for brand in processed_model_by_brand[type_of_review][quarter]:
                total_count += 1
    
    print("{} quarters and brands to process.".format(str(total_count)))
    
    # Process by quarter, by brand
    for type_of_review in processed_model_by_brand:
        for quarter in processed_model_by_brand[type_of_review]:
            for brand in processed_model_by_brand[type_of_review][quarter]:
                print("Processing ... {} {}".format(brand, quarter))

                # Extracts variables from dictionary
                raw_text = processed_model_by_brand[type_of_review][quarter][brand][5]
                hdp_coherence = processed_model_by_brand[type_of_review][quarter][brand][4]
                topic_list = processed_model_by_brand[type_of_review][quarter][brand][3]

                # Generate TF-IDF matrix using sklearn's TfidfVectorizer
                vectorizer = TfidfVectorizer(min_df=1, max_df=.5, stop_words = 'english')
                vectorizer.fit_transform(raw_text)
                idf = vectorizer.idf_
                # Create a dictionary of keywords and their corresponding inverse document frequency (IDF)
                dict_of_words_and_idf = dict(zip(vectorizer.get_feature_names(), idf))

                # Taking a maximum of 5 topics each, figure out which topic to accept or reject, based on the sum of weights which has to be greater than THRESHOLD_WEIGHT_OF_TOPIC
                topics_nos = [x[0] for x in topic_list ]
                # Sums up the weights for every topic
                weights = [ sum([item[1] for item in topic_list[topicN][1]]) for topicN in topics_nos ]

                top_sorted_weight_index = np.argsort(weights)[-MAX_NUMBER_OF_TOPICS:]
                # Get the index of the topics which have a weight greater than the topic weight, and store it to a list
                list_of_shortlisted_index = [index for index in top_sorted_weight_index if weights[index] >= THRESHOLD_WEIGHT_OF_TOPIC]
                number_of_topics = len(list_of_shortlisted_index)

                # Get list of coherence if they are in the number of topics
                list_of_coherence = return_coherence(hdp_coherence)
                count += 1
                print("{} out of {} products being processed ... ".format(str(count), str(total_count)))
                if number_of_topics == 0:
                    continue
                else:
                    for i in range(number_of_topics):
                        # Record all the information, for every topic into a DataFrame
                        topic_index = list_of_shortlisted_index[i]
                        topic = "Topic " + str(i+1)
                        coherence = list_of_coherence[topic_index]
                        topic_list_w_keywords = topic_list[topic_index][1::2]
                        for keywords in topic_list_w_keywords:
                            for keyword in keywords:
                                keyword_text = keyword[0]
                                keyword_weight = keyword[1]
                                topic_dict = {'Brand': brand,
                                            'Keyword': keyword_text,
                                            'Keyword Weight': keyword_weight,
                                            'Keyword TF-IDF Weight': dict_of_words_and_idf[keyword_text],
                                            'Quarter': quarter,
                                            'Topic': topic,
                                            'Type of Review': type_of_review,
                                             'Coherence Level' : coherence}
                                topic_df = topic_df.append(topic_dict, ignore_index=True)

        #To normalise keyword weights, we take the keyword weight divided by the sum of keyword weights
        frames = []
        def normalise(value):
            return value/sum_of_kw_weights
        for type_of_review in topic_df['Type of Review'].unique():
            for brand in topic_df['Brand'].unique():
                for quarter in topic_df['Quarter'].unique():
                    for topic in topic_df['Topic'].unique():
                        specific_topic_df = topic_df[(topic_df['Quarter'] == quarter) & (topic_df['Brand'] == brand) & (topic_df['Type of Review'] == type_of_review) & (topic_df['Topic'] == topic)]
                        sum_of_kw_weights = specific_topic_df['Keyword Weight'].sum()
                        specific_topic_df['Keyword Weight'] = specific_topic_df['Keyword Weight'].apply(normalise)
                        frames.append(specific_topic_df)
        topic_df = pd.concat(frames, ignore_index = True)
    writer = pd.ExcelWriter('Topic_Modelling/Topic Model Results/HDP Topic Model by Quarter by Brand {}.xlsx'.format(SEARCH_TERM))
    topic_df.to_excel(writer,'Topic Model by Quarter by Brand')
    writer.save()
    writer.close()
                
    return topic_df