#Standard library imports
import pickle
from collections import OrderedDict
import os
import sys

#Third party imports
from gensim.corpora import Dictionary
import openpyxl
from gensim.corpora import Dictionary
from gensim.models import HdpModel, CoherenceModel
import pandas as pd
import numpy as np
import nltk

#Python File Imports
sys.path.append(os.getcwd())

from PreProcessing import Preprocessing

def model_fitting(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE):
    if os.path.isfile('pickle_files/processed_data_by_quarter_by_brand.pickle'): 
        with open('pickle_files/processed_data_by_quarter_by_brand.pickle', 'rb') as handle_2:
            dict_of_clean_doc_by_quarter_by_brand = pickle.load(handle_2)
    else:
        dict_of_clean_doc_by_quarter_by_brand = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    filtered_dict = OrderedDict()

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

    for type_of_review in filtered_dict:
        processed_model_by_brand[type_of_review] = OrderedDict()
        for quarter in filtered_dict[type_of_review]:
            processed_model_by_brand[type_of_review][quarter] = OrderedDict()
            for brand in filtered_dict[type_of_review][quarter]:

                dictionary = Dictionary(filtered_dict[type_of_review][quarter][brand])
                corpus = [dictionary.doc2bow(text) for text in filtered_dict[type_of_review][quarter][brand]]
                hdpmodel = HdpModel(corpus=corpus, id2word=dictionary, alpha = 1)
                topic_list = hdpmodel.show_topics(num_topics= -1, num_words = 10, formatted= False)
                hdp_coherence = CoherenceModel(model = hdpmodel, texts = filtered_dict[type_of_review][quarter][brand],corpus = corpus, coherence='c_v')

                processed_model_by_brand[type_of_review][quarter][brand] = [dictionary, corpus, hdpmodel, topic_list, hdp_coherence]
    return processed_model_by_brand

def return_coherence(hdp_model):
    list_of_coherence = hdp_model.get_coherence_per_topic()
    return list_of_coherence

def HDP_topic_modeller_by_quarter_by_brand(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE):
    processed_model_by_brand = model_fitting(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE)
    
    # Constraints for the model
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

                hdp_coherence = processed_model_by_brand[type_of_review][quarter][brand][4]
                topic_list = processed_model_by_brand[type_of_review][quarter][brand][3]

                # Taking a maximum of 5 topics each, figure out which topic to accept or reject, based on the weight which has to be greater than 0.5
                topics_nos = [x[0] for x in topic_list ]
                weights = [ sum([item[1] for item in topic_list[topicN][1]]) for topicN in topics_nos ]

                top_sorted_weight_index = np.argsort(weights)[-MAX_NUMBER_OF_TOPICS:]
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
                                            'Quarter': quarter,
                                            'Topic': topic,
                                            'Type of Review': type_of_review,
                                             'Coherence Level' : coherence}
                                topic_df = topic_df.append(topic_dict, ignore_index=True)
                

    writer = pd.ExcelWriter('Topic_Modelling/topic_modelling/HDP Topic Model by Quarter by Brand.xlsx')
    topic_df.to_excel(writer,'Topic Model by Quarter by Brand')
    writer.save()
    writer.close()

    return topic_df