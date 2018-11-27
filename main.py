#Standard library imports
import os
import pickle

#Third party library imports
import pandas as pd

#Python File Imports
## For Webscraping
from Webscraping.Scrapers.amazon import amazon_scrape_to_df
from Webscraping.Scrapers.walmart import walmart_scrape_to_df
from Webscraping.Scrapers.bestbuy import bestbuy_scrape_to_df
## For Topic Modelling
from Topic_Modelling.topic_modelling.LDA_GridSearch import LDA_topic_modeller_by_quarter_by_brand_multiprocessing
from Topic_Modelling.topic_modelling.HDP import HDP_topic_modeller_by_quarter_by_brand

## For Finding Context Similarity
from Finding_Context_Similarity.context_similarity import construct_similarity_table

##########################################################################################
#----------------------------------- OPERATIONS CONFIG ----------------------------------- 

# Set the following flags to true or false to run the following operations

WEBSCRAPER_FLAG = {"AMAZON" : True,
              "WALMART" : True,
              "BESTBUY" : True}

TOPIC_MODELLING_FLAG = {"LDA_W_GRIDSEARCH" : True,
                   "HDP" : True}

CONTEXTUAL_SIMILARITY_FLAG = {"CONTEXTUAL_SIMILARITY_W_FASTTEXT" : True}

#----------------------------------- WEB SCRAPER CONFIG ----------------------------------- 

# SEARCH_TERM (str): The term that you want to search for, same term will be used to create the topic model and contextual similarity
# e.g. SEARCH_TERM = "coffee machine"
SEARCH_TERM = "coffee machine"

#----------------------------------- TOPIC MODELLING CONFIG -----------------------------------

# PATH_TO_REVIEW_DOC (str): File path of the scraped excel file, advised not to change file path 
# e.g. PATH_TO_REVIEW_DOC = "Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)
PATH_TO_REVIEW_DOC = "Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)

# LIST_OF_WORDS_TO_EXCLUDE (list): Additional list of words to exclude from the topic model 
# e.g. LIST_OF_WORDS_TO_EXCLUDE = ['one', 'two', 'three', 'four', 'five', 'star']
LIST_OF_WORDS_TO_EXCLUDE = ['one', 'two', 'three', 'four', 'five', 'star']

# LIST_OF_COMMON_WORDS (list): Additional list of words to exclude, synonyms of words will also be excluded from model
# e.g. LIST_OF_COMMON_WORDS = ["good", "great", "love"]
LIST_OF_COMMON_WORDS = ["good", "great", "love"]

# NUMBER_OF_TOPICS_RANGE (list): List of integers that the GridSearch Algorithm will search over, applicable only for LDA, not HDP
# e.g. NUMBER_OF_TOPICS_RANGE = [2,3,4,5]
NUMBER_OF_TOPICS_RANGE = [2,3,4,5]

# LIST_OF_YEARS_TO_INCLUDE (list): List of years (str) to include in the topic model. Years not in list will not be considered.
# e.g. LIST_OF_YEARS_TO_INCLUDE = ['2016', '2017', '2018']
LIST_OF_YEARS_TO_INCLUDE = ['2016', '2017', '2018']

#----------------------------------- CONTEXT SIMILARITY CONFIG -----------------------------------

# HYPOTHESIS_STATEMENT (str): Hypothesis provided will be used to compare with each user review to provide a similarity score
# e.g. HYPOTHESIS_STATEMENT = 'refund returns'
HYPOTHESIS_STATEMENT = 'refund returns'

# THRESHOLD (float): Overall Similarity Score is calculated by setting a THRESHOLD (between 0 to 1). If similarity to hypothesis exceeds the THRESHOLD, it is counted as one "occurence". Overall Similarity Score is the proportion of occurences with respect to total number of reviews by brand by quarter.
# e.g. THRESHOLD = 0.6 
THRESHOLD = 0.6

# POLARITY_BIAS (float): In order to get a more accurate prediction of the similarity score, we help the algorithm by giving some context of whether the HYPOTHESIS_STATEMENT is a positive statement or a negative statement. For a positive hypothesis, POLARITY BIAS will be a positive number and negative for a negative hypothesis. The recommended value of bias should be between -0.2 and 0.2. The POLARITY BIAS will then be deducted or added to the similarity score.
# e.g. POLARITY_BIAS = -0.2
POLARITY_BIAS = -0.2

# REPROCESS (bool): Because the dataset will be preprocessed before we fit it into the model, we save the processed data in a pickle file, with the location and name: 'pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), such that if we were to run the code multiple times, we will not have to keep reprocessing the data. If REPROCESS is set to True, even if there is an existing pickle file, we will re-run the preprocessing pipeline. This is usually set to True in the event that the preprocessing pipeline is edited.
# e.g. REPROCESS = False
REPROCESS = False

##########################################################################################
#----------------------------------- Webscraping ----------------------------------- 

# Triggers the amazon, bestbuy and walmart webscrapers
def scrape_to_corpus(SEARCH_TERM, WEBSCRAPER_FLAG):
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    if WEBSCRAPER_FLAG['AMAZON'] == True:
        amazon_df = amazon_scrape_to_df(SEARCH_TERM)
    if WEBSCRAPER_FLAG['WALMART'] == True:
        walmart_df = walmart_scrape_to_df(SEARCH_TERM)
    if WEBSCRAPER_FLAG['BESTBUY'] == True:
        bestbuy_df = bestbuy_scrape_to_df(SEARCH_TERM)

    frames = [amazon_df, walmart_df, bestbuy_df]
    result = pd.concat(frames)

    writer = pd.ExcelWriter(PATH_TO_REVIEW_DOC)
    result.to_excel(writer,'{}'.format(SEARCH_TERM))
    writer.save()
    writer.close()
    return

# In the event that pickle files were generated from the individual webscrapers, 
# and you just want to combine the various pickle files to 1 document.
def using_cached_data(SEARCH_TERM):
    with open(r'pickle_files\amazon_web_scrape.pickle', 'rb') as handle_1:
        amazon_df = pickle.load(handle_1)
    with open(r'pickle_files\bestbuy_web_scrape.pickle', 'rb') as handle_2:
        bestbuy_df = pickle.load(handle_2)
    with open(r'pickle_files\walmart_web_scrape.pickle', 'rb') as handle_3:
        walmart_df = pickle.load(handle_3)

    frames = [amazon_df, walmart_df, bestbuy_df]
    result = pd.concat(frames)
    writer = pd.ExcelWriter('Webscraping/Review Corpus/Customer Reviews of {}.xlsx'.format(SEARCH_TERM))
    result.to_excel(writer,'{}'.format(SEARCH_TERM))
    writer.save()
    writer.close()
    return

#----------------------------------- Topic Modelling ----------------------------------- 
# Merge both lists to generate complete set of words to exclude.
LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_WORDS_TO_EXCLUDE + SEARCH_TERM.split()

#Convert to pandas DataFrame
DF = pd.read_excel(PATH_TO_REVIEW_DOC)

def construct_topic_modelling(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE, TOPIC_MODELLING_FLAG):
    if TOPIC_MODELLING_FLAG['LDA_W_GRIDSEARCH'] == True:
        LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, NUMBER_OF_TOPICS_RANGE, SEARCH_TERM)
    if TOPIC_MODELLING_FLAG['HDP'] == True:
        HDP_topic_modeller_by_quarter_by_brand(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, SEARCH_TERM)
    return

#----------------------------------- Contextual Similarity ----------------------------------- 
def building_similarity_table(HYPOTHESIS_STATEMENT, CONTEXTUAL_SIMILARITY_FLAG):
    if CONTEXTUAL_SIMILARITY_FLAG['CONTEXTUAL_SIMILARITY_W_FASTTEXT'] == True:
        if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
            construct_similarity_table(HYPOTHESIS_STATEMENT,LIST_OF_YEARS_TO_INCLUDE, THRESHOLD, POLARITY_BIAS, SEARCH_TERM, REPROCESS)
        else:
            print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")
    return

if __name__ == '__main__':
    scrape_to_corpus(SEARCH_TERM, WEBSCRAPER_FLAG)
    construct_topic_modelling(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE, TOPIC_MODELLING_FLAG)
    building_similarity_table(HYPOTHESIS_STATEMENT, CONTEXTUAL_SIMILARITY_FLAG)