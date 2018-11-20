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


#----------------------------------- OPERATIONS CONFIG ----------------------------------- 

WEBSCRAPER_FLAG = {"AMAZON" : True,
              "WALMART" : True,
              "BESTBUY" : True}

TOPIC_MODELLING_FLAG = {"LDA_W_GRIDSEARCH" : True,
                   "HDP" : True}

CONTEXTUAL_SIMILARITY_FLAG = {"DOC2VEC": True,
                              "CONTEXTUAL_SIMILARITY_W_FASTTEXT" : True}

#----------------------------------- WEB SCRAPER CONFIG ----------------------------------- 
SEARCH_TERM = "coffee machine"

#----------------------------------- TOPIC MODELLING CONFIG -----------------------------------
PATH_TO_REVIEW_DOC = "Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)
LIST_OF_WORDS_TO_EXCLUDE = ['one', 'two', 'three', 'four', 'five', 'star']
LIST_OF_COMMON_WORDS = ["good", "great", "love"]
NUMBER_OF_TOPICS_RANGE = [2,3,4,5] #This would apply only for LDA, not HDP
LIST_OF_YEARS_TO_INCLUDE = ['2016', '2017', '2018']

# CONTEXT SIMILARITY TABLE CONFIG
HYPOTHESIS_STATEMENT = 'refund returns'
THRESHOLD = 0.6
POLARITY_BIAS = -0.2
REPROCESS = False
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
        LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, NUMBER_OF_TOPICS_RANGE)
    if TOPIC_MODELLING_FLAG['HDP'] == True:
        HDP_topic_modeller_by_quarter_by_brand(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE)
    return

#----------------------------------- Contextual Similarity ----------------------------------- 
def building_similarity_table(HYPOTHESIS_STATEMENT, CONTEXTUAL_SIMILARITY_FLAG):
    if CONTEXTUAL_SIMILARITY_FLAG['CONTEXTUAL_SIMILARITY_W_FASTTEXT'] == True:
        if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
            construct_similarity_table(HYPOTHESIS_STATEMENT,LIST_OF_YEARS_TO_INCLUDE, THRESHOLD, POLARITY_BIAS, REPROCESS)
        else:
            print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")
    if CONTEXTUAL_SIMILARITY_FLAG['DOC2VEC'] == True:
        pass
    return

if __name__ == '__main__':
    scrape_to_corpus(SEARCH_TERM, WEBSCRAPER_FLAG)
    construct_topic_modelling(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE, TOPIC_MODELLING_FLAG)
    building_similarity_table(HYPOTHESIS_STATEMENT, CONTEXTUAL_SIMILARITY_FLAG)