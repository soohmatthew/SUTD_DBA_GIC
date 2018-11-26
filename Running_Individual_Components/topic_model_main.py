#Standard library imports
import os
import sys

#Third party library imports
import pandas as pd

#Python File Imports
sys.path.append(os.getcwd())
from Topic_Modelling.topic_modelling.LDA_GridSearch import LDA_topic_modeller_by_quarter_by_brand_multiprocessing
from Topic_Modelling.topic_modelling.HDP import HDP_topic_modeller_by_quarter_by_brand

#Define the parameters of the model

SEARCH_TERM = "coffee machine"
PATH_TO_REVIEW_DOC = "Webscraping/Review Corpus/Customer Reviews of coffee machine.xlsx"
LIST_OF_WORDS_TO_EXCLUDE = ['one', 'two', 'three', 'four', 'five', 'star']
LIST_OF_COMMON_WORDS = ["good", "great", "love"]
LIST_OF_YEARS_TO_INCLUDE = ['2016','2017','2018']
NUMBER_OF_TOPICS_RANGE = [2,3,4,5]

# Merge both lists to generate complete set of words to exclude.
LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_WORDS_TO_EXCLUDE + SEARCH_TERM.split()

#Convert to pandas DataFrame
DF = pd.read_excel(PATH_TO_REVIEW_DOC)

def main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE):
    LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, NUMBER_OF_TOPICS_RANGE, SEARCH_TERM)
    HDP_topic_modeller_by_quarter_by_brand(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE)

if __name__ == '__main__':
    main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE)