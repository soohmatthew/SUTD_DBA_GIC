#Standard library imports
import os

#Third party library imports
import pandas as pd

#Python File Imports
from topic_modelling.LDA_GridSearch import LDA_topic_modeller_by_quarter_by_brand_multiprocessing, LDA_topic_modeller_by_quarter_multiprocessing

#Define the parameters of the model

SEARCH_TERM = "coffee machine"
PATH_TO_REVIEW_DOC = "Review Corpus/Customer Reviews of coffee machine.xlsx"
LIST_OF_WORDS_TO_EXCLUDE = ['one', 'two', 'three', 'four', 'five', 'star']
LIST_OF_COMMON_WORDS = ["good", "great", "love"]
NUMBER_OF_TOPICS_RANGE = [2,3,4,5]

# Merge both lists to generate complete set of words to exclude.
LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_WORDS_TO_EXCLUDE + SEARCH_TERM.split()

#Convert to pandas DataFrame
DF = pd.read_excel(PATH_TO_REVIEW_DOC)

def main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE):
    LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE)
    LDA_topic_modeller_by_quarter_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE)

if __name__ == '__main__':
    main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, NUMBER_OF_TOPICS_RANGE)