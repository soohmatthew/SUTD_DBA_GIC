#Standard library imports
import os

#Third party library imports
import pandas as pd

#Python File Imports
from topic_modelling.LDA_GridSearch import LDA_topic_modeller_by_quarter_by_brand_multiprocessing, LDA_topic_modeller_by_quarter_multiprocessing

#Define the parameters of the model
DF = pd.read_excel("Review Corpus/Customer Reviews of coffee machine.xlsx")
LIST_OF_ADDITIONAL_STOP_WORDS = ['coffee', 'machine', 'one', 'two', 'three', 'four', 'five', 'star']
LIST_OF_COMMON_WORDS = ["good", "great", "love"]
number_of_topics_range = [2,3,4,5]


def main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range):
    LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range)
    LDA_topic_modeller_by_quarter_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range)

if __name__ == '__main__':
    main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range)