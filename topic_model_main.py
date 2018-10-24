#Standard library imports
import os

#Python File Imports
from topic_modelling.LDA_1 import *

#Define the parameters of the model
current_dir = os.getcwd()
LIST_OF_ADDITIONAL_STOP_WORDS = ['the', 'of', 'and', 'a', 'to', 'in', 'is', 'you', 'that', 'it', 'he', 'was',
                            'for', 'on', 'are', 'as', 'with', 'his', 'they', 'I', '', 'at', 'be', 'this',
                            'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all',
                            'were', 'we', 'when', 'your', 'can', 'said', '', 'there', 'use', 'an', 'each',
                            'which', 'she', 'do', 'how', 'their', 'if', 'will', 'up', 'other', 'about',
                            'out', 'many', 'then', 'them', 'these', 'so', '', 'some', 'her', 'would',
                            'make', 'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more', 'write',
                            'go', 'see', 'number', 'no', 'way', 'could', 'people', '', 'my', 'than', 
                            'first', 'water', 'been', 'call', 'who', 'oil', 'its', 'now', 'find', 'long',
                            'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part', 'coffee', 'machine']

DF = pd.read_excel("output corpus/Customer Reviews of coffee machine.xlsx")
LIST_OF_COMMON_WORDS = ["good", 'bad', 'love']

def main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    LDA_topic_modeller(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

if __name__ == '__main__':
    main(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

