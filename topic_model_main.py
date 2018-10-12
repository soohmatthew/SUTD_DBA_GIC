import os

from topic_modelling.LDA import *
from topic_modelling.LSA import *
from topic_modelling.K_means import *

#Define the parameters of the model
current_dir = os.getcwd()


DF = pd.read_excel(r"{}\output corpus\amazon output.xlsx".format(current_dir))
NUMBER_OF_TOPICS = 3
NUMBER_OF_WORDS = 5
LIST_OF_ADDITIONAL_STOP_WORDS = ["coffee", "machine"]

def main(DF, NUMBER_OF_TOPICS, NUMBER_OF_WORDS):
    LDA_topic_modeller(DF, num_topics = NUMBER_OF_TOPICS, num_words = NUMBER_OF_WORDS, LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS)
    LSA_topic_modeller(DF, num_topics = NUMBER_OF_TOPICS, num_words = NUMBER_OF_WORDS, LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS)
    K_means_topic_modeller(DF, num_topics = NUMBER_OF_TOPICS, num_words = NUMBER_OF_WORDS, LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS)

if __name__ == '__main__':
    main(DF, NUMBER_OF_TOPICS, NUMBER_OF_WORDS)