#Standard Library Imports
import os
import sys

#Python File Imports
sys.path.append(os.getcwd())
from Finding_Context_Similarity.context_similarity import construct_similarity_table

#Define the parameters of the model

HYPOTHESIS_STATEMENT = 'breakdown'
LIST_OF_YEARS_TO_INCLUDE = ['2016', '2017', '2018']
THRESHOLD = 0.6
POLARITY_BIAS = -0.2
SEARCH_TERM = "coffee machine"
REPROCESS = False

if __name__ == '__main__':
    if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
        construct_similarity_table(HYPOTHESIS_STATEMENT, LIST_OF_YEARS_TO_INCLUDE, THRESHOLD, POLARITY_BIAS, SEARCH_TERM, REPROCESS)
    else:
        print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")