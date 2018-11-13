#Standard Library Imports
import os
import sys

#Python File Imports
sys.path.append(os.getcwd())
from Finding_Context_Similarity.context_similarity import construct_similarity_table

#Define the parameters of the model

HYPOTHESIS_STATEMENT = 'breakdown'

if __name__ == '__main__':
    if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
        construct_similarity_table(HYPOTHESIS_STATEMENT)
    else:
        print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")