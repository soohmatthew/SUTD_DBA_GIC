#Standard library imports
import re
import string
import pickle
import os
import itertools
import sys

#Third party library imports
import pandas as pd
import spacy
import textacy
from nltk.stem import WordNetLemmatizer
import numpy as np
from gensim.models.fasttext import FastText
import swifter
from scipy import spatial
import openpyxl

"""
The main function of context_similarity.py is "construct_similarity_table".
Please make sure that:

(1) Raw data file is saved in the location as "Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)

(2) Pre-trained english FastText Word Vector (bin + text) is saved under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'. 

*Pre-trained vectors can be dowloaded at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md
"""

# We load the spacy's "en_core_web_sm" model to aid us with text preprocessing, as well as noun chunk extraction.
# type(nlp) = <class 'spacy.lang.en.English'> 
nlp = spacy.load('en_core_web_sm')

# For debugging purposes, we want to add the file path ".../SUTD_DBA_GIC", so that individual functions can be run. If file locations cannot be found, please check if your current working directory is ".../SUTD_DBA_GIC".
sys.path.append(os.getcwd())

def clean_up_corpus(sentence):
    """
    Function:
    ---------
        (1) Apply preprocessing to a given sentence, by:

            (a) Changing text to lower case.

            (b) Remove non alphabetical characters

            (c) Lemmatization of words (using spacy's lemmatization library)

            (d) Removal of stopwords

    Args:
    -----
        (1) sentence (str): A given sentence of the user review.

    Returns:
    --------
        (1) words_without_stopwords (str): Single string of processed words 

    References:
    -----------
        [1] API Reference:
        https://spacy.io/usage/linguistic-features#pos-tagging
    """

    # List of stop words to remove are predefined here. We do not use the conventional NLTK library to remove stop words, as those libraries include word like "not", negation that plays an important role in representing the phrase. Only most common words are removed, but the list can and should be expanded, based on the context of the text. 
    stop_words = ["is", "the", "are", "a"]
    # 1. Convert to lower case
    text = " ".join([text.lower() for text in sentence.split()])
    # 2. Remove non alphabetic characters
    words = ' '.join([word for word in text.split() if not word.isdigit()])
    # 3. Lemmatization
    words = " ".join([word.lemma_ for word in nlp(words) if word.lemma_ not in ['-PRON-']])
    # 4. Remove stopwords
    words_without_stopwords = ' '.join([word for word in words.split() if word not in stop_words])
    return words_without_stopwords

def noun_phrase_extractor(sentence):
    """
    Function:
    ---------
        (1) Function extracts all possible noun phrases from a given sentence, using spacy's dependency parser.

    Args:
    -----
        (1) sentence (str): String belonging to either a user review or a hypothesis

    Returns:
    --------
        (1) list_of_noun_phrase (list): List of noun phrases extracted. Each element may be a word or a combination of words. 

    References:
    -----------
        [1] API Reference:
        https://spacy.io/usage/linguistic-features#noun-chunks
    """
    # Pass in sentence into spacy's Language object, type(spacy_doc) = <class 'spacy.tokens.doc.Doc'>
    spacy_doc = nlp(sentence)
    # Extract text of noun chunks
    list_of_noun_phrase = [np.text for np in spacy_doc.noun_chunks]
    return list_of_noun_phrase

def verb_phrase_extractor(sentence):
    """
    Function:
    ---------
        (1) Function extracts all possible verb phrases from a given sentence, using textacy, which is an NLP tool built on spacy. Spacy was not used as it did not have the functions available to extract phrases. 

        (2) Instead, this function uses regular expression to extract the expected form of the verb phrase.

    Args:
    -----
        (1) sentence (str): String belonging to either a user review or a hypothesis

    Returns:
    --------
        (1) list_of_verb_phrase (list): List of verb phrases extracted. Each element may be a word or a combination of words. 

    References:
    -----------
        [1] API Reference:
        https://github.com/chartbeat-labs/textacy

        [2] Glossary of spacy's POS tags
        https://github.com/explosion/spaCy/blob/a793174ae92f0802970cf19821e24a1004af28d0/spacy/glossary.py#L10

        [3] Understanding Universal POS tags
        http://universaldependencies.org/u/pos/all.html
    """

    # Pass in the sentence to create a 'textacy.doc.Doc'
    doc = textacy.Doc(sentence, lang='en_core_web_sm')

    # Regular expression to extract out phrases
    verb_phrase = r'<VERB>?<ADV | DET>*<VERB>+'

    # Extract text of verb phrases matching the regular expression
    list_of_verb_phrase = [list.text for list in textacy.extract.pos_regex_matches(doc, verb_phrase)]
    return list_of_verb_phrase

def prepositional_phrase_extractor(sentence):
    """
    Function:
    ---------
        (1) Function extracts all possible prepositional phrases from a given sentence, using textacy, which is an NLP tool built on spacy. Spacy was not used as it did not have the functions available to extract phrases. 

        (2) Instead, this function uses regular expression to extract the expected form of the prepositional phrase.

    Args:
    -----
        (1) sentence (str): String belonging to either a user review or a hypothesis

    Returns:
    --------
        (1) list_of_prepositional_phrase (list): List of prepositional phrases extracted. Each element may be a word or a combination of words. 

    References:
    -----------
        [1] API Reference:
        https://github.com/chartbeat-labs/textacy

        [2] Glossary of spacy's POS tags
        https://github.com/explosion/spaCy/blob/a793174ae92f0802970cf19821e24a1004af28d0/spacy/glossary.py#L10

        [3] Understanding Universal POS tags
        http://universaldependencies.org/u/pos/all.html
    """

    # Pass in the sentence to create a 'textacy.doc.Doc'
    doc = textacy.Doc(sentence, lang='en_core_web_sm')

    # Regular expression to extract out phrases
    prepositional_phrase = r'<PREP>? <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    # Extract text of verb phrases matching the regular expression
    list_of_prepositional_phrase = [list.text for list in textacy.extract.pos_regex_matches(doc, prepositional_phrase)]
    return list_of_prepositional_phrase

def break_down_hypothesis(hypothesis):
    """
    Function:
    ---------
        (1) The function will try and figure out if the hypothesis constitutes a noun, verb or prepositional phrase. 
        (2) It calls the 3 functions 'noun_phrase_extractor', 'verb_phrase_extractor' and 'prepositional_phrase_extractor'. 
        (3) If multiple phrases detected within the hypothesis, the function will return the type of phrase with the most number of phrases.

    Assumptions:
    ------------
        (1) If the list of phrases returned by the phrase_extractors has the most elements, it will represent the hypothesis the best.

    Args:
    -----
        (1) hypothesis (str): Hypothesis given by the user. Suggested hypothesis is a short and concise phrase.

    Returns:
    --------
        (1) tuple of (type_of_phrase, list_of_phrases):
            (a) type_of_phrase (str): "Noun Phrases"/"Verb Phrases"/"Prepositional Phrases"
            (b) list_of_phrases (list): a list of all the possible phrases extracted from the hypothesis, based on the type_of_phrase
    """

    # Calls the 3 different functions 'noun_phrase_extractor', 'verb_phrase_extractor' and 'prepositional_phrase_extractor', and collates the results into a dictionary.
    dict_of_hypothesis_phrases = {"Noun Phrases" : noun_phrase_extractor(hypothesis), 
                                  "Verb Phrases" : verb_phrase_extractor(hypothesis),
                                  "Prepositional Phrases" : prepositional_phrase_extractor(hypothesis)}

    # If there is 1 or more phrases extracted from the hypothesis, the type of phrase, and the list of phrases is compiled into output.
    output = [(type_of_phrase, list_of_phrases) for type_of_phrase, list_of_phrases in dict_of_hypothesis_phrases.items() if len(list_of_phrases) > 0]

    # If there are no phrases extracted from hypothesis
    if len(output) == 0:
        print("Unable to pick out any noun phrase, verb phrase or prepositional phrase, try a different hypothesis!")
        return None, None
    
    # If there is more than 1 type of phrase extracted from hypothesis, e.g. There is a presence of both Noun Phrases and Verb Phrases
    elif len(output) > 1:
        # Returns the (type_of_phrase, list_of_phrases) with the list_of_phrases with the greatest length.
        length = ("length_check", [])
        for pair in output:
            if len(pair[1]) > len(length[1]):
                length = pair
        return length
    
    # If there is 1 type of phrase extracted from hypothesis, return it
    else:
        return output[0]
    
def clean_and_load_data(LIST_OF_YEARS_TO_INCLUDE, SEARCH_TERM, REPROCESS = False):
    """
    Function:
    ---------
        (1) Function executes data munging on "df", which is the excel file ".../Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM) converted into a pandas DataFrame. It uses the following functions:
            
            (a) "clean_up_corpus"

            (b) "noun_phrase_extractor"

            (c) "verb_phrase_extractor"

            (d) "prepositional_phrase_extractor"

        (2) 5 extra columns will be added to "df", namely "Y-Quarter", "User Comment Cleaned", "User Comment (Noun Phrases)", "User Comment (Verb Phrases)" and "User Comment (Prepositional Phrases)".
        
        (2) Initializes FastText objects, from pickle file or by loading it into gensim's function.
        
        (3) Prepares the "df" (pandas DataFrame) and "en_model" (FastText object) to be used for the rest of the functions.

    Assumptions:
    ------------
        Webscraping has already been prepared, and excel file is stored in the file path ".../Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)
    
    Args:
    -----
        (1) LIST_OF_YEARS_TO_INCLUDE (list): LIST_OF_YEARS_TO_INCLUDE is a list of years (str), which will filter the dataset based on the years listed in LIST_OF_YEARS_TO_INCLUDE.

        (2) SEARCH_TERM (str): Term searched by the user.
        
        (3) REPROCESS (bool): Because the dataset will be preprocessed before we fit it into the model, we save the processed data in a pickle file, with the location and name: 'pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), such that if we were to run the code multiple times, we will not have to keep reprocessing the data. If REPROCESS is set to True, even if there is an existing pickle file, we will re-run the preprocessing pipeline. This is usually set to True in the event that the preprocessing pipeline is edited.

    Returns:
    --------
        (1) df (pandas DataFrame): Cleaned up pandas DataFrame with 5 additional columns "Y-Quarter", "User Comment Cleaned", "User Comment (Noun Phrases)", "User Comment (Verb Phrases)" and "User Comment (Prepositional Phrases)"

        (2) en_model (gensim.models.fasttext.FastText): Loaded weight matrix from Facebook’s native fasttext .bin and .vec output files.

    * Returns:
    ----------
        (1) None, None is returned when:
            (a) Raw data file (".../Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)) not found, or not located in the correct folder.

            (b) FastText pre-trained vectors (".../Finding_Context_Similarity/wiki.en/wiki.en.bin") not found, or not located in the correct folder.

    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.load_fasttext_format
        https://github.com/jmcarpenter2/swifter
    """

    # Checks if raw data is available to be processed.
    if os.path.isfile("Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)):
        LIST_OF_YEARS_TO_INCLUDE = [int(year) for year in LIST_OF_YEARS_TO_INCLUDE]
        # Tries to load and process raw data
        if REPROCESS == True:
            print("Reprocessing text data...")
            df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM))
            # Add in an extra column based on which quarter and year the date of the user review falls under
            df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
            df['Y-Quarter'] = df['Date'].dt.to_period("Q")
            df = df[df['Y-Quarter'].dt.year.isin(LIST_OF_YEARS_TO_INCLUDE)]

            # Only include the user reviews that have more than 10 user reviews for a particular brand and quarter
            frames = []
            for brand in df['Brand'].unique():
                for quarter in df['Y-Quarter'].unique():
                    if df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)].shape[0] >= 10:
                        frames.append(df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)])
            df = pd.concat(frames, ignore_index = True)

            # Apply preprocessing function "clean_up_corpus"
            df['User Comment Cleaned'] = df['User Comment'].swifter.apply(clean_up_corpus)

            # Create another column, whereby each value is a list of the phrases extracted
            df['User Comment (Noun Phrases)'] = df['User Comment Cleaned'].swifter.apply(noun_phrase_extractor)
            df['User Comment (Verb Phrases)'] = df['User Comment Cleaned'].swifter.apply(verb_phrase_extractor)
            df['User Comment (Prepositional Phrases)'] = df['User Comment Cleaned'].swifter.apply(prepositional_phrase_extractor)
            print("Done processing text data ...")

            # Write the DataFrame into a pickle file for faster loading later on 
            with open('pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # If User does not require reprocessing of data, function tries to load a cached version of the processed data. If it cannot find a cached version, it will process the data again.
        else:
            # Load Cleaned up file
            if os.path.isfile('pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM)):
                with open('pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), 'rb') as handle:
                    print("Loading Processed Text from cache...")
                    df = pickle.load(handle)
            else:
                print("Cache not found, processing text data...")
                df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM))
                df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
                df['Y-Quarter'] = df['Date'].dt.to_period("Q")
                df = df[df['Y-Quarter'].dt.year.isin(LIST_OF_YEARS_TO_INCLUDE)]
                frames = []
                for brand in df['Brand'].unique():
                    for quarter in df['Y-Quarter'].unique():
                        if df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)].shape[0] >= 10:
                            frames.append(df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)])
                df = pd.concat(frames, ignore_index = True)
                df['User Comment Cleaned'] = df['User Comment'].swifter.apply(clean_up_corpus)
                df['User Comment (Noun Phrases)'] = df['User Comment Cleaned'].swifter.apply(noun_phrase_extractor)
                df['User Comment (Verb Phrases)'] = df['User Comment Cleaned'].swifter.apply(verb_phrase_extractor)
                df['User Comment (Prepositional Phrases)'] = df['User Comment Cleaned'].swifter.apply(prepositional_phrase_extractor)
                print("Done processing text data ...")
                with open('pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), 'wb') as handle:
                    pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Tries to load Fast text pre-trained vectors from pickle file
        if os.path.isfile('pickle_files/{}.pickle'.format('fast_text_loaded')):
            print("Loading Fast Text from cache...")
            with open('pickle_files/{}.pickle'.format('fast_text_loaded'), 'rb') as handle2:
                en_model = pickle.load(handle2)
        else:
        # If no pickle file is found, it will search "Finding_Context_Similarity/wiki.en/wiki.en.bin" for the downloaded word vectors binary file
            if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
                print("Converting Fast Text into FastText Object...")
                en_model = FastText.load_fasttext_format("Finding_Context_Similarity/wiki.en/wiki.en.bin")
                with open('pickle_files/{}.pickle'.format('fast_text_loaded'), 'wb') as handle2:
                    pickle.dump(en_model, handle2, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")
                return None, None
        return df, en_model
    else:
        print("The file 'Webscraping/Review Corpus/Customer Reviews of {}.xlsx' is not found!".format(SEARCH_TERM))
        return None, None

def find_max_similarity_score(list_of_phrases, list_of_hypothesis, en_model):
    """
    Function:
    ---------
        (1) Creates a list of tuples of all the possible combinations of phrases from the reviews and hypothesis, based on the input "list_of_phrases" and "list_of_hypothesis".

        (2) Individual similarity scores are then calculated can stored in a list ("scores").

        (2) We return the maximum score of the list. In some cases, the maximum score is 0. Function can be adjusted to output the median score instead. Change the last line accordingly.

    Assumptions:
    ------------
        (1) The maximum score is the best representation of the similarity between user review and hypothesis.

    Args:
    -----
        (1) list_of_phrases (list): List of phrases belonging to a user review, based on what is extracted.

        (2) list_of_hypothesis (list): List of phrases belonging to the hypothesis, based on what is extracted.

        (3) en_model (gensim.models.fasttext.FastText): Loaded weight matrix from Facebook’s native fasttext .bin and .vec output files.

    Returns:
    --------
        (1) maximum_score (float): A number between 0 and 1 that represents the maximum similarity score among all phrases.

    References:
    -----------
        [1] API Reference:
        https://docs.python.org/3.6/library/itertools.html#itertools.product
    """

    # Generate all possible combinations of the hypothesis and user review phrases
    # combination_of_phrases will be in the form [(user_phrase1, hypothesis_phrase1), ...]
    combination_of_phrases = list(itertools.product(list_of_phrases, list_of_hypothesis))

    # Generate a list of similarity scores by calling the function "compute_cosine_similarity"
    scores = [compute_cosine_similarity(phrases[0], phrases[1], en_model) for phrases in combination_of_phrases]     

    # Get the maximum score within the list
    maximum_score = max(scores, default = 0)
    return maximum_score

def avg_phrase_vector(phrase, en_model):
    """
    Function:
    ---------
        (1) Splits the phrase into individual words.

        (2) Try to find the corresponding vector of the word from the loaded FastText object for all words in the phrase.

        (2) Calculates the average vector of the phrase.

    Assumptions:
    ------------
        (1) An average of the word vectors will give a sufficient vector representation for the phrase.

    Args:
    -----
        (1) phrase (str): Phrase belonging to user review or hypothesis

        (2) en_model (gensim.models.fasttext.FastText): Loaded weight matrix from Facebook’s native fasttext .bin and .vec output files.

    Returns:
    --------
        (1) featureVec (numpy.ndarray) of the average word vectors.

    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.load_fasttext_format
    """

    # Initialize featureVec as a vector of zeros, with the shape (300,)
    featureVec = np.zeros((300,), dtype="float32")

    # Split phrase into individual words.
    words = phrase.split()

    # We only consider words that are in the "en_model"
    length_of_words = len([word for word in words if word in en_model])
    
    # Calculate average word vector
    for word in words:
        if word in en_model:
            featureVec = np.add(featureVec, en_model.wv[word])
    featureVec /= length_of_words
    return featureVec

def compute_cosine_similarity(USER_REVIEW, HYPOTHESIS, en_model):
    """
    Function:
    ---------
        (1) Using FastText pre-trained vectors ("en_model"), the "en_model" is passed into the function "avg_phrase_vector" to generate the average vector representing the phrases for USER_REVIEW and HYPOTHESIS. See "avg_phrase_vector" for more details.

        (2) Once the average word vectors (USER_REVIEW_VEC and HYPOTHESIS_VEC) are generated, we calculate the cosine similarity of the word vectors of USER_REVIEW and HYPOTHESIS, using spatial.distance.cosine found in the scipy library.

    Args:
    -----
        (1) USER_REVIEW (str): Phrase belonging to user review

        (2) HYPOTHESIS (str): Phrase belonging to hypothesis

        (3) en_model (gensim.models.fasttext.FastText): Loaded weight matrix from Facebook’s native fasttext .bin and .vec output files.

    Returns:
    --------
        (1) Similarity score (float), which is a value between 0 and 1, representing the cosine similarity between USER_REVIEW and HYPOTHESIS.

    References:
    -----------
        [1] API Reference:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html  
    """
    USER_REVIEW_VEC = avg_phrase_vector(USER_REVIEW, en_model)
    HYPOTHESIS_VEC = avg_phrase_vector(HYPOTHESIS, en_model)
    return 1 - spatial.distance.cosine(HYPOTHESIS_VEC, USER_REVIEW_VEC)

def apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, SEARCH_TERM, REPROCESS = False):
    """
    Function:
    ---------
        (1) Function first loads a preprocessed pandas DataFrame, or calls "clean_and_load_data" to preprocess the dataset.

        (2) Function breaks down the hypothesis into its constituent phrases, and identifies if it is a noun, verb or prepositional phrase.

        (3) Function computes the maximum similarity score and adds in the scores as a new column 'Similarity to Hypothesis'

        (4) Similarity scores are changed based on the Polarity Bias. 

    Args:
    -----
        (1) hypothesis (list or str): Hypothesis given by the user. If a list of hypothesis are given, then the function will iterate through the list and compute one table per hypothesis in the list. 

        (2) LIST_OF_YEARS_TO_INCLUDE (list): LIST_OF_YEARS_TO_INCLUDE is a list of years (str), which will filter the dataset based on the years listed in LIST_OF_YEARS_TO_INCLUDE.

        (3) POLARITY BIAS (float): In order to get a more accurate prediction of the similarity score, we help the algorithm by giving some context of whether the HYPOTHESIS_STATEMENT is a positive statement or a negative statement. For a positive hypothesis, POLARITY BIAS will be a positive number and negative for a negative hypothesis. The recommended value of bias should be between -0.2 and 0.2. The POLARITY BIAS will then be deducted or added to the similarity score.

        (4) SEARCH_TERM (str): Term searched by the user.

        (5) REPROCESS (bool): Because the dataset will be preprocessed before we fit it into the model, we save the processed data in a pickle file, with the location and name: 'pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), such that if we were to run the code multiple times, we will not have to keep reprocessing the data. If REPROCESS is set to True, even if there is an existing pickle file, we will re-run the preprocessing pipeline. This is usually set to True in the event that the preprocessing pipeline is edited.

    Returns:
    --------
        pandas DataFrames loaded from function "clean_and_load_data", with the columns "Hypothesis" (str, contains hypothesis provided by user) and "Similarity to Hypothesis" (float, contains similarity score) added.

    References:
    -----------
        [1] API Reference:
        https://github.com/jmcarpenter2/swifter    
    """

    print("Analyzing hypothesis ...")

    # Identifies whether the hypothesis is most likely a noun, verb or prepositional phrase (type_of_phrase), and a list of the different phrases identified (list_of_hypothesis_phrases). Refer to "break_down_hypothesis" for more details.
    type_of_phrase, list_of_hypothesis_phrases = break_down_hypothesis(hypothesis)
    
    print("Hypothesis evaluated: ")
    print(list_of_hypothesis_phrases)
    print("Loading reviews and fast Text vectors ...")

    # If the hypothesis contains at least 1 noun, verb or prepositional phrase
    if list_of_hypothesis_phrases is not None:

        # Loads the preprocessed dataset (df, pandas DataFrame), as well as the FastText word vector object, that is loaded in gensim's FastText format (en_model). Refer to "clean_and_load_data" for more details.
        df, en_model = clean_and_load_data(LIST_OF_YEARS_TO_INCLUDE, SEARCH_TERM, REPROCESS)

        # If preprocessed dataset is loaded properly
        if df is not None:
            # (1) df contains 3 columns of interest, "User Comment Noun Phrases", "User Comment Verb Phrases" & "User Comment Prepositional Phrases", each row containing a list of phrases extracted from the user comment. 
            # (2) New column, "Similarity to Hypothesis" is created, by applying the "find_max_similarity_score" to one of the 3 columns, based on which type_of_phrase corresponds to the hypothesis. See "find_max_similarity_score" for more details.
            # (3) swifter is a package which efficiently applies any function to a pandas dataframe or series in the fastest available manner.
            df['Similarity to Hypothesis'] = df['User Comment ({})'.format(type_of_phrase)].swifter.apply(find_max_similarity_score, list_of_hypothesis = list_of_hypothesis_phrases, en_model = en_model)
            # Hypothesis is added in as a column
            df['Hypothesis'] = hypothesis

            # Add polarity bias.
            df.loc[df.Rating >= 4, 'Similarity to Hypothesis'] += POLARITY_BIAS
            df.loc[df.Rating <= 2, 'Similarity to Hypothesis'] -= POLARITY_BIAS 

            return df
        else:
            return None

def construct_similarity_table(hypothesis, LIST_OF_YEARS_TO_INCLUDE, THRESHOLD, POLARITY_BIAS, SEARCH_TERM, REPROCESS = False):
    """
    Function:
    ---------
        (1) Constructs the similarity table, based on the parameters given to the function. 

        (2) We take the hypothesis and the user reviews, and calculate the maximum cosine similarity (similarity score) between the hypothesis and the user review. 

        (3) User reviews are split by quarter, then by brand. Results are saved into an excel sheet.

    Assumptions:
    ------------
        (1) The excel file "Customer Reviews of {}.xlsx".format(SEARCH_TERM) is saved in the file location ".../Webscraping/Review Corpus/Customer Reviews of {}.xlsx".format(SEARCH_TERM)

    Args:
    -----
        (1) hypothesis (list or str): Hypothesis given by the user. If a list of hypothesis are given, then the function will iterate through the list and compute one table per hypothesis in the list. 

        (2) LIST_OF_YEARS_TO_INCLUDE (list): LIST_OF_YEARS_TO_INCLUDE is a list of years (str), which will filter the dataset based on the years listed in LIST_OF_YEARS_TO_INCLUDE.

        (3) THRESHOLD (float): Overall Similarity Score is calculated by setting a THRESHOLD (between 0 to 1). If similarity to hypothesis exceeds the THRESHOLD, it is counted as one "occurence". Overall Similarity Score is the proportion of occurences with respect to total number of reviews by brand by quarter.

        (4) POLARITY BIAS (float): In order to get a more accurate prediction of the similarity score, we help the algorithm by giving some context of whether the HYPOTHESIS_STATEMENT is a positive statement or a negative statement. For a positive hypothesis, POLARITY BIAS will be a positive number and negative for a negative hypothesis. The recommended value of bias should be between -0.2 and 0.2. The POLARITY BIAS will then be deducted or added to the similarity score.

        (5) SEARCH_TERM (str): Term searched by the user.

        (6) REPROCESS (bool): Because the dataset will be preprocessed before we fit it into the model, we save the processed data in a pickle file, with the location and name: 'pickle_files/processed_data_for_context_{}.pickle'.format(SEARCH_TERM), such that if we were to run the code multiple times, we will not have to keep reprocessing the data. If REPROCESS is set to True, even if there is an existing pickle file, we will re-run the preprocessing pipeline. This is usually set to True in the event that the preprocessing pipeline is edited.

    Returns:
    --------
        (1) None
        (2) "Similarity Table {}.xlsx".format(hypothesis) will be saved to ".../Finding_Context_Similarity/Similarity Table Results/"
    """

    if isinstance(hypothesis, list):
        for hypothesis in hypothesis:
            print(hypothesis)
            # The function 'apply_cosine_similarity_to_df' is called, which generates a pandas dataframe. See function for more details.
            vectorised_reviews_df = apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, SEARCH_TERM, REPROCESS)

            if vectorised_reviews_df is not None:
                # Data Munging to construct table, data is sorted and grouped by quarter and brand.
                vectorised_reviews_df_sorted = vectorised_reviews_df.sort_values(['Similarity to Hypothesis'], ascending=[False])
                vectorised_reviews_df_final = vectorised_reviews_df_sorted.groupby(['Y-Quarter', 'Brand'])

                # Columns to include in the final excel file. "Company" column will be added in if available.
                cols_to_include_with_coy = ['Y-Quarter', 'Company', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity Score']
                cols_to_include = ['Y-Quarter', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity Score']

                # Calculating the Overall Similarity Score for each quarter and brands of that quarter.
                frames = []
                for key in vectorised_reviews_df_final.groups.keys():
                    df_sorted_by_brand = vectorised_reviews_df_sorted[(vectorised_reviews_df_sorted['Y-Quarter'] == key[0]) & (vectorised_reviews_df_sorted['Brand'] == key[1])]
                    # Overall Similarity Score is calculated by setting a THRESHOLD (between 0 to 1). If similarity to hypothesis exceeds the THRESHOLD, it is counted as one "occurence". Overall Similarity Score is the proportion of occurences with respect to total number of reviews by brand by quarter.
                    df_sorted_by_brand['Overall Similarity Score'] = ((df_sorted_by_brand['Similarity to Hypothesis'] > THRESHOLD) * 1).mean()
                    # Subset DF to only the columns in "cols_to_include_with_coy" or "cols_to_include"
                    try:
                        df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include_with_coy]
                    except IndexError:
                        df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include]
                    frames.append(df_sorted_by_brand_selected_cols.head(5))

                output_df = pd.concat(frames, ignore_index = True)

                # Write to excel
                # File name
                writer = pd.ExcelWriter("Finding_Context_Similarity/Similarity Table Results/Similarity Table {}.xlsx".format(hypothesis))
                # Sheet name
                output_df.to_excel(writer,'Ranked by Quarter by Brand')
                writer.save()
                writer.close()

    elif isinstance(hypothesis, str):
        # Same process as above, just that it is only applied for 1 hypothesis

        # The function 'apply_cosine_similarity_to_df' is called, which generates a pandas dataframe. See function for more details.
        vectorised_reviews_df = apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, SEARCH_TERM, REPROCESS)

        if vectorised_reviews_df is not None:
            # Data Munging to construct table, data is sorted and grouped by quarter and brand.
            vectorised_reviews_df_sorted = vectorised_reviews_df.sort_values(['Similarity to Hypothesis'], ascending=[False])
            vectorised_reviews_df_final = vectorised_reviews_df_sorted.groupby(['Y-Quarter', 'Brand'])

            # Columns to include in the final excel file. "Company" column will be added in if available.
            cols_to_include_with_coy = ['Y-Quarter', 'Company', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity Score']
            cols_to_include = ['Y-Quarter', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity Score']

            # Calculating the Overall Similarity Score for each quarter and brands of that quarter.
            frames = []
            for key in vectorised_reviews_df_final.groups.keys():
                df_sorted_by_brand = vectorised_reviews_df_sorted[(vectorised_reviews_df_sorted['Y-Quarter'] == key[0]) & (vectorised_reviews_df_sorted['Brand'] == key[1])]
                # Overall Similarity Score is calculated by setting a THRESHOLD (between 0 to 1). If similarity to hypothesis exceeds the THRESHOLD, it is counted as one "occurence". Overall Similarity Score is the proportion of occurences with respect to total number of reviews by brand by quarter.
                df_sorted_by_brand['Overall Similarity Score'] = ((df_sorted_by_brand['Similarity to Hypothesis'] > THRESHOLD) * 1).mean()
                # Subset DF to only the columns in "cols_to_include_with_coy" or "cols_to_include"
                try:
                    df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include_with_coy]
                except IndexError:
                    df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include]
                frames.append(df_sorted_by_brand_selected_cols.head(5))

            output_df = pd.concat(frames, ignore_index = True)

            # Write to excel
            # File name
            writer = pd.ExcelWriter("Finding_Context_Similarity/Similarity Table Results/Similarity Table {}.xlsx".format(hypothesis))
            # Sheet name
            output_df.to_excel(writer,'Ranked by Quarter by Brand')
            writer.save()
            writer.close()
        else:
            return
    else:
        # If the hypothesis input is not a list or string
        print("Hypothesis statement should be a list or a string.")