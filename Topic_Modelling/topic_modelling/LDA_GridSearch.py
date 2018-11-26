#Standard library imports
import string
import pickle
from collections import OrderedDict
import os
import itertools
import sys
from multiprocessing import Pool, cpu_count, Manager

#Third party imports
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np
import spacy
import gensim
import openpyxl
from scipy import spatial
from gensim.models.fasttext import FastText

"""
LDA_GridSearch.py contains the functions needed to prepare and genereate LDA models to get the best topics of a document. It is also able to conduct GridSearch on LDA models over a range of topic numbers, to get the best number of topics for each set of documents.

The main function of interest is 'LDA_topic_modeller_by_quarter_by_brand_multiprocessing', which was written with the ability to do multiprocessing.
"""

#Python File Imports
from Topic_Modelling.topic_modelling.PreProcessing import Preprocessing

# For debugging purposes, we want to add the file path ".../SUTD_DBA_GIC", so that individual functions can be run. If file locations cannot be found, please check if your current working directory is ".../SUTD_DBA_GIC".
sys.path.append(os.getcwd())

# Corpus Download necessary to run text processing
import nltk
nltk.download('wordnet')

class LDAUsingPerplexityScorer(LatentDirichletAllocation):
    """
    sklearn's GridSearchCV uses the approximate log-likelihood as score to determine which is the best model for LatentDirichletAllocation.

    In order to implement GridSearch for the LDA, based on the perplexity score as a scoring metric instead, we make a wrapper over sklearn's LDA here and then re-implement the score() function to return perplexity.

    LDAUsingPerplexityScorer inherits all the properties of sklearn's original LatentDirichletAllocation, with only the score modified.

    """
    def score(self, X, y=None):
        score = super().perplexity(X, sub_sampling=False)
        # Perplexity is lower for better, negative scoring to simulate that.
        return -1*score

def build_single_LDA_model(dict_of_clean_doc, quarter, brand, type_of_review, number_of_topics_range):
    """
    Function:
    ---------
        (1) Prepares the "dict_of_clean_doc", 

        (2) Build an LDA model for every type of review, quarter and brand, by calling "build_single_LDA_model". GridSearch is implemented in "build_single_LDA_model".

        (3) Calculate coherence score of the final topic using word embedding, a similar method used in "context_similarity.py"

        (4) Write Results into Excel 
    Args:
    -----
        (1) dict_of_clean_doc (OrderedDict): A nested Ordered Dictionary with the following three layers:

            (a) Type of Review
            
            (b) Quarter

            (c) Brand

        (2) quarter (str): The specific quarter of the user reviews to look at 

        (3) brand (str): The specific brand of the user reviews to look at 

        (4) type_of_review (str): The specific type of reviews of the user reviews to look at.
            
            (a) User Ratings 4 and above will be 'positive'

            (b) User Ratings 2 and below will be 'negative'

        (5) number_of_topics_range (list): List of number of topics (int) from which the GridSearch algorithm will iterate over, to find the best number of topics.

    Returns:
    --------
        topic_model_df (pandas DataFrame): 
            DataFrame with the following columns:
            (a) "Brand"
            (b) "Keyword"
            (c) "Keyword Weight"
            (d) "Quarter"
            (e) "Topic"
            (f) "Topic Frequency"
            (g) "Type of Review"
            (h) "Keyword TF-IDF weight"

    References:
    -----------
        [1] API Reference:
        http://www.nltk.org/howto/wordnet.html

    """
    try:
        print("Building LDA model for ... {}, {} ".format(str(quarter), brand))

        # doc_clean is a list, where each element in an individual user review
        doc_clean = dict_of_clean_doc[type_of_review][str(quarter)][brand]
        
        # Generate TF-IDF matrix using sklearn's TfidfVectorizer
        tfidf_vectorizer = TfidfVectorizer(min_df=1, max_df=.5, stop_words = 'english')
        tfidf_vectorizer.fit_transform(doc_clean)
        idf = tfidf_vectorizer.idf_

        # Create a dictionary of keywords and their corresponding inverse document frequency (IDF)
        dict_of_words_and_idf = dict(zip(tfidf_vectorizer.get_feature_names(), idf))

        # Use CountVectorizer to vectorize the data
        vectorizer = CountVectorizer(analyzer='word',       
                                min_df= 3,                        # minimum required occurence of a word 
                                stop_words='english',             # remove stop words
                                lowercase=True,                   # convert all words to lowercase
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )
        data_vectorized = vectorizer.fit_transform(doc_clean)

        # Create LDAUsingPerplexityScorer class, see "LDAUsingPerplexityScorer" for more details. If you wish to use the approximate log-likelihood as score to determine which is the best model for LatentDirichletAllocation, then you can use "LatentDirichletAllocation" instead of "LDAUsingPerplexityScorer"
        lda_model = LDAUsingPerplexityScorer(max_iter=10,               # Max learning iterations
                                            learning_method='online',   
                                            random_state=100,          # Random state
                                            batch_size=128,            # n docs in each learning iter
                                            evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                            n_jobs = -1,               # Use all available CPUs
                                            )

        # Search parameters to fit into the "GridSearchCV"
        search_params = {'n_components': number_of_topics_range, 'learning_decay': [.5, .7, .9]}
        grid_search_model = GridSearchCV(lda_model, param_grid=search_params)
        grid_search_model.fit(data_vectorized)

        # Get the best LDA model
        best_lda_model = grid_search_model.best_estimator_

        #Creates an Ordered Dictionary, where the key is the nth topic and the corresponding value is a dictionary, with a 'keywords' key, and 
        #list of tuples of the top "n_words" keyword and the associated weight as the value
    
        def show_topics(vectorizer, lda_model, n_words):
            keywords = np.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in lda_model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]

                # Get top "n_words" keywords and the associated weight
                top_keyword_weight = sorted(-topic_weights, key=float)[:n_words]
                top_keyword_list = keywords.take(top_keyword_locs).tolist()
                topic_keywords.append([(top_keyword_list[i], -top_keyword_weight[i]) for i in range(len(top_keyword_weight))])

            topic_keywords_dict = OrderedDict()
            for topic in topic_keywords:
                topic_keywords_dict["topic " + str(topic_keywords.index(topic)+1)] = {'keywords':topic}
            return topic_keywords_dict

        topic_keywords_dict = show_topics(vectorizer=vectorizer, lda_model=best_lda_model, n_words=15)

        #Creates a pandas dataframe to count the frequency of the nth topic being the main topic of a document.
        #Count the number of time topic n is the main topic

        lda_output = best_lda_model.transform(data_vectorized)
        # column names
        topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_components)]
        # index names
        docnames = ["Doc" + str(i) for i in range(len(doc_clean))]
        # Make the pandas dataframe
        df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

        # Get dominant topic for each document
        dominant_topic = np.argmax(df_document_topic.values, axis=1).tolist()

        #Count of dominant topic of each document, and add it to the topic_keywords_dict constructed earlier
        for topic_number in range(best_lda_model.n_components):
            topic_keywords_dict['topic ' + str(topic_number+1)]['frequency'] = dominant_topic.count(topic_number)

        # Create final DataFrame to be returned
        topic_model_df = pd.DataFrame()
        for topic in topic_keywords_dict:
                for keyword in topic_keywords_dict[topic]['keywords']:
                    keyword_dict = OrderedDict()
                    keyword_dict['Quarter'] = str(quarter)
                    keyword_dict['Brand'] = brand
                    keyword_dict['Type of Review'] = type_of_review.capitalize()
                    keyword_dict['Topic'] = topic
                    keyword_dict['Topic Frequency'] = topic_keywords_dict[topic]['frequency']
                    keyword_dict['Keyword'] = keyword[0]
                    keyword_dict['Keyword Weight'] = keyword[1]
                    # Get TF-IDF weight from "dict_of_words_and_idf", the dictionary created earlier
                    if keyword[0] in dict_of_words_and_idf:
                        keyword_dict['Keyword TF-IDF weight'] = dict_of_words_and_idf[keyword[0]]
                    else:
                        keyword_dict['Keyword TF-IDF weight'] = 0
                    topic_model_df = topic_model_df.append(keyword_dict, ignore_index=True)

        # To normalise keyword weights, we find the proportion of the keyword weight to the sum of all keyword weights
        frames = []
        for topic in topic_model_df['Topic'].unique():
            df = topic_model_df[topic_model_df['Topic'] == topic]
            sum_of_kw_weights = df['Keyword Weight'].sum()
            def normalise(value):
                return value/sum_of_kw_weights
            df['Keyword Weight'] = df['Keyword Weight'].apply(normalise)
            frames.append(df)
        topic_model_df = pd.concat(frames, ignore_index = True)
        return topic_model_df
    except ValueError:
        return pd.DataFrame()
    
def generate_keyword_similarity(pair, en_model):
    """
    Function:
    ---------
        (1) Try to find the corresponding vector of the word from the loaded FastText object for words in "pair".

        (2) Calculates the cosine similarity between the 2 word vectors

    Args:
    -----
        (1) pair (tuple): A tuple with 2 elements, which are the 2 words that are to be compared.

        (2) en_model (gensim.models.fasttext.FastText): Loaded weight matrix from Facebook’s native fasttext .bin and .vec output files.

    Returns:
    --------
        (1) similarity_score (float): A value between 0 and 1, representing the cosine similarity between keyword 1 and keyword 2 in "pair".

    References:
    -----------
        [1] API Reference:
        https://radimrehurek.com/gensim/models/fasttext.html#gensim.models.fasttext.FastText.load_fasttext_format
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html 
    """
    # Define keywords from pair
    keyword_1 = pair[0]
    keyword_2 = pair[1]
    
    # output_vec generates a word vector by trying to find the word in "en_model". If word is not in "en_model", it will output a matrix of zeros.
    def output_vec(keyword):
        featureVec = np.zeros((300,), dtype="float32")
        words = keyword.split()
        length_of_words = len([word for word in words if word in en_model])

        for word in words:
            if word in en_model:
                featureVec = np.add(featureVec, en_model.wv[word])
        featureVec /= length_of_words
        return featureVec

    # Calculate similarity score using scipy's "spatial.distance.cosine"
    keyword_1_vec = output_vec(keyword_1)
    keyword_2_vec = output_vec(keyword_2)
    similarity_score = 1 - spatial.distance.cosine(keyword_1_vec, keyword_2_vec)
    return similarity_score

def LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, LIST_OF_YEARS_TO_INCLUDE, number_of_topics_range, SEARCH_TERM):
    """
    Function:
    ---------
        (1) Prepares the "dict_of_clean_doc_by_quarter_by_brand", a nested Ordered Dictionary with the following three layers:

            (a) Type of Review
            
            (b) Quarter

            (c) Brand

        (2) Build an LDA model for every type of review, quarter and brand, by calling "build_single_LDA_model". GridSearch is implemented in "build_single_LDA_model".

        (3) Calculate coherence score of the final topic using word embedding, a similar method used in "context_similarity.py"

        (4) Write Results into Excel 
    Args:
    -----
        (1) DF (pandas DataFrame): DataFrame of user reviews

        (2) LIST_OF_ADDITIONAL_STOP_WORDS (list): List of additional stop words (str)

        (3) LIST_OF_COMMON_WORDS (list): List of common words (str), that you would want to not only remove the word from the review, but synonyms of the word as well.

        (4) LIST_OF_YEARS_TO_INCLUDE (list): List of years (str) to include for topic modelling. User comments will be removed if their date of comment is not in the list of years.

        (5) number_of_topics_range (list): List of number of topics (int) from which the GridSearch algorithm will iterate over, to find the best number of topics.

        (6) SEARCH_TERM (str): Term searched by the user.

    Returns:
    --------
        None
        Excel file will be saved to location 'Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand {}.xlsx'.format(SEARCH_TERM)

    References:
    -----------
        [1] API Reference:
        http://www.nltk.org/howto/wordnet.html

    """
 
    # Read in processed documents from cache, or process new document by calling from PreProcessing.py
    # Please check out PreProcessing.py for greater detail of the format of "dict_of_clean_doc_by_quarter_by_brand"
    if os.path.isfile('pickle_files/processed_data_by_quarter_by_brand.pickle'): 
        with open('pickle_files/processed_data_by_quarter_by_brand.pickle', 'rb') as handle_2:
            dict_of_clean_doc_by_quarter_by_brand = pickle.load(handle_2)
    else:
        dict_of_clean_doc_by_quarter_by_brand = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    # Generate list of quarters
    DF['Date'] = pd.to_datetime(DF['Date'],infer_datetime_format=True)
    DF['Y-Quarter'] = DF['Date'].dt.to_period("Q")
    list_of_quarters = DF['Y-Quarter'].unique()
    list_of_brands = DF['Brand'].unique()
    
    # Limit quarters to those in 2016, 2017, 2018
    list_of_quarters = [quarter for quarter in list_of_quarters if any(year in str(quarter) for year in LIST_OF_YEARS_TO_INCLUDE)]
    
    # Find all possible combinations of type of review, quarter, and brand using itertools.product
    combination_of_brands = []
    for type_of_review in ['positive', 'negative']:
        for quarter in list_of_quarters:
            combination_of_brands += list(itertools.product([str(quarter)], dict_of_clean_doc_by_quarter_by_brand[type_of_review][str(quarter)].keys(), [type_of_review]))
    
    # Constructed list of arguments to be passed into "build_single_LDA_model"
    print("{} products found... ".format(str(len(combination_of_brands))))
    list_of_arguments = [(dict_of_clean_doc_by_quarter_by_brand, str(quarter_brand[0]), quarter_brand[1], quarter_brand[2], number_of_topics_range) for quarter_brand in combination_of_brands]

    # Conduct multiprocessing, and concat the resulting dataframes into 1 dataframe    
    output_df = Manager().list()
    with Pool(processes= cpu_count() * 2) as pool:
        review_df = pool.starmap(build_single_LDA_model, list_of_arguments)

    pool.terminate()
    pool.join()
    
    output_df = pd.concat(review_df, ignore_index = True)    
    
    # Load Fast text pre-trained vectors, used to calculate coherence scores
    if os.path.isfile('pickle_files/{}.pickle'.format('fast_text_loaded')):
        print("Loading from Fast Text from cache...")
        with open('pickle_files/{}.pickle'.format('fast_text_loaded'), 'rb') as handle2:
            en_model = pickle.load(handle2)
    else:
        if os.path.isfile("Finding_Context_Similarity/wiki.en/wiki.en.bin"):
            print("Converting Fast Text into FastText Object...")
            en_model = FastText.load_fasttext_format("Finding_Context_Similarity/wiki.en/wiki.en.bin")
            with open('pickle_files/{}.pickle'.format('fast_text_loaded'), 'wb') as handle2:
                pickle.dump(en_model, handle2, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            print("Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'")
            return None
    
    # Calculate Coherence by iterating through every type of review, quarter, brand and topic number (as there will be multiple topics for one brand for  one quarter for a type of review)
    list_of_topics = output_df['Topic'].unique()
    frames_coherence_added = []
    
    for type_of_review in ['Positive', 'Negative']:
        for quarter in list_of_quarters:
            for brand in list_of_brands:
                for topic in list_of_topics:
                    df_brand = output_df[(output_df['Brand'] == brand) & (output_df['Quarter'] == str(quarter)) & (output_df['Type of Review'] == type_of_review) & (output_df['Topic'] == topic)]
                    if not df_brand.empty:
                        print("Generating for {}, {}...".format(brand, quarter))
                        # Generate list_of_keywords from the keywords generated from topic model for the particular brand, quarter, type of review, and topic number
                        list_of_keywords = df_brand['Keyword'].tolist()
                        # Generate a list of all the keyword pairings possible
                        combination_of_keywords = list(itertools.product(list_of_keywords,list_of_keywords))
                        # Exclude keyword pairing if both keywords in the pair are the same.
                        combination_of_keywords = [pair for pair in combination_of_keywords if pair[0] != pair[1]]
                        # Generate a similarity score for each keyword pairing, look at "generate_keyword_similarity" for more details. 
                        similarity_scores = [generate_keyword_similarity(pair, en_model) for pair in combination_of_keywords]
                        # Coherence level will be the average similarity score for all keyword pairings
                        try:
                            average_coherence = sum(similarity_scores)/len(similarity_scores)
                        except ZeroDivisionError:
                            average_coherence = 0
                        df_brand['Coherence Level'] = average_coherence
                        frames_coherence_added.append(df_brand)
                        
    output_df = pd.concat(frames_coherence_added, ignore_index = True)
    
    # Write results to excel
    writer = pd.ExcelWriter('Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand {}.xlsx'.format(SEARCH_TERM))
    output_df.to_excel(writer,'Topic Model by Quarter by Brand')
    writer.save()
    writer.close()
    return