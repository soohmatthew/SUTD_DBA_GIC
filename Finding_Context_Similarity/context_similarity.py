#Standard library imports
import re
import string
import pickle
import os

#Third party library imports
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
from gensim.models.fasttext import FastText
import swifter
from scipy import spatial
import openpyxl

# Preprocessing of Text
def clean_up_review(text):
    # 1. Tokenize review
    tokens = word_tokenize(text)
    # 2. Convert to lower case
    tokens = [w.lower() for w in tokens]
    # 3. Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # 4. Remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # 5. Filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

# Function to average all words vectors in a given sentence
def avg_sentence_vector(words, en_model):
    model = en_model
    featureVec = np.zeros((300,), dtype="float32")
    nwords = 0

    for word in words:
        try:
            nwords = nwords+1
            featureVec = np.add(featureVec, model.wv[word])
        except KeyError:
            continue

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

# Returns the vectorised version of the user reviews, as well as the hypothesis statement
def vectorise_user_comments(PROCESSED_HYPOTHESIS_STATEMENT, REPROCESS = False):
    if REPROCESS == True:
        df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of coffee machine.xlsx")
        df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
        df['Y-Quarter'] = df['Date'].dt.to_period("Q")
        df['User Comment Cleaned'] = df['User Comment'].apply(clean_up_review)
        with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load Cleaned up file
        if os.path.isfile('pickle_files/{}.pickle'.format('processed_data_for_context')):
            with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'rb') as handle:
                print("Loading from Processed Text from cache...")
                df = pickle.load(handle)
        else:
            print("Cache not found, processing text data...")
            df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of coffee machine.xlsx")
            df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
            df['Y-Quarter'] = df['Date'].dt.to_period("Q")
            df['User Comment Cleaned'] = df['User Comment'].apply(clean_up_review)
            with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load Fast text pre-trained vectors
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
            return None, None

    df['User Comment Vectorised'] = df['User Comment Cleaned'].swifter.apply(avg_sentence_vector, en_model = en_model)
    vectorised_hypothesis_statement = avg_sentence_vector(PROCESSED_HYPOTHESIS_STATEMENT , en_model) 
    return df, vectorised_hypothesis_statement

# Computes cosine similarity of the vectorised user review, and vectorised hypothesis, returns a score
def compute_cosine_similarity(USER_REVIEW, HYPOTHESIS):
    return 1 - spatial.distance.cosine(HYPOTHESIS, USER_REVIEW)

# Implements cosine similarity to dataframe, returns a dataframe with scores
def check_cosine_similarity(HYPOTHESIS_STATEMENT):
    # Preprocess hypothesis statement
    print("Cleaning up hypothesis statement...")
    processed_hypothesis = clean_up_review(HYPOTHESIS_STATEMENT)

    # Create vectorised user comment
    vectorised_reviews_df, vectorised_hypothesis_statement = vectorise_user_comments(processed_hypothesis)

    if vectorised_reviews_df is not None:        
        vectorised_reviews_df["Similarity to Hypothesis"] = vectorised_reviews_df['User Comment Vectorised'].swifter.apply(compute_cosine_similarity, HYPOTHESIS = vectorised_hypothesis_statement)
        vectorised_reviews_df["Hypothesis"] = HYPOTHESIS_STATEMENT
        return vectorised_reviews_df
    else:
        return None


# Combining everything together
def construct_similarity_table(HYPOTHESIS_STATEMENT):
    vectorised_reviews_df = check_cosine_similarity(HYPOTHESIS_STATEMENT)

    if vectorised_reviews_df is not None:
        # Data Munging to construct table
        vectorised_reviews_df_sorted = vectorised_reviews_df.sort_values(['Similarity to Hypothesis'], ascending=[False])
        vectorised_reviews_df_final = vectorised_reviews_df_sorted.groupby(['Y-Quarter', 'Brand'])

        frames = []
        cols_to_include = ['Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Average Similarity']
        for key in vectorised_reviews_df_final.groups.keys():
            df_sorted_by_brand = vectorised_reviews_df_sorted[(vectorised_reviews_df_sorted['Y-Quarter'] == key[0]) & (vectorised_reviews_df_sorted['Brand'] == key[1])]
            df_sorted_by_brand['Average Similarity'] = df_sorted_by_brand['Similarity to Hypothesis'].mean()
            df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include]
            frames.append(df_sorted_by_brand_selected_cols.head(5))

        output_df = pd.concat(frames, ignore_index = True)
        
        # Write to excel
        writer = pd.ExcelWriter("Finding_Context_Similarity/Similarity Table Results/Similarity Table - '{}'.xlsx".format(HYPOTHESIS_STATEMENT))
        output_df.to_excel(writer,'Ranked by Quarter by Brand')
        writer.save()
        writer.close()
    else:
        return
