#Standard library imports
import re
import string
import pickle
import os
import itertools

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

nlp = spacy.load('en_core_web_sm')

def clean_up_corpus(sentence):
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
    spacy_doc = nlp(sentence)
    list_of_noun_phrase = [np.text for np in spacy_doc.noun_chunks]
    return list_of_noun_phrase

def verb_phrase_extractor(sentence):
    doc = textacy.Doc(sentence, lang='en_core_web_sm')
    verb_phrase = r'<VERB>?<ADV | DET>*<VERB>+'
    list_of_verb_phrase = [list.text for list in textacy.extract.pos_regex_matches(doc, verb_phrase)]
    return list_of_verb_phrase

def prepositional_phrase_extractor(sentence):
    doc = textacy.Doc(sentence, lang='en_core_web_sm')
    prepositional_phrase = r'<PREP>? <DET>? (<NOUN>+<ADP>)* <NOUN>+'
    list_of_prepositional_phrase = [list.text for list in textacy.extract.pos_regex_matches(doc, prepositional_phrase)]
    return list_of_prepositional_phrase

# Figure out if the hypothesis constitutes a noun, verb or prepositional phrase. If multiple phrases detected, return the list with the most phrases.
def break_down_hypothesis(hypothesis):
    dict_of_hypothesis_phrases = {"Noun Phrases" : noun_phrase_extractor(hypothesis), 
                                  "Verb Phrases" : verb_phrase_extractor(hypothesis),
                                  "Prepositional Phrases" : prepositional_phrase_extractor(hypothesis)}

    output = [(type_of_phrase, list_of_phrases) for type_of_phrase, list_of_phrases in dict_of_hypothesis_phrases.items() if len(list_of_phrases) > 0]
    if len(output) == 0:
        print("Unable to pick out any noun phrase, verb phrase or prepositional phrase, try a different hypothesis!")
        return None, None
    elif len(output) > 1:
        length = ("length_check", [])
        for pair in output:
            if len(pair[1]) > len(length[1]):
                length = pair
        return length
    else:
        return output[0]
    
# Processes the user comments, and loads the FastText en_model as well.
def clean_and_load_data(LIST_OF_YEARS_TO_INCLUDE, REPROCESS = False):
    LIST_OF_YEARS_TO_INCLUDE = [int(year) for year in LIST_OF_YEARS_TO_INCLUDE]
    if REPROCESS == True:
        print("Reprocessing text data...")
        df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of coffee machine.xlsx")
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
        with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'wb') as handle:
            pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load Cleaned up file
        if os.path.isfile('pickle_files/{}.pickle'.format('processed_data_for_context')):
            with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'rb') as handle:
                print("Loading Processed Text from cache...")
                df = pickle.load(handle)
        else:
            print("Cache not found, processing text data...")
            df = pd.read_excel("Webscraping/Review Corpus/Customer Reviews of coffee machine.xlsx")
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
            with open('pickle_files/{}.pickle'.format('processed_data_for_context'), 'wb') as handle:
                pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Load Fast text pre-trained vectors
    if os.path.isfile('pickle_files/{}.pickle'.format('fast_text_loaded')):
        print("Loading Fast Text from cache...")
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
    return df, en_model

# Creates a list of all the different possible combinations of 
def find_max_similarity_score(list_of_phrases, list_of_hypothesis, en_model):
    combination_of_phrases = list(itertools.product(list_of_phrases, list_of_hypothesis))
    scores = [compute_cosine_similarity(phrases[0], phrases[1], en_model) for phrases in combination_of_phrases]     
    return max(scores, default = 0)

# Function to average all words vectors in a given sentence
def avg_phrase_vector(phrase, en_model):
    featureVec = np.zeros((300,), dtype="float32")
    words = phrase.split()
    length_of_words = len([word for word in words if word in en_model])
    
    for word in words:
        if word in en_model:
            featureVec = np.add(featureVec, en_model.wv[word])
    featureVec /= length_of_words
    return featureVec

# Computes cosine similarity of the vectorised user review, and vectorised hypothesis, returns a score
def compute_cosine_similarity(USER_REVIEW, HYPOTHESIS, en_model):
    USER_REVIEW_VEC = avg_phrase_vector(USER_REVIEW, en_model)
    HYPOTHESIS_VEC = avg_phrase_vector(HYPOTHESIS, en_model)
    return 1 - spatial.distance.cosine(HYPOTHESIS_VEC, USER_REVIEW_VEC)

def apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, REPROCESS = False):
    print("Analyzing hypothesis ...")
    type_of_phrase, list_of_hypothesis_phrases = break_down_hypothesis(hypothesis)
    print("Hypothesis evaluated: ")
    print(list_of_hypothesis_phrases)
    print("Loading reviews and fast Text vectors ...")
    if list_of_hypothesis_phrases is not None:
        df, en_model = clean_and_load_data(LIST_OF_YEARS_TO_INCLUDE, REPROCESS)
        if df is not None:
            df['Similarity to Hypothesis'] = df['User Comment ({})'.format(type_of_phrase)].swifter.apply(find_max_similarity_score, list_of_hypothesis = list_of_hypothesis_phrases, en_model = en_model)
            df['Hypothesis'] = hypothesis

            # Add polarity bias
            df.loc[df.Rating >= 4, 'Similarity to Hypothesis'] += POLARITY_BIAS
            df.loc[df.Rating <= 2, 'Similarity to Hypothesis'] -= POLARITY_BIAS 

            return df
        else:
            return None

def construct_similarity_table(hypothesis, LIST_OF_YEARS_TO_INCLUDE, THRESHOLD, POLARITY_BIAS, REPROCESS = False):
    if isinstance(hypothesis, list):
        for hypothesis in hypothesis:
            print(hypothesis)
            vectorised_reviews_df = apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, REPROCESS)

            if vectorised_reviews_df is not None:
                # Data Munging to construct table
                vectorised_reviews_df_sorted = vectorised_reviews_df.sort_values(['Similarity to Hypothesis'], ascending=[False])
                vectorised_reviews_df_final = vectorised_reviews_df_sorted.groupby(['Y-Quarter', 'Brand'])

                frames = []
                cols_to_include = ['Y-Quarter', 'Company', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity Score']
                for key in vectorised_reviews_df_final.groups.keys():
                    df_sorted_by_brand = vectorised_reviews_df_sorted[(vectorised_reviews_df_sorted['Y-Quarter'] == key[0]) & (vectorised_reviews_df_sorted['Brand'] == key[1])]
                    # Overall Similarity Score is calculated by setting a threshold of 0.6, if similarity to hypothesis exceeds the threshold, and the proportion of occurences with respect to total number of reviews by brand by quarter.
                    df_sorted_by_brand['Overall Similarity Score'] = ((df_sorted_by_brand['Similarity to Hypothesis'] > THRESHOLD) * 1).mean()
                    df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include]
                    frames.append(df_sorted_by_brand_selected_cols.head(5))

                output_df = pd.concat(frames, ignore_index = True)

                # Write to excel
                writer = pd.ExcelWriter("Finding_Context_Similarity/Similarity Table Results/Similarity Table {}.xlsx".format(hypothesis))
                output_df.to_excel(writer,'Ranked by Quarter by Brand')
                writer.save()
                writer.close()

    elif isinstance(hypothesis, str):
        vectorised_reviews_df = apply_cosine_similarity_to_df(hypothesis, LIST_OF_YEARS_TO_INCLUDE, POLARITY_BIAS, REPROCESS)
        if vectorised_reviews_df is not None:
            # Data Munging to construct table
            vectorised_reviews_df_sorted = vectorised_reviews_df.sort_values(['Similarity to Hypothesis'], ascending=[False])
            vectorised_reviews_df_final = vectorised_reviews_df_sorted.groupby(['Y-Quarter', 'Brand'])

            frames = []
            cols_to_include = ['Y-Quarter', 'Company', 'Brand', 'Name', 'Rating', 'Source', 'User Comment', 'Similarity to Hypothesis' ,'Overall Similarity']
            for key in vectorised_reviews_df_final.groups.keys():
                df_sorted_by_brand = vectorised_reviews_df_sorted[(vectorised_reviews_df_sorted['Y-Quarter'] == key[0]) & (vectorised_reviews_df_sorted['Brand'] == key[1])]
                # Overall Similarity Score is calculated by setting a threshold of 0.6, if similarity to hypothesis exceeds the threshold, and the proportion of occurences with respect to total number of reviews by brand by quarter.
                df_sorted_by_brand['Overall Similarity Score'] = ((df_sorted_by_brand['Similarity to Hypothesis'] > THRESHOLD) * 1).mean()
                df_sorted_by_brand_selected_cols = df_sorted_by_brand.loc[:,cols_to_include]
                frames.append(df_sorted_by_brand_selected_cols.head(5))

            output_df = pd.concat(frames, ignore_index = True)

            # Write to excel
            writer = pd.ExcelWriter("Finding_Context_Similarity/Similarity Table Results/Similarity Table {}.xlsx".format(hypothesis))
            output_df.to_excel(writer,'Ranked by Quarter by Brand')
            writer.save()
            writer.close()
        else:
            return
    else:
        print("Hypothesis statement should be a list or a string.")