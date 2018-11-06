#Standard library imports
import string
import pickle
from collections import OrderedDict
import os
import itertools

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

class LDAUsingPerplexityScorer(LatentDirichletAllocation):
    def score(self, X, y=None):
        score = super().perplexity(X, sub_sampling=False)
        # Perplexity is lower for better, negative scoring to simulate that.
        return -1*score

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def construct_stopwords(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    list_of_brands = DF["Brand"].unique()

    # If the word is in the list of common words, word and its synonyms will be added to list of stop words to be removed. 
    for word in LIST_OF_COMMON_WORDS:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                LIST_OF_ADDITIONAL_STOP_WORDS.append(l.name())

    #Remove brand name
    list_of_brands = [brand.lower() for brand in list_of_brands]
    LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS + list_of_brands

    list_of_stop_words = set(stopwords.words('english'))
    for additional_word in LIST_OF_ADDITIONAL_STOP_WORDS:
        list_of_stop_words.add(additional_word)
    return list_of_stop_words

def Preprocessing(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):

    stop_words = construct_stopwords(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)
    processed_data_by_quarter = OrderedDict()
    processed_data_by_quarter_by_brand = OrderedDict()

    df['Date'] = pd.to_datetime(df['Date'],infer_datetime_format=True)
    df['Y-Quarter'] = df['Date'].dt.to_period("Q")
    list_of_quarters = df['Y-Quarter'].unique()
    list_of_brands = df["Brand"].unique()
    
    df_positive = df[(df['Rating'] == 5) | (df['Rating'] == 4)]
    df_negative = df[(df['Rating'] == 1) | (df['Rating'] == 2)]

    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")

    for type_of_review in ['positive', 'negative']:
        if type_of_review == 'positive':
            df = df_positive
        elif type_of_review == 'negative':
            df = df_negative
        processed_data_by_quarter[type_of_review] = OrderedDict()
        processed_data_by_quarter_by_brand[type_of_review] = OrderedDict()
    
        for quarter in list_of_quarters:
            print("Processing {}...".format(quarter))
            
            doc_of_quarter = df[(df['Y-Quarter'] == quarter)]["User Comment"].tolist()

            #Tokenise words
            doc_of_quarter_token = list(sent_to_words(doc_of_quarter))

            #Lemmatize words
            doc_of_quarter_lemma = lemmatization(doc_of_quarter_token, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

            #Remove stopwords, digits.
            doc_of_quarter_stop = []
            for sentence in doc_of_quarter_lemma:
                sentence = ' '.join(word for word in sentence.split() if word not in stop_words and not word.isdigit())
                doc_of_quarter_stop.append(sentence)

            processed_data_by_quarter[type_of_review][str(quarter)] = doc_of_quarter_stop
            processed_data_by_quarter_by_brand[type_of_review][str(quarter)] = OrderedDict()

            for brand in list_of_brands:
                doc_of_quarter_by_brand = df[(df['Y-Quarter'] == quarter) & (df['Brand'] == brand)]["User Comment"].tolist()
                if doc_of_quarter_by_brand == []:
                    continue
                else:
                    doc_of_quarter_by_brand_token = list(sent_to_words(doc_of_quarter_by_brand))

                    #Lemmatize words
                    doc_of_quarter_by_brand_lemma = lemmatization(doc_of_quarter_by_brand_token, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

                    #Remove stopwords, digits.
                    doc_of_quarter_by_brand_stop = []
                    for sentence in doc_of_quarter_by_brand_lemma:
                        sentence = ' '.join(word for word in sentence.split() if word not in stop_words and not word.isdigit())
                        doc_of_quarter_by_brand_stop.append(sentence)

                    processed_data_by_quarter_by_brand[type_of_review][str(quarter)][brand] = doc_of_quarter_by_brand_stop

        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter'), 'wb') as handle:
            pickle.dump(processed_data_by_quarter, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand'), 'wb') as handle:
            pickle.dump(processed_data_by_quarter_by_brand, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return processed_data_by_quarter, processed_data_by_quarter_by_brand


def build_single_LDA_model(dict_of_clean_doc, quarter, brand, type_of_review, number_of_topics_range):
    try:
        vectorizer = CountVectorizer(analyzer='word',       
                                min_df= 3,                        # minimum required occurence of a word 
                                stop_words='english',             # remove stop words
                                lowercase=True,                   # convert all words to lowercase
                                token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                )

        if brand is None:
            print("Building LDA model for ... " + str(quarter))
            doc_clean = dict_of_clean_doc[type_of_review][str(quarter)]
        else:
            print("Building LDA model for ... {}, {} ".format(str(quarter), brand))
            doc_clean = dict_of_clean_doc[type_of_review][str(quarter)][brand]

        data_vectorized = vectorizer.fit_transform(doc_clean)

        lda_model = LDAUsingPerplexityScorer(max_iter=10,               # Max learning iterations
                                            learning_method='online',   
                                            random_state=100,          # Random state
                                            batch_size=128,            # n docs in each learning iter
                                            evaluate_every = -1,       # compute perplexity every n iters, default: Don't
                                            n_jobs = -1,               # Use all available CPUs
                                            )

        search_params = {'n_components': number_of_topics_range, 'learning_decay': [.5, .7, .9]}
        grid_search_model = GridSearchCV(lda_model, param_grid=search_params)
        grid_search_model.fit(data_vectorized)

        best_lda_model = grid_search_model.best_estimator_

        #Creates a dictionary, where the key is the nth topic, 
        #and the corresponding value is a dictionary, with a 'keyword' key, and 
        #list of tuples of the top 10 keyword and the associated weight as the value

        def show_topics(vectorizer, lda_model, n_words=10):
            keywords = np.array(vectorizer.get_feature_names())
            topic_keywords = []
            for topic_weights in lda_model.components_:
                top_keyword_locs = (-topic_weights).argsort()[:n_words]

                # Get top 10 keywords and the associated weight
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

        topic_model_df = pd.DataFrame()
        if brand is None:
            for topic in topic_keywords_dict:
                    for keyword in topic_keywords_dict[topic]['keywords']:
                        keyword_dict = OrderedDict()
                        keyword_dict['Quarter'] = str(quarter)
                        keyword_dict['Type of Review'] = type_of_review.capitalize()
                        keyword_dict['Topic'] = topic
                        keyword_dict['Topic Frequency'] = topic_keywords_dict[topic]['frequency']
                        keyword_dict['Keyword'] = keyword[0]
                        keyword_dict['Keyword Weight'] = keyword[1]
                        topic_model_df = topic_model_df.append(keyword_dict, ignore_index=True)

        else:
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
                        topic_model_df = topic_model_df.append(keyword_dict, ignore_index=True)

        return topic_model_df
    except ValueError:
        return pd.DataFrame()

def LDA_topic_modeller_by_quarter_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range):
    #Read in processed documents from cache, or process new document
    if os.path.isfile('pickle_files/{}.pickle'.format('processed_data_by_quarter')) and os.path.isfile('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand')): 
        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter'), 'rb') as handle:
            dict_of_clean_doc_by_quarter = pickle.load(handle)
    else:
        dict_of_clean_doc_by_quarter, _ = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    #Generate list of quarters
    DF['Date'] = pd.to_datetime(DF['Date'],infer_datetime_format=True)
    DF['Y-Quarter'] = DF['Date'].dt.to_period("Q")
    list_of_quarters = DF['Y-Quarter'].unique().tolist()
    
    #Limit quarters to those in 2016, 2017, 2018
    list_of_years_to_include = ['2016','2017','2018']
    list_of_quarters = [quarter for quarter in list_of_quarters if any(year in str(quarter) for year in list_of_years_to_include)]

    from multiprocessing import Pool, cpu_count, Manager
    print("{} products found... ".format(str(len(list_of_quarters))))
    
    combination_of_brands = []
    for quarter in list_of_quarters:
        combination_of_brands += list(itertools.product([str(quarter)], ['positive', 'negative']))

    list_of_arguments = [(dict_of_clean_doc_by_quarter, str(quarter_brand[0]), None, quarter_brand[1], number_of_topics_range) for quarter_brand in combination_of_brands]

    output_df = Manager().list()

    with Pool(processes= cpu_count() * 2) as pool:
        review_df = pool.starmap(build_single_LDA_model, list_of_arguments)

    output_df = output_df.append(review_df)
    pool.terminate()
    pool.join()
    
    output_df = pd.concat(review_df, ignore_index = True)

    writer = pd.ExcelWriter('Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter.xlsx')
    
    output_df.to_excel(writer,'Topic Model by Quarter')
    writer.save()
    writer.close()
    return

def LDA_topic_modeller_by_quarter_by_brand_multiprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS, number_of_topics_range):
    #Read in processed documents from cache, or process new document
    if os.path.isfile('pickle_files/{}.pickle'.format('processed_data_by_quarter')) and os.path.isfile('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand')): 
        with open('pickle_files/{}.pickle'.format('processed_data_by_quarter_by_brand'), 'rb') as handle_2:
            dict_of_clean_doc_by_quarter_by_brand = pickle.load(handle_2)
    else:
        _, dict_of_clean_doc_by_quarter_by_brand = Preprocessing(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

    #Generate list of quarters
    DF['Date'] = pd.to_datetime(DF['Date'],infer_datetime_format=True)
    DF['Y-Quarter'] = DF['Date'].dt.to_period("Q")
    list_of_quarters = DF['Y-Quarter'].unique()
    
    #Limit quarters to those in 2016, 2017, 2018
    list_of_years_to_include = ['2016','2017','2018']
    list_of_quarters = [quarter for quarter in list_of_quarters if any(year in str(quarter) for year in list_of_years_to_include)]
    
    combination_of_brands = []
    for type_of_review in ['positive', 'negative']:
        for quarter in list_of_quarters:
            combination_of_brands += list(itertools.product([str(quarter)], dict_of_clean_doc_by_quarter_by_brand[type_of_review][str(quarter)].keys(), [type_of_review]))

    from multiprocessing import Pool, cpu_count, Manager
    print("{} products found... ".format(str(len(combination_of_brands))))
    list_of_arguments = [(dict_of_clean_doc_by_quarter_by_brand, str(quarter_brand[0]), quarter_brand[1], quarter_brand[2], number_of_topics_range) for quarter_brand in combination_of_brands]

    output_df = Manager().list()

    with Pool(processes= cpu_count() * 2) as pool:
        review_df = pool.starmap(build_single_LDA_model, list_of_arguments)

    pool.terminate()
    pool.join()
    
    
    output_df = pd.concat(review_df, ignore_index = True)    
    
    writer = pd.ExcelWriter('Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand.xlsx')
    output_df.to_excel(writer,'Topic Model by Quarter by Brand')
    writer.save()
    writer.close()
    return