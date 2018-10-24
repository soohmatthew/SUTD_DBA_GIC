#Standard library imports
import string
import pickle
from collections import OrderedDict

#Third party imports
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim
from gensim import corpora
from textblob import TextBlob

# Text Preprocessing
def clean(doc, LIST_OF_ADDITIONAL_STOP_WORDS):
    doc = str(doc)

    # For removing stopwords
    stop = set(stopwords.words('english'))
    for additional_word in LIST_OF_ADDITIONAL_STOP_WORDS:
        stop.add(additional_word)
    #stop.add(brand_name.lower())
    
    # For removing punctuations
    exclude = set(string.punctuation)

    lemma = WordNetLemmatizer()

    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def Preprocessing(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    dictionary_of_synonyms = OrderedDict()
    for word in LIST_OF_COMMON_WORDS:
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                dictionary_of_synonyms[l.name()] = word
    

    processed_data = OrderedDict()
    list_of_brands = df["Brand"].unique()
    for brand in list_of_brands:
        print("Processing {}...".format(brand))
        LIST_OF_ADDITIONAL_STOP_WORDS.append(brand.lower())
        positive_negative_dict = OrderedDict()

        doc_complete_positive = df[(df['Brand'] == brand) & (df['Rating'] == 5)]["User Comment"].tolist()
        doc_clean_positive = [clean(doc, LIST_OF_ADDITIONAL_STOP_WORDS).split() for doc in doc_complete_positive]
        doc_clean_syn_positive = [[dictionary_of_synonyms.get(word, word) for word in list_of_words] for list_of_words in doc_clean_positive]
        doc_clean_num_positive = [[word for word in list_of_words if not word.isdigit()] for list_of_words in doc_clean_syn_positive]

        doc_complete_negative = df[(df['Brand'] == brand) & (df['Rating'] == 1)]["User Comment"].tolist()
        doc_clean_negative = [clean(doc, LIST_OF_ADDITIONAL_STOP_WORDS).split() for doc in doc_complete_negative]
        doc_clean_syn_negative = [[dictionary_of_synonyms.get(word, word) for word in list_of_words] for list_of_words in doc_clean_negative]
        doc_clean_num_negative = [[word for word in list_of_words if not word.isdigit()] for list_of_words in doc_clean_syn_negative]

        positive_negative_dict["Positive"] = doc_clean_num_positive
        positive_negative_dict["Negative"] = doc_clean_num_negative

        processed_data[brand] = positive_negative_dict

    with open('pickle_files/{}.pickle'.format('processed_data'), 'wb') as handle:
        pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return processed_data

def build_single_LDA_model(dict_of_clean_doc, brand_name, positive_review, number_of_topics):
    if positive_review:
        doc_clean = dict_of_clean_doc[brand_name]["Positive"]
    else:
        doc_clean = dict_of_clean_doc[brand_name]["Negative"]
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(doc_clean)

    # convert tokenized documents into a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Try Running and Training LDA model on the document term matrix.
    try:
        ldamodel = gensim.models.LdaModel(corpus = doc_term_matrix, num_topics= number_of_topics, id2word = dictionary, passes = 100)
        print(ldamodel.print_topics(number_of_topics, 10))
        coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model_lda.get_coherence()
        print(coherence_score)
        # with open('pickle_files/{}.pickle'.format(brand_name), 'wb') as handle:
        #     pickle.dump(ldamodel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ldamodel, coherence_score
    except Exception as e:
        print("Not enough data")
        print(e)
        return None, None

def LDA_topic_modeller(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS):
    dict_of_clean_doc = Preprocessing(df, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)
    
    for number_of_topics in [2,3,4,5]:
        writer = pd.ExcelWriter('topic model results/LDA Topic Model {} topics.xlsx'.format(str(number_of_topics)))
        for type_of_review in [True, False]:
            topic_model = pd.DataFrame()
            if type_of_review:
                type_of_review_str = 'Positive'
            else:
                type_of_review_str = 'Negative'

            for brand in dict_of_clean_doc:
                print("Building LDA model for ... " + brand)
                model, coherence_score = build_single_LDA_model(dict_of_clean_doc, brand, type_of_review, number_of_topics)
                if model is not None:
                    list_of_topics = model.print_topics(number_of_topics, 10)
                    topic_model_dict = { "Brand" : brand}
                    for i in list_of_topics:
                        topic_model_dict["Topic No. {}".format(str(int(i[0]) + 1))] = i[1]
                    topic_model_dict["Coherence Score"] = coherence_score
                    topic_model = topic_model.append(topic_model_dict, ignore_index=True)
            
            with open('pickle_files/LDA_topic_model_{}_df.pickle'.format(type_of_review_str), 'wb') as handle:
                pickle.dump(topic_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
            topic_model.to_excel(writer, 'LDA Topic Model {}'.format(type_of_review_str))
        writer.save()
        writer.close()
    return

if __name__ == "__main__":
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
    LDA_topic_modeller(DF, LIST_OF_ADDITIONAL_STOP_WORDS, LIST_OF_COMMON_WORDS)

# # To be used only from Jupyter notebook
# import pyLDAvis.gensim
# lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display)
