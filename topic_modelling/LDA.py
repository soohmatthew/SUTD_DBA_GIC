#Standard library imports
import string
import pickle

#Third party imports
import pandas as pd
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim import corpora

def build_single_LDA_model(df, brand_name, number_of_topics,  LIST_OF_ADDITIONAL_STOP_WORDS = [], positive_reviews = True):
    doc_complete = []
    if positive_reviews:
        doc_complete = df[(df['Brand'] == brand_name) & (df['Rating'] == 5)]["User Comment"].tolist()
    else:
        doc_complete = df[(df['Brand'] == brand_name) & (df['Rating'] == 1)]["User Comment"].tolist()

    # Text Preprocessing
    def clean(doc):
        doc = str(doc)
        stop = set(stopwords.words('english'))
        for additional_word in LIST_OF_ADDITIONAL_STOP_WORDS:
            stop.add(additional_word)
        stop.add(brand_name)
        exclude = set(string.punctuation) 
        lemma = WordNetLemmatizer()

        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]
    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(doc_clean)

    # convert tokenized documents into a document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Try Running and Training LDA model on the document term matrix.
    try:
        ldamodel = Lda(doc_term_matrix, num_topics= number_of_topics, id2word = dictionary, passes=100)
        with open('pickle_files/{}.pickle'.format(brand_name), 'wb') as handle:
            pickle.dump(ldamodel, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return ldamodel
    except ValueError:
        print("Not enough data")
        return None

def LDA_topic_modeller(df, num_topics = 3, num_words = 10, LIST_OF_ADDITIONAL_STOP_WORDS = []):
    list_of_brands = df["Brand"].unique()
    writer = pd.ExcelWriter('topic model results/LDA Topic Model.xlsx')
    for type_of_review in [True, False]:
        topic_model = pd.DataFrame()
        if type_of_review:
            type_of_review_str = 'Positive'
        else:
            type_of_review_str = 'Negative'

        for brand in list_of_brands:
            print("Building LDA model for ... " + brand)
            model = build_single_LDA_model(df, brand, num_topics, LIST_OF_ADDITIONAL_STOP_WORDS, positive_reviews = type_of_review)
            if model is not None:
                list_of_topics = model.print_topics(num_topics, num_words)
                topic_model_dict = { "Brand" : brand}
                for i in list_of_topics:
                    topic_model_dict["Topic No. {}".format(str(int(i[0]) + 1))] = i[1]
                topic_model = topic_model.append(topic_model_dict, ignore_index=True)
        
        with open('pickle_files/LDA_topic_model_{}_df.pickle'.format(type_of_review_str), 'wb') as handle:
            pickle.dump(topic_model, handle, protocol=pickle.HIGHEST_PROTOCOL)   
        topic_model.to_excel(writer, 'LDA Topic Model {}'.format(type_of_review_str))
    writer.save()
    return

# # To be used only from Jupyter notebook
# import pyLDAvis.gensim
# lda_display = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary, sort_topics=False)
# pyLDAvis.display(lda_display)
