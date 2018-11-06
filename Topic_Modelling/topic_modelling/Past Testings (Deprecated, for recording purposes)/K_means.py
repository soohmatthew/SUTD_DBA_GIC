#Standard library imports
from collections import OrderedDict

#Third party imports
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd

def build_single_K_means_model(df, brand_name, num_topics, num_words, LIST_OF_ADDITIONAL_STOP_WORDS, positive_reviews = True):
        topic_model = pd.DataFrame()
        if positive_reviews:
                document = df[(df["Rating"] == 5)]["User Comment"].tolist()
        else:
                document = df[(df["Rating"] == 1)]["User Comment"].tolist()
        if len(document) <= num_words:
                return None
        else:   
                # Ensure that only strings are passed in
                document = [str(doc) for doc in document]

                # Accommodate extra words that you want to exclude from the model
                stop_words = text.ENGLISH_STOP_WORDS.union(LIST_OF_ADDITIONAL_STOP_WORDS)
                vectorizer = TfidfVectorizer(stop_words = stop_words)
                X = vectorizer.fit_transform(document)

                model = KMeans(n_clusters=num_topics, init='k-means++', max_iter=100, n_init=1)
                model.fit(X)

                # Sort the coordinates based on distance from the centroids, get the index
                order_centroids = model.cluster_centers_.argsort()

                #Reverse the order of the indexs, because we want those closest to the centroid.
                order_centroids = order_centroids[:, ::-1]

                terms = vectorizer.get_feature_names()

                topic_model_dict = OrderedDict()
                topic_model_dict["Brand"] = brand_name
                for i in range(num_topics):
                        topic_model_dict["Topic {}".format(str(i+1))] = [terms[ind] for ind in order_centroids[i, :num_words]]
                topic_model = topic_model.append(topic_model_dict, ignore_index=True)
                        
                return topic_model

def K_means_topic_modeller(df, LIST_OF_ADDITIONAL_STOP_WORDS, num_topics = 3, num_words = 10):
        writer = pd.ExcelWriter('topic model results/K Means Topic Model_{}_topics.xlsx'.format(num_topics))
        list_of_topic_model = []
        list_of_brands = df["Brand"].unique()
        for type_of_review in [True, False]:
                if type_of_review:
                        type_of_review_str = 'Positive'
                else:
                        type_of_review_str = 'Negative'
                for brand in list_of_brands:
                        print("Building k-means model for... {}".format(brand))
                        LIST_OF_ADDITIONAL_STOP_WORDS.append(brand)
                        df_by_brand = df[(df["Brand"] == brand)]
                        model = build_single_K_means_model(df = df_by_brand, 
                                                           brand_name = brand,
                                                           num_topics = num_topics,
                                                           num_words = num_words, 
                                                           LIST_OF_ADDITIONAL_STOP_WORDS = LIST_OF_ADDITIONAL_STOP_WORDS, 
                                                           positive_reviews = type_of_review)
                        if model is not None:
                                list_of_topic_model.append(model)
                topic_model = pd.concat(list_of_topic_model)
                topic_model.to_excel(writer, 'K Means Topic Model {}'.format(type_of_review_str))
        writer.save()
        return