# Investment Insights with NLP
 
 This is a repository for the <i>40.011 : Data and Business Analytics Project</i>, in collaboration with GIC to build a data pipeline that would gather investment insights on various products.

## Group Members:
1. Matthew Soh
2. Toni Celine Gutierrez Hilos
3. Lam Xue Wei
4. Clarence Toh Jun Wen
5. Ng Au Ker Wesson

## Project Flow

1. Data Collection: Web-scraping of popular e-commerce websites, to build a corpus of product reviews. 
2. Topic Modelling: Building a topic model, using various topic modelling algorithms, trained with the corpus we have built.
3. Contextual Similarity: Comparing a given hypothesis with the corpus of product reviews, to validate which reviews yield the greatest similarity with the hypothesis. Reviews can then be sorted by brand.

## Prerequisites

1. The script is tested on Python 3.6.5 
2. Install the following libraries on your system, in order for the code to run. 

```
numpy==1.15.3
textblob==0.15.1
fake_useragent==0.1.11
swifter==0.260
pandas==0.23.0
lxml==4.2.1
nltk==3.3
spacy==2.0.16
scipy==1.1.0
gensim==3.6.0
requests==2.20.0
openpyxl==2.5.3
beautifulsoup4==4.6.3
python_dateutil==2.7.5
scikit_learn==0.20.0
```

Alternatively, pip install the requirements.txt

```
pip install -r requirements.txt
```

3. Download SpaCy's English model by running the following code in your terminal:

```
python -m spacy download en
python -m spacy download en_core_web_sm
```

"en_core_web_sm" model is used for context similarity, while "en" model is used for preprocessing of text for topic modelling.

4. Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'

## Usage

### Configuration of ```main.py```

<i> Flags </i>

1. Individual functions can be turned on or off by passing ```True``` or ```False``` in the following dictionary in ```main.py```

```
WEBSCRAPER_FLAG = {"AMAZON" : True,
              "WALMART" : True,
              "BESTBUY" : True}

TOPIC_MODELLING_FLAG = {"LDA_W_GRIDSEARCH" : True,
                   "HDP" : True}

CONTEXTUAL_SIMILARITY_FLAG = {"DOC2VEC": True,
                              "CONTEXTUAL_SIMILARITY_W_FASTTEXT" : True}
```

<i> For the webscraper </i>

1. `SEARCH_TERM`, according to whatever product you wish to scrape reviews for. Default is `coffee machines`

<i> For the topic model </i>

1. `PATH_TO_REVIEW_DOC`, where the results of the webscraping is stored. Advised not to change location.

2. `LIST_OF_WORDS_TO_EXCLUDE`, to remove any words that may not be useful during the topic modelling process. Default is `['one', 'two', 'three', 'four', 'five', 'star']`.

3. `LIST_OF_COMMON_WORDS`, which is an extension of ```LIST_OF_WORDS_TO_EXCLUDE```, just that it also includes the synonym of the words. Default is `["good", "great", "love"]`.

4. `NUMBER_OF_TOPICS_RANGE`, a range of number of topics in which the algorithm will search over, to generate the most suitable number of topic for the set of documents. Default is `[2,3,4,5]`. <b>Only applicable for the LDA model</b> 

5. `LIST_OF_YEARS_TO_INCLUDE`, the list of years that will be considered when building the model, any year that is not inside this list will be excluded.

<i> For the contextual similarity model </i>

1. `HYPOTHESIS_STATEMENT`, based on what your hypothesis statement is. Default is `breakdown`.

```main.py``` can be run through your terminal, once relevant packages and documents have been downloaded and installed, and configured to the appropriate settings.

### Configuration of Individual Scripts

Running of individual portions can be done via the following scripts, scripts need to be configured as well, with the same settings as above.
Webscraping:  ```scrape_main.py```
Topic Modelling: ```topic_model_main.py```
Contextual Similarity: ```find_contextual_similarity_main.py```

#### Dealing with pickle files

For this project, both webscraping and topic model generation take an extended period of time. In order to avoid any accidental loss of intermediate data, pickle files are used for caching purposes.

If you can delete the pickle file once the python script has finished running, or you could add the following code at the end of the ``` if __name__ == "__main__" ``` statement in ```main.py```. 

```
currentdir = os.getcwd()
shutil.rmtree(currentdir + '/pickle_files')
```


## Methodology

### 1. Data Collection

The main script that triggers the Data Collection process is ```scrape_main.py```. 

```scrape_main.py``` takes in ```SEARCH TERM``` as a keyword, and will initialize the scrapping process from the following websites (more to be added):

1. Amazon
2. Best Buy 
3. Walmart 

<i> *Each scraper is written by a different contributor, thus the functionality is slightly different for each. Furthermore, each scraper addresses the format of a specific website, and is sometimes limited to the restrictions of the website. </i>

The default keyword for ```SEARCH_TERM``` is ```coffee machine```. 
Scraping process will take ~1 hour, depending on how much there is to scrape. Multiprocessing is available for Amazon and Best Buy.

#### Data Collection: Expected Output:

An excel file labelled ```'Webscraping/Review Corpus/Customer Reviews of SEARCH_TERM.xlsx'``` will be downloaded to your system.

### 2. Topic Modelling

The main script that triggers the Topic Modelling process is ```topic_model_main.py```. This script will take in the output of ```scrape_main.py```, namely ```'Webscraping/Review Corpus/Customer Reviews of SEARCH_TERM.xlsx'``` and generate topic models. 

Generally, topic models will produce a list of keywords that most likely form a topic. As it is, the output of these topic modelling algorithms still require some form of interpretation and investors still need to piece the keywords together to form the story. Each keyword is accompanied with a keyword weight, which is the model's interpretation of how important the keyword is in association with the topic. For our implementation, we only take the top 10 keywords, but this can be changed very easily in the code.

The metric we used to judge the performance of a topic model is using the <b>Coherence Score</b>. Simply put, the coherence score would be calculated based on whether the keywords produced are contextually relevant, either by comparing these words to the corpus or by using word embeddings to generate some average similarity score.

Raw text was first preprocessed, to remove all stopwords, punctuations, and for words to all be lemmatized. Processed text is then split into different categories, by quarter, and then by brand.

Several topic models were built using different algorithms. The following algorithms tested did not yield as good results. They can be viewed under ```Topic_Modelling/topic_modelling/Past Testings (Deprecated, for recording purposes```.

1. Latent Semantic Analysis (LSA)

2. k-means Clustering

#### 2a. [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

Latent Dirichlet Allocation (LDA) <i>Blei, David M.; Ng, Andrew Y.; Jordan, Michael I (January 2003)</i> is usually the de facto algorithm that is used for topic modelling, it is tried and tested, and the results are usually reasonable, as long as the model is properly tuned and the corpus used has been preprocessed well <i>(Garbage in, garbage out)</i>. 

LDA requires the user to pre-specify the number of topics that it should search for. Because of this, the ```scikit learn``` library was picked over the ```gensim``` library, due to the ability for the ```scikit learn``` library to apply GridSearch over a range of topic numbers to find the best topic model. The ```scikit learn``` library was used to generate the LDA model, and the metric used for selecting the best model is lowest perplexity (as opposed to coherence, as it was not readily available).

Topic modelling is implemented with multiprocessing, and should take no more than 15 minutes.

#### 2b. [Hierarchical Dirichlet Process (HDP)](https://en.wikipedia.org/wiki/Hierarchical_Dirichlet_process)

Hierarchical Dirichlet Process (HDP) <i>Yee Whye Teh, Y. W.; Jordan, M. I.; Beal, M. J.; Blei, D. M. (2006).</i> was built as an extension of LDA, and has an edge over LDA, in the sense that the user does not need to pre-specify the number of topics that HDP should search over. Instead, it outputs a maximum number of topics, and the user can then choose the important topics. For our current implementation, we use the sum of keyword weights within a topic for the top 10 words, to rank the topics by importance, and set a threshold weight of 0.25 to filter out unimportant topics. We implemented HDP with the help of the ```gensim``` library, and was sorted by quarter by brand, like LDA. We give a coherence score to each topic generated, which can act as a filter for the users. A short and sweet explanation of the difference between HDP and LDA can be found [here](https://datascience.stackexchange.com/a/296)

Topic modelling is not implemented with multiprocessing for HDP, and will take slightly longer to run compared to LDA.

#### Topic Modelling: Expected Output

2 Excel Files, ```Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand.xlsx``` and ```Topic_Modelling/Topic Model Results/HDP Topic Model by Quarter by Brand.xlsx```.

### 3. Contextual Similarity

The main script that triggers the Contextual Similarity process is ```find_contextual_similarity_main.py```. Essentially, the algorithm takes in a hypothesis as an input, and compares the hypothesis with the user reviews, and returns similarity score of the hypothesis and the user review. Scores are then aggregated and grouped based on the Quarter and the Brand. Pre-trained word embeddings trained on Wikipedia using fastText were employed, to convert words with similar meaning to have similar representation. fastText was used instead of Word2Vec so that the rare words could be represented as well.

Once words were decomposed into their vector representation, user reviews sentences were represented by taking the vector sum of all the word vectors composing the sentence, divided by the number of words. The hypothesis is also converted into a vector representation. We then take the cosine similarity of the vectorised user review sentences and the vectorised hypothesis statement.

Contextual Similarity is implemented with multiprocessing in certain steps, thus results can be generated in ~10 minutes.

#### Use of Pre-trained word embeddings


#### Contextual Similarity: Expected Output

1 Excel File, ```Finding_Context_Similarity\Similarity Table Results\Similarity Table - 'HYPOTHESIS STATEMENT'.xlsx```

## References

[Latent Dirichlet Allocation (LDA)](http://jmlr.csail.mit.edu/papers/v3/blei03a.html) <i>Blei, David M.; Ng, Andrew Y.; Jordan, Michael I (January 2003)</i>

[Hierarchical Dirichlet Processes](http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/jasa2006.pdf) <i>Yee Whye Teh, Y. W.; Jordan, M. I.; Beal, M. J.; Blei, D. M. (2006).</i>

[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) <i>P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, (2016)</i>


