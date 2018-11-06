# Investment Insights with NLP
 
 This is a repository for the <i>40.011 : Data and Business Analytics Project</i>, in collaboration with GIC to build a data pipeline that would gather consumer sentiment on various products. 

## Group Members:
1. Matthew Soh
2. Toni Celine Gutierrez Hilos
3. Lam Xue Wei
4. Clarence Toh Jun Wen
5. Ng Au Ker Wesson

## Project Flow

1. Data Collection: Web-scraping of popular e-commerce websites, to build a corpus of product reviews. 
2. Topic Modelling: Building a topic model, using various topic modelling algorithms, trained with the corpus we have built.

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
```

4. Please download the pre-trained english FastText Word Vector (bin + text) at https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md , save it under the Finding_Context_Similarity folder, in the format '.../Finding_Context_Similarity/wiki.en'

## Usage

### Configuration of ```main.py```

<i> For the webscraper </i>

1. `SEARCH_TERM`, according to whatever product you wish to scrape reviews for. Default is `coffee machines`

<i> For the topic model </i>

2. `PATH_TO_REVIEW_DOC`, where the results of the webscraping is stored. Advised not to change location.

3. `LIST_OF_WORDS_TO_EXCLUDE`, to remove any words that may not be useful during the topic modelling process. Default is `['one', 'two', 'three', 'four', 'five', 'star']`.

4. `LIST_OF_COMMON_WORDS`, which is an extension of ```LIST_OF_WORDS_TO_EXCLUDE```, just that it also includes the synonym of the words. Default is `["good", "great", "love"]`.

5. `NUMBER_OF_TOPICS_RANGE`, a range of number of topics in which the algorithm will search over, to generate the most suitable number of topic for the set of documents. Default is `[2,3,4,5]`

<i> For the contextual similarity model </i>

6. `HYPOTHESIS_STATEMENT`, based on what your hypothesis statement is. Default is `breakdown`.

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

An excel file labelled <i>'output corpus/Customer Reviews of ```SEARCH_TERM```.xlsx'</i> will be downloaded to your system.

### 2. Topic Modelling

The main script that triggers the Topic Modelling process is ```topic_model_main.py```. 

This script will take in the output of ```scrape_main.py```, namely ```'Review Corpus/Customer Reviews of SEARCH_TERM.xlsx'``` and generate topic models. The algorithm used to generate the topic model is the Latent Dirichlet Allocation (LDA) <i>Blei, David M.; Ng, Andrew Y.; Jordan, Michael I (January 2003)</i>.

Data was first preprocessed, to remove all stopwords, punctuations, and for words to all be lemmatized. Data is then split into different categories, by quarter, and then by brand. The algorithm will generate 2 sets of results, one with reviews categorised by quarter (in order to observe quarterly trends), and the oher one categorised by quarter, by brand (in order to observe quarterly trends, by brand).

The ```scikit learn``` library was used to generate the LDA model, and the metric used for selecting the best model is lowest perplexity. The ```scikit learn``` library was picked over the ```gensim``` library, due to the ability for the ```scikit learn``` library to apply GridSearch to find the best topic model.

Other algorithms tested were as follows, but did not yield as good results. They can be viewed under ```topic_modelling/Past Testings```.

1. Latent Semantic Analysis (LSA)

2. k-means Clustering

Topic modelling is implemented with multiprocessing, and should take no more than 1 hour.

#### Topic Modelling: Expected Output

2 Excel Files, ```Topic Model Results/LDA Topic Model by Quarter by Brand.xlsx``` and ```Topic Model Results/LDA Topic Model by Quarter.xlsx```.


### 3. Sentiment Analysis 

Currently, we are still working on building Sentiment Analysis Models, testing out various models to determine which has the greatest accuracy. We will be using the Valence Aware Dictionary and sEntiment Reasoner (VADER) library as an accuracy benchmark.

### 4. Contextual Similarity

The main script that triggers the Contextual Similarity process is ```find_contextual_similarity_main.py```. Essentially, the algorithm takes in a hypothesis as an input, and compares the hypothesis with the user reviews, and returns similarity score of the hypothesis and the user review. Scores are then aggregated and grouped based on the Quarter and the Brand. Pre-trained word embeddings trained on Wikipedia using fastText were employed, to convert words with similar meaning to have similar representation. fastText was used instead of Word2Vec so that the rare words could be represented as well.

Once words were decomposed into their vector representation, user reviews sentences were represented by taking the vector sum of all the word vectors composing the sentence, divided by the number of words. The hypothesis is also converted into a vector representation. We then take the cosine similarity of the vectorised user review sentences and the vectorised hypothesis statement.

Contextual Similarity is implemented with multiprocessing in certain steps, thus results can be generated in ~10 minutes.

#### Use of Pre-trained word embeddings
P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, [<i>Enriching Word Vectors with Subword Information</i>](https://arxiv.org/abs/1607.04606)

#### Contextual Similarity: Expected Output

1 Excel File, ```Finding_Context_Similarity\Similarity Table Results\Similarity Table - 'HYPOTHESIS STATEMENT'.xlsx```

## Future Works

1. Visualisation

