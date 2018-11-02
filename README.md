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

The script is tested on Python 3.6.5 
Install the following libraries on your system, in order for the code to run. 

```
1. requests==2.20.0
2. lxml==4.2.1
3. spacy==2.0.16
4. nltk==3.3
5. fake_useragent==0.1.11
6. textblob==0.15.1
7. openpyxl==2.5.3
8. gensim==3.6.0
9. numpy==1.15.3
10. pandas==0.23.0
11. beautifulsoup4==4.6.3
12. python_dateutil==2.7.5
13. scikit_learn==0.20.0
```

Alternatively, pip install the requirements.txt

```
pip install -r requirements.txt
```

## Usage

### Configuration

#### ```scrape_main.py```

Configure the ```SEARCH TERM``` in ```scrape_main.py```, according to whatever product you wish to scrape reviews for.

#### ```topic_model_main.py```

Configure the ```SEARCH_TERM``` in ```topic_model_main.py``` accordingly. 
Configure the ```PATH_TO_REVIEW_DOC``` in ```topic_model_main.py```, where the results of the webscraping is stored.

<i>This is a temporary measure, as the final data pipeline has not been built yet.</i>

Configure the ```LIST_OF_WORDS_TO_EXCLUDE```, to remove any words that may not be useful during the topic modelling process.
Configure the ```LIST_OF_COMMON_WORDS```, which is an extension of ```LIST_OF_WORDS_TO_EXCLUDE```, just that it also includes the synonym of the words.
Configure the ```NUMBER_OF_TOPICS_RANGE```, a range of number of topics in which the algorithm will search over, to generate the most suitable number of topic for the set of documents.

#### Dealing with pickle files

For this project, both webscraping and topic model generation take an extended period of time. In order to avoid any accidental loss of intermediate data, pickle files are used for caching purposes.

If you can delete the pickle file once the python script has finished running, or you could add the following code at the end of the ``` if __name__ == "__main__" ``` statement in both ```scrape_main.py``` and ```topic_model_main.py```. 
```
currentdir = os.getcwd()
shutil.rmtree(currentdir + '/pickle_files')
```

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

## Future Works

1. We will be trying out sentence entailment, exploring various implementations (fasttext, word2vec).

2. We will be cleaning up the webscraping to include (hopefully) 2 more websites.

3. Visualisation
