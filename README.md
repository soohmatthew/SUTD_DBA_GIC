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

2. `REPROCESS`, in the event that you have made changes to your review corpus, you can reprocess the review corpus by changing `REPROCESS = True`.

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

### Data Collection Results

A total of 18171 reviews were scraped. 'Company' was not scraped from website, but was an important piece of information for the user, thus we input the Company column manually. After data collection, a necessary step would be to input 'Company' as an extra column within the dataset. 

Company	| Brand | Date | Name | Rating | Source | Usefulness | User Comment
--- | --- |  --- |  --- |  --- |  --- |  --- |  --- |  
49 unique companies | 61 unique brands | Date range from 2001 to 2018 | Name of the specific product scraped is indicated.| Ratings ranged from 1 to 5, the distribution can be seen below. | 3 unique sources, Amazon, Walmart, Best Buy | How useful the review is, which was not considered for the scope of this project | Actual raw text of customer's reviews.

Distribution of Reviews by Ratings

![alt text](https://github.com/soohmatthew/SUTD_DBA_GIC/blob/master/imgs/distributed_by_rank.png "Distribution by Ratings")

Distribution of Reviews by Brand (Only top 10 shown)

![alt text](https://github.com/soohmatthew/SUTD_DBA_GIC/blob/master/imgs/most_reviewed.png "Distribution by Brand")

Distribution of Reviews by Source

![alt text](https://github.com/soohmatthew/SUTD_DBA_GIC/blob/master/imgs/distribution_by_source.png "Distribution by Source")

Distribution of Reviews by Quarter (Quarter is extracted out from date, and not scraped)

![alt text](https://github.com/soohmatthew/SUTD_DBA_GIC/blob/master/imgs/distributed_by_quarter.png "Distribution by Quarter")

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

#### Topic Modelling: Results



#### Topic Modelling: Expected Output

2 Excel Files, ```Topic_Modelling/Topic Model Results/LDA Topic Model by Quarter by Brand.xlsx``` and ```Topic_Modelling/Topic Model Results/HDP Topic Model by Quarter by Brand.xlsx```.

### 3. Contextual Similarity

The main script that triggers the Contextual Similarity process is ```find_contextual_similarity_main.py```. Essentially, the algorithm takes in a hypothesis as an input, and compares the hypothesis with the user reviews, and returns similarity score of the hypothesis and the user review. Scores are then aggregated and grouped based on the Quarter and the Brand. Pre-trained word embeddings trained on Wikipedia using fastText were employed, to convert words with similar meaning to have similar representation. fastText was used instead of Word2Vec so that the rare words could be represented as well.

We first try to find the best representation for the hypothesis in question. This is done by phrase extraction, which we used regular expression, as well as ```textacy```, to determine if the hypothesis is a noun phrase, verb phrase or prepositional phrase (there are other types of phrases, but we limit it to these 3 for the scope of this project). Usually we hope to see only one type of phrase in the hypothesis, but in the event that there are multiple, we pick the type of phrase based on how many individual phrases are being extracted. 

<b>Example of Phrase Extraction</b>

Review | Noun Phrases | Verb Phrases | Prepositional Phrases
------ | ----- | ----- | -----
Don't like the style I did not expect the coffee pod is totally different from what I expected. I prefer K cup if I could choose from the beginning. | style, coffee pod, what, cup | do not like, did not expect, expected, prefer, could choose, beginning | style, coffee pod, what, cup

The review corpus that we have scraped, will also need to be processed. For the use of word embeddings, we preprocess the text slightly differently from what we would do for topic modelling. For instance, we cannot remove all stopwords using the ```NLTK``` library, since it includes words like "not". Negation would completely change the meaning of the word, thus, we have to create our own set of stopwords to process the data. Subsequently, based on the review sentence, we extract out the relevant noun, verb and prepositional phrases as well.

In order to narrow down our search to what is most similar with the hypothesis, if the hypothesis is a noun phrase, we will use only the noun phrases of the review to compare it against the hypothesis, applying the same concept to verb phrases and prepositional phrases. To derive a vector representation of each phrase, the phrase is broken up into words. We then leverage on fastText's pre-trained word vector, to calculate a vector representation of the phrase by taking an average of the word vectors comprising the phrase.

<i> A concise representation of the idea behind word embedding, and how words can be expressed as vectors </i>
![alt text](https://github.com/soohmatthew/SUTD_DBA_GIC/blob/master/imgs/Word-Vectors.png "Word Vectors")
<i>Credits: [An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec](https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/)</i>

Once words were decomposed into their vector representation, the hypothesis is also converted into a vector representation. We then iterate through every single possible combination of review phrases and hypothesis phrases, taking the cosine similarity of the vectors. With this list, we are able to get the maximum cosine similarity between the review and the hypothesis. This will be the score that we use to judge the similarity between review and hypothesis.

Contextual Similarity is implemented with multiprocessing in certain steps, thus results can be generated in ~10 minutes.

#### Contextual Similarity: Results

As this is an unsupervised learning problem, we do not have any metrics to determine the accuracy.The current algorithm does a good job extracting out reviews that are similar to the hypothesis, but can give an inaccurate (higher than the average) similarity score for topics that are irrelevant. Average score of a similar review is around 0.8, while average score of an irrelevant review is around 0.35.  

<i> Example: </i>

Review | Similarity Score | Hypothesis | Comment
----- | ----- | ----- | ----
Makes a horrible odor when brewing!I have made a couple dozen pots of coffee in hopes of the odor going away or diminishing but it has not. The smell seems to be related to the plastic the appliance is made of. I would have returned it but the cost to ship it back/ the hassle of getting a refund isn't worth it. I hope this review will keep others from the same bad expericence as I. |0.845106483 | refund | Positive example of the review being similar to the hypothesis.
Defective Product and Poor Customer Service We heard great things about these machines, and decided to purchase one via Amazon after reading good reviews. To provide context, we live overseas in an official government capacity and have a similar status to military members to included a US mailing and shipping address. Our apartment uses standard US wattage. This machine stopped functioning properly within the first month of use, just outside of Amazon's return policy. We did not submit it to heavy usage; on average we would use it once a day. It would make exploding noises and coffee grinds would end up in every cup of coffee. The quantity that it was dispensing was also off every time, despite our attempts to reset it. Upon calling the Nespresso customer service line, they informed us that since we lived abroad, the machine was no longer covered under warranty. After much persuasion, we were able to talk to a manager, who merely suggested we decalcify the machine. We did this, but the problems still persist. If we were to return the product, we would have to pay to have it repaired and for shipping, despite only owning it for a few months. We are very disappointed in the product and in the customer service we received from Nespresso. | 0.569656551 | refund | Positive example of the review having a similar sentiment to the hypothesis.
Love the machine, could leave the frother I'm obsessed with this. I make at least two lattes a day. I only give it 4 stars because the milk frother is a waste of time and money for me. It doesn't get as hot as I would like so what I do is microwave milk first and then put it in the frother to get a bit warmer and of course, frothy. But if you froth it for more than a few seconds it's too frothy which makes it more of a cappuccino than a latte (from my understanding of what the drinks actually are) which means microwaving the milk first is even more necessary. So, if I could do it over again, I would purchase the machine minus the frother and just buy one of those cheap hand frothers to add just a bit of froth. That being said, if you love very frothy drinks this is perfect for you. Froth. | 0.510653079 | refund | Negative example of an irrelevant review, which should receive a much lower score, however, it got a higher than average score.
Plain but niceNo complaints....makes good coffee./ dependable and uncomplicated.Very happy with this product! | 0.387510419 | refund | Positive example of an irrelevant review which got a low score.

User must be aware that in crafting the hypothesis, he must not to include any irrelevant words that may throw off the algorithm. E.g. Using ```"coffee machine refunds"``` would cause the algorithm to search for reviews containing phrases relevant to coffee machines, which is more or less every review, results would not be accurate. A more useful hypothesis would be ```"refunds, returns"```, as it captures reviews that are more similar in sentiment to what he is searching for.

#### Contextual Similarity: Expected Output

1 Excel File, ```Finding_Context_Similarity\Similarity Table Results\Similarity Table - 'HYPOTHESIS STATEMENT'.xlsx```

## References

[Latent Dirichlet Allocation (LDA)](http://jmlr.csail.mit.edu/papers/v3/blei03a.html) <i>Blei, David M.; Ng, Andrew Y.; Jordan, Michael I (January 2003)</i>

[Hierarchical Dirichlet Processes](http://www.gatsby.ucl.ac.uk/~ywteh/research/npbayes/jasa2006.pdf) <i>Yee Whye Teh, Y. W.; Jordan, M. I.; Beal, M. J.; Blei, D. M. (2006).</i>

[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) <i>P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, (2016)</i>


