#Standard library imports
import os
import pickle

#Third party library imports
import pandas as pd

#Python File Imports
from Scrapers.amazon import amazon_scrape_to_df
from Scrapers.walmart import walmart_scrape_to_df
from Scrapers.bestbuy import bestbuy_scrape_to_df

# CONFIG
SEARCH_TERM = "coffee machine"

# Triggers the amazon, bestbuy and walmart webscrapers
def main(SEARCH_TERM):
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")

    amazon_df = amazon_scrape_to_df(SEARCH_TERM)
    walmart_df = walmart_scrape_to_df(SEARCH_TERM)
    bestbuy_df = bestbuy_scrape_to_df(SEARCH_TERM)

    frames = [amazon_df, walmart_df, bestbuy_df]
    result = pd.concat(frames)

    writer = pd.ExcelWriter('Customer Reviews of {}.xlsx'.format(SEARCH_TERM))
    result.to_excel(writer,'{}'.format(SEARCH_TERM))
    writer.save()
    writer.close()
    return

# In the event that pickle files were generated from the individual webscrapers, 
# and you just want to combine the various pickle files to 1 document.
def using_cached_data(SEARCH_TERM):
    with open(r'pickle_files\amazon_web_scrape.pickle', 'rb') as handle_1:
        amazon_df = pickle.load(handle_1)
    with open(r'pickle_files\bestbuy_web_scrape.pickle', 'rb') as handle_2:
        bestbuy_df = pickle.load(handle_2)
    with open(r'pickle_files\walmart_web_scrape.pickle', 'rb') as handle_3:
        walmart_df = pickle.load(handle_3)

    frames = [amazon_df, walmart_df, bestbuy_df]
    result = pd.concat(frames)
    writer = pd.ExcelWriter('Review Corpus/Customer Reviews of {}.xlsx'.format(SEARCH_TERM))
    result.to_excel(writer,'{}'.format(SEARCH_TERM))
    writer.save()
    writer.close()
    return

if __name__ == '__main__':
    main(SEARCH_TERM)