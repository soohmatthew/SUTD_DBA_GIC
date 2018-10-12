#Standard library imports
import os

#Python File Imports
from scraper.amazon import *
from scraper.walmart import *
from scraper.bestbuy import *

SEARCH_TERM = "coffee machine"

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

    return

if __name__ == '__main__':
    main(SEARCH_TERM)

