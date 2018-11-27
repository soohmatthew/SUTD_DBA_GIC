#Standard library imports
import pickle
import sys

#Third party imports
from lxml import html  
import requests
import pandas as pd
from fake_useragent import UserAgent
from multiprocessing import Pool, cpu_count, Manager

"""
amazon.py scrapes product information as well as product reviews relevant to the "keyword" that the user specifies.

The main function of interest is "amazon_scrape_to_df_multithreading" (with multiprocessing) and "amazon_scrape_to_df" (without multiprocessing), which will allow for the data scraped to be returned as a pandas DataFrame.

Multiprocessing is implemented for amazon.py.
"""


def amazon_get_asin(keyword, user_agent_str):
    """
    Function:
    ---------

        (1) amazon_get_asin pulls all the available ASINs from the first page of the search query on the Amazon website.

    Args:
    -----
        (1) keyword (str): Search term defined by the user

        (2) user_agent_str (str): Random User Agent String

    Returns:
    --------
        (1) list_of_asin (list): List of ASINs (str)
    """
    # amazon_url is the URL that shows us the search result page for the keyword  
    amazon_url = "https://www.amazon.com/s/ref=nb_sb_noss_2?url=search-alias%3Daps&field-keywords={}".format(keyword)
    headers = {'User-Agent': user_agent_str}
    page = requests.get(amazon_url, headers = headers)
    parser = html.fromstring(page.content)

    asin_list = '//div[@id = "resultsCol"]//div//div//ul//li'
    l = parser.xpath(asin_list)

    list_of_asin = []
    for i in l:
        asin = i.get('data-asin')
        if asin is not None:
            list_of_asin.append(asin)
    
    #Check if the page exist, if it does not exists, it will contain the text:
    # "Sorry! We couldn't find that page. Try searching or go to Amazon's home page."
    page_does_not_exist = []
    for ASIN in list_of_asin:
        amazon_url_product_info = "https://www.amazon.com/dp/{}".format(ASIN)
        headers = {'User-Agent': user_agent_str}
        page_product_info = requests.get(amazon_url_product_info, headers = headers)
        parser_product_info = html.fromstring(page_product_info.content)
        
        page_not_found_content = '//div[@id = "g"]//div//a//img'
        l = parser_product_info.xpath(page_not_found_content)
        for i in l:
            page_not_found = i.get('alt')
            if page_not_found == "Sorry! We couldn't find that page. Try searching or go to Amazon's home page.":
                page_does_not_exist.append(ASIN)      
    for ASIN in set(page_does_not_exist):
        list_of_asin.remove(ASIN)
    return list_of_asin

#
def amazon_get_max_page_num(ASIN, user_agent_str):
    """
    Function:
    ---------

        (1)  For each product review page, amazon_get_max_page_num gets the maximum page number of reviews, so that the web scrapper is able to scrap all the possible reviews from the product.
    
    Args:
    -----
        (1) ASIN (str): Unique identifier for Amazon Product 

        (2) user_agent_str (str): Random User Agent String

    Returns:
    --------
        (1) last_page_num (int): Last page number for specific product's product review
    """

    amazon_url = 'https://www.amazon.com/product-reviews//{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=1&sortBy=recent'.format(ASIN)
    headers = {'User-Agent': user_agent_str}
    page = requests.get(amazon_url, headers = headers)
    parser = html.fromstring(page.content)
    xpath_page_number = '//div[@id = "cm_cr-pagination_bar"]//ul//li[@class="page-button"]//text()'
    page_numbers = parser.xpath(xpath_page_number)

    page_numbers = [int(page_num) for page_num in page_numbers]
    if len(page_numbers) == 0:
        last_page_num = 1
    else:
        last_page_num = max(page_numbers)
    return last_page_num

# amazon_review_scraper scraps for reviews, based on the ASIN given, and the number of pages it is going to scrape.
def amazon_review_scraper(ASIN, number_of_pages, user_agent_str):
    """
    Function:
    ---------

        (1) amazon_review_scraper pulls the following details from a specific product's page:
        
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source

    Args:
    -----
        (1) ASIN (str): Unique identifier for Amazon Product 

        (2) number_of_pages (int): Number of pages to iterate over

        (3) user_agent_str (str): Random User Agent String

    Returns:
    --------
        (1) reviews_df (pandas DataFrame): A pandas DataFrame of the scraped data, with the following columns:
        
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source
    """
    # Initialise the DataFrame we are going to return
    reviews_df = pd.DataFrame()
    
    # Iterate though maximum page numbers
    for page_num in range(1, number_of_pages+1):
        try:
            # Progress bar to view progress
            def progress(count, total, status=''):
                bar_len = 60
                filled_len = int(round(bar_len * count / float(total)))

                percents = round(100.0 * count / float(total), 1)
                bar = '=' * (filled_len - 1) + '>' + '-' * (bar_len - filled_len)

                sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
                sys.stdout.flush()
            # Outputs a nice progress bar
            progress(page_num, number_of_pages, status='Scraping {}/{}, {} reviews scraped in total.'.format(str(page_num), str(number_of_pages), str(reviews_df.shape[0])))

            page_num = str(page_num)
            amazon_url_product_info = "https://www.amazon.com/dp/{}".format(ASIN)
            amazon_url = 'https://www.amazon.com/product-reviews/{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={}&sortBy=recent'.format(ASIN, page_num)
            headers = {'User-Agent': user_agent_str}
            
            page_product_info = requests.get(amazon_url_product_info, headers = headers)
            parser_product_info = html.fromstring(page_product_info.content)
            xpath_product_info = '//div[@id = "centerCol"]'
            product_info = parser_product_info.xpath(xpath_product_info)

            for element in product_info:
                xpath_product_name  = './/div//div//h1//span[@id="productTitle"]//text()'
                xpath_product_brand = './/div//div//a[@id= "bylineInfo"]//text()'
                product_name = element.xpath(xpath_product_name)
                product_brand = element.xpath(xpath_product_brand)
            
            product_name_full = product_name[0].replace('\n', '')
            product_name_full = product_name_full.lstrip()
            product_name_full = product_name_full.rstrip()

            # In case product name gets cut off accidentally
            if len(product_name_full) <= 2:
                product_name_full = product_name
            
            page = requests.get(amazon_url, headers = headers)

            parser = html.fromstring(page.content)
            
            # xpaths for the various information that we require
            xpath_reviews = '//div[@data-hook="review"]'
            reviews = parser.xpath(xpath_reviews)

            xpath_rating  = './/i[@data-hook="review-star-rating"]//text()' 
            xpath_title   = './/a[@data-hook="review-title"]//text()'
            xpath_date    = './/span[@data-hook="review-date"]//text()'
            xpath_body    = './/span[@data-hook="review-body"]//text()'
            xpath_usefulness = './/span[@class = "a-size-base a-color-tertiary cr-vote-text cr-vote-error cr-vote-component"]//text()'

            for review in reviews:
                rating  = review.xpath(xpath_rating)
                title   = review.xpath(xpath_title)
                date    = review.xpath(xpath_date)
                body    = review.xpath(xpath_body)

                usefulness = review.xpath(xpath_usefulness)
                if usefulness == []:
                    usefulness = ["0 people found this helpful"]

                usefulness_processed = usefulness[0].replace(" people found this helpful", "")
                usefulness_processed = usefulness_processed.replace(" person found this helpful", "")
                usefulness_processed = usefulness_processed.replace(",","")
                usefulness_processed = usefulness_processed.replace("One","1")
                usefulness_processed = int(usefulness_processed)

                date_processed = date[0].replace("on ", "")
                rating_processed = float(rating[0].replace(" out of 5 stars", ""))
                
                # Consolidate information into a pandas DataFrame
                review_dict = {'Name': product_name_full,
                            'Rating': rating_processed,
                            'User Comment': title[0] + " " + body[0],
                            'Date': date_processed,
                            'Brand': product_brand[0],
                            'Usefulness': usefulness_processed,
                            'Source': "Amazon"}
                reviews_df = reviews_df.append(review_dict, ignore_index=True)
        except:
            continue
    return reviews_df

# 
def amazon_scrape_to_df(keyword):
    """
    Function:
    ---------

        (1) amazon_scrape_to_df calls the amazon_review_scraper, and iterates through all available ASINs, returning a Pandas DataFrame

        (2) Output DataFrame is also saved as a pickle file, for caching purposes.

        (3) NO MULTIPROCESSING

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        output_df (pandas DataFrame): pandas DataFrame with the following columns:
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source
    """
    # Fake User Agent library is used, so that the User Agent is randomized, so as to be able to circumvent IP bans. 
    # It will make the code run slightly slower, but we are able to yield better results.
    ua = UserAgent(verify_ssl=False)
    list_of_asin = amazon_get_asin(keyword, ua.random)
    print("{} products found... ".format(str(len(list_of_asin))))
    output_df = pd.DataFrame()

    # Iterate through every possible ASIN
    for asin in list_of_asin:
        print("Scaping {} of {} products.".format(str(list_of_asin.index(asin)), str(len(list_of_asin))))
        print("Scraping from... https://www.amazon.com/dp/{}".format(asin))
        max_page_num = amazon_get_max_page_num(asin, ua.random)
        print('{} pages found...'.format(max_page_num))
        reviews_df = amazon_review_scraper(asin, max_page_num, ua.random)
        output_df = output_df.append(reviews_df, ignore_index = True)
    # Save results in pickle file
    with open('pickle_files/amazon_web_scrape.pickle', 'wb') as handle:
        pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_df

# Function that multithreading function will call.
def amazon_df_one_asin(asin, ua):
    print("Scraping from... https://www.amazon.com/dp/{}".format(asin))
    max_page_num = amazon_get_max_page_num(asin, ua)
    print('{} pages found...'.format(max_page_num))
    reviews_df = amazon_review_scraper(asin, max_page_num, ua)
    return reviews_df

def amazon_scrape_to_df_multithreading(keyword):
    """
    Function:
    ---------

        (1) amazon_scrape_to_df calls the amazon_review_scraper, and iterates through all available ASINs, returning a Pandas DataFrame

        (2) Output DataFrame is also saved as a pickle file, for caching purposes.

        (3) WITH MULTIPROCESSING

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        output_df (pandas DataFrame): pandas DataFrame with the following columns:
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source
    """
    # Fake User Agent library is used, so that the User Agent is randomized, so as to be able to circumvent IP bans. 
    # It will make the code run slightly slower, but we are able to yield better results.
    ua = UserAgent(cache=False, verify_ssl=False)
    
    list_of_asin = amazon_get_asin(keyword, ua.random)

    print("{} products found... ".format(str(len(list_of_asin))))
    list_of_asin_and_ua = [(asin, ua.random) for asin in list_of_asin]

    output_df = Manager().list()

    with Pool(processes= cpu_count() * 2) as pool:
            review_df = pool.starmap(amazon_df_one_asin, list_of_asin_and_ua)
    
    output_df = output_df.append(review_df)
    pool.terminate()
    pool.join()
    
    output_df = pd.concat(output_df, ignore_index = True)

    with open('pickle_files/amazon_web_scrape.pickle', 'wb') as handle:
        pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_df