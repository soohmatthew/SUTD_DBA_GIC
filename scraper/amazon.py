#Standard library imports
import pickle

#Third party imports
from lxml import html  
import requests
import pandas as pd

# Each Amazon product is identified by a unique ASIN identifier. 
# amazon_get_asin pulls all the available ASINs from the first page of the search query on the Amazon website.
def amazon_get_asin(keyword):
    amazon_url = "https://www.amazon.com/s/ref=nb_sb_noss_2?url=search-alias%3Daps&field-keywords={}".format(keyword)
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
    headers = {'User-Agent': user_agent}
    page = requests.get(amazon_url, headers = headers)
    parser = html.fromstring(page.content)

    asin_list = '//div[@id = "resultsCol"]//div//div//ul//li'
    l = parser.xpath(asin_list)

    list_of_asin = []
    for i in l:
        asin = i.get('data-asin')
        if asin is not None:
            list_of_asin.append(asin)
    
    #Check if the page exist
    page_does_not_exist = []
    for ASIN in list_of_asin:
        amazon_url_product_info = "https://www.amazon.com/dp/{}".format(ASIN)
        headers = {'User-Agent': user_agent}
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

# For each product review page, get the maximum page number of reviews, so that the web scrapper is able to scrap all the possible reviews from the product.
def amazon_get_max_page_num(ASIN):

    amazon_url = 'https://www.amazon.com/product-reviews//{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber=1&sortBy=recent'.format(ASIN)
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'
    headers = {'User-Agent': user_agent}
    page = requests.get(amazon_url, headers = headers)
    parser = html.fromstring(page.content)
    xpath_page_number = '//div[@id = "cm_cr-pagination_bar"]//li[@class="page-button"]//text()'
    page_numbers = parser.xpath(xpath_page_number)

    page_numbers = [int(page_num) for page_num in page_numbers]
    last_page_num = max(page_numbers, default = 5)

    return last_page_num

# amazon_review_scraper scraps for reviews, based on the ASIN given, and the number of pages it is going to scrape.
def amazon_review_scraper(ASIN, number_of_pages):
    
    reviews_df = pd.DataFrame()
    
    for page_num in range(1, number_of_pages+1):
        try:
            page_num = str(page_num)
            amazon_url_product_info = "https://www.amazon.com/dp/{}".format(ASIN)
            amazon_url = 'https://www.amazon.com/product-reviews//{}/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber={}&sortBy=recent'.format(ASIN, page_num)

            user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36'
            headers = {'User-Agent': user_agent}
            
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

            xpath_reviews = '//div[@data-hook="review"]'
            reviews = parser.xpath(xpath_reviews)

            xpath_rating  = './/i[@data-hook="review-star-rating"]//text()' 
            xpath_title   = './/a[@data-hook="review-title"]//text()'
            xpath_date    = './/span[@data-hook="review-date"]//text()'
            xpath_body    = './/span[@data-hook="review-body"]//text()'

            for review in reviews:
                rating  = review.xpath(xpath_rating)
                title   = review.xpath(xpath_title)
                date    = review.xpath(xpath_date)
                body    = review.xpath(xpath_body)

                date_processed = date[0].replace("on ", "")
                rating_processed = float(rating[0].replace(" out of 5 stars", ""))
                
                review_dict = {'Name': product_name_full,
                            'Rating': rating_processed,
                            'User Comment': title[0] + " " + body[0],
                            'Date': date_processed,
                            'Brand': product_brand[0],
                            'Source': "Amazon"}
                reviews_df = reviews_df.append(review_dict, ignore_index=True)
        except:
            continue
    return reviews_df

# Calls the amazon_review_scraper, and iterates through all available ASINs, returning a Pandas DataFrame
def amazon_scrape_to_df(keyword):
    list_of_asin = amazon_get_asin(keyword)
    output_df = pd.DataFrame()

    for asin in list_of_asin:
        print("Scraping from... https://www.amazon.com/dp/{}".format(asin))
        max_page_num = amazon_get_max_page_num(asin)
        print('{} pages found...'.format(max_page_num))
        reviews_df = amazon_review_scraper(asin, max_page_num)
        output_df = output_df.append(reviews_df, ignore_index = True)
        
        with open('pickle_files/amazon_web_scrape.pickle', 'wb') as handle:
            pickle.dump(output_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_df