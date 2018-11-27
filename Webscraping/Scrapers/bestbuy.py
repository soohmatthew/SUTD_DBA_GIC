# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 01:11:55 2018

@author: Sp_ceinvader
"""

#Standard library imports
from urllib.request import urlopen as uReq
import pickle
import os
import datetime

#Third party imports
import pandas as pd
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup as soup

"""
bestbuy.py scrapes product information as well as product reviews relevant to the "keyword" that the user specifies.

The main function of interest is "bestbuy_scrape_to_df_multiprocessing" (with multiprocessing) and "bestbuy_scrape_to_df" (without multiprocessing), which will allow for the data scraped to be returned as a pandas DataFrame.

Multiprocessing is implemented for bestbuy.py.
"""

def bestbuy_search(keyword):
    """
    Function:
    ---------

        (1) Generates the url for the given product

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        search_pages (list): List of URLs

    """
    product = keyword
    product = product.split()
    search_inp = ''
    for i in range(len(product)-1):
        search_inp += str(product[i])
        search_inp += '%20'
    search_inp += str(product[-1])
    x = 1
    search_pages = []
    while x <= 2: #change this number based on how many pages of products to scrape
        link = 'https://www.bestbuy.com/site/searchpage.jsp?cp=' + str(x) + '&intl=nosplash&st='  + search_inp 
        search_pages.append(link)
        x += 1
    return search_pages

def bestbuy_get_id(keyword):
    """
    Function:
    ---------

        (1) bestbuy_get_id generates the product name/product id for URL

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        product_url (list): List of URLs
    """
    search_url = bestbuy_search(keyword)
    id_url = []
    for i in range(len(search_url)):
        uClient = uReq(search_url[i])
        page_html = uClient.read()
        uClient.close()
    
        page_soup = soup(page_html, 'html.parser')

        id_url_raw = page_soup.findAll("h4",{"class":"sku-header"})
        for j in range(len(id_url_raw)):
            s = id_url_raw[j].a.get('href').split('/')
            name = s[-2]
            ID = s[-1].split('.')[0]
            id_url.append(name + '/' + ID)
    return id_url

def bestbuy_product_review_url(keyword):
    """
    Function:
    ---------

        (1) bestbuy_product_review_url generates the product review url

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        product_url (list): List of product review URLs
    """
    product_id = bestbuy_get_id(keyword)
    product_url = []
    for i in range(len(product_id)):
        x = 1
        while x <= 20: #change this number based on how many pages of reviews to scrape
            urlWithPages = 'https://www.bestbuy.com/site/reviews/' + str(product_id[i]) + '?page=' + str(x)
            product_url.append(urlWithPages)
            x += 1
    return product_url

def bestbuy_scrape_to_df(keyword):
    """
    Function:
    ---------

        (1) bestbuy_scrape_to_df pulls the following details from a specific product's page:
        
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source

        (2) Output DataFrame is also saved as a pickle file, for caching purposes.

        (3) NO MULTIPROCESSING

    Args:
    -----
        (1) keyword (str): Search term defined by the user

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
    reviews_df = pd.DataFrame()
    my_url_all = bestbuy_product_review_url(keyword)
    
    for n in range(len(my_url_all)):
        try:
            my_url = my_url_all[n]
            print(my_url)
            uClient = uReq(my_url)
            page_html = uClient.read()
            uClient.close()

            page_soup = soup(page_html, 'html.parser')
            
            #get product name
            productName = page_soup.find('h2', {'class':"product-title"}).a.text
            print(productName)
            
            #get brand name
            brand = str(productName).split('-')[0]
            
            productcontainers = page_soup.findAll("div", {"class":"col-xs-12 col-md-9"})
            
            for productcontainer in productcontainers:
                
                #get review date
                review_date = productcontainer.findAll("time", {"class":"submission-date"})
                date = review_date[0].text
                parsed_date = [date.split()[:2]]
                time_dict = dict((fmt,int(amount)) for amount,fmt in parsed_date)
                dt = relativedelta(**time_dict)
                past_time = datetime.datetime.now() - dt
                date = past_time.strftime("%B %d, %Y")
                
                #get ratings
                reviewRating = productcontainer.findAll("p", {"class":"sr-only"})
                rating = reviewRating[0].text
                rating = rating.replace("Rating: ", "")
                rating = rating.replace(" out of 5 stars", "")
                rating = float(rating)
                
                #get review details
                reviewDetailsRaw = productcontainer.findAll("p",{"class":"pre-white-space"})
                review = reviewDetailsRaw[0].text.replace('\n\n','')

                review_dict = {'Name' : productName,
                            'Rating' : rating,
                            'User Comment' : review,
                            'Date' : date,
                            'Brand' : brand,
                            # Unable to get "Usefulness" label, set to 0
                            'Usefulness': 0,
                            'Source' : "Best Buy"}
                
                reviews_df = reviews_df.append(review_dict, ignore_index=True)
        except:
            continue    
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    with open('pickle_files/bestbuy_web_scrape.pickle', 'wb') as handle:
        pickle.dump(reviews_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return reviews_df       

# Function that multithreading function will call.
def bestbuy_scrape_one(my_url):
    reviews_df = pd.DataFrame()
    try:
        my_url
        print(my_url)
        uClient = uReq(my_url)
        page_html = uClient.read()
        uClient.close()

        page_soup = soup(page_html, 'html.parser')
        
        #get product name
        productName = page_soup.find('h2', {'class':"product-title"}).a.text
        print(productName)
        
        #get brand name
        brand = str(productName).split('-')[0]
        
        productcontainers = page_soup.findAll("div", {"class":"col-xs-12 col-md-9"})
        
        for productcontainer in productcontainers:
            
            #get review date
            review_date = productcontainer.findAll("time", {"class":"submission-date"})
            date = review_date[0].text
            parsed_date = [date.split()[:2]]
            time_dict = dict((fmt,int(amount)) for amount,fmt in parsed_date)
            dt = relativedelta(**time_dict)
            past_time = datetime.datetime.now() - dt
            date = past_time.strftime("%B %d, %Y")

            #get ratings
            reviewRating = productcontainer.findAll("p", {"class":"sr-only"})
            rating = reviewRating[0].text
            rating = rating.replace("Rating: ", "")
            rating = rating.replace(" out of 5 stars", "")
            rating = float(rating)
            
            #get review details
            reviewDetailsRaw = productcontainer.findAll("p",{"class":"pre-white-space"})
            review = reviewDetailsRaw[0].text.replace('\n\n','')

            review_dict = {'Name' : productName,
                        'Rating' : rating,
                        'User Comment' : review,
                        'Date' : date,
                        'Brand' : brand,
                        'Usefulness': 0,
                        'Source' : "Best Buy"}
            
            reviews_df = reviews_df.append(review_dict, ignore_index=True)
            
    except Exception as e:
        print(e)
        reviews_df = pd.DataFrame()
    print(reviews_df) 
    return reviews_df

def bestbuy_scrape_to_df_multiprocessing(keyword):
    """
    Function:
    ---------

        (1) bestbuy_scrape_to_df_multiprocessing calls the bestbuy_scrape_one to scrape product data, returning a Pandas DataFrame

        (2) Output DataFrame is also saved as a pickle file, for caching purposes.

        (3) WITH MULTIPROCESSING

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        final_output (pandas DataFrame): pandas DataFrame with the following columns:
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source
    """
    my_url_all = bestbuy_product_review_url(keyword)
    
    from multiprocessing import Pool, cpu_count

    print("{} products found... ".format(str(len(my_url_all))))

    with Pool(processes= cpu_count() * 2) as pool:
        review_df = pool.map(bestbuy_scrape_one, my_url_all)
    
    final_output = pd.concat(review_df)
    pool.terminate()
    pool.join()
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    with open('pickle_files/bestbuy_web_scrape.pickle', 'wb') as handle:
        pickle.dump(final_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return final_output