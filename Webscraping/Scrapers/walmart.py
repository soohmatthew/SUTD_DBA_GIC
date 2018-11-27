# -*- coding: utf-8 -*-
"""

Created on Fri Sep 28 00:27:18 2018

@author: Xue Wei
"""

#Standard library imports
from urllib.request import urlopen as uReq
import pickle
import os

#Third party imports
from bs4 import BeautifulSoup as soup
import pandas as pd

"""
walmart.py scrapes product information as well as product reviews relevant to the "keyword" that the user specifies.

The main function of interest is "walmart_scrape_to_df", which will allow for the data scraped to be returned as a pandas DataFrame.

Multiprocessing is NOT implemented for walmart.py.
"""

def walmart_search(keyword):
    """
    Function:
    ---------

        (1) walmart_search gathers all the links we are able to access to get product data

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        search_pages (list): List of URLs
    """
    default_link = 'https://www.walmart.com/search/?page=1&ps=40&query=coffee+machine#searchProductResult'
    uClient = uReq(default_link)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, 'html.parser')
    
    keyword = keyword.split()
    search_inp = ''
    for i in range(len(keyword)-1):
        search_inp += str(keyword[i])
        search_inp += '+'
    search_inp += str(keyword[-1])
    x = 1    
    max_page = int(page_soup.find("div",{"class":"paginator"}).ul.li.a.get('aria-label').split()[-2])
    
    search_pages = []
    while x <= max_page: #change the number here to varies the number of pages you want to scrap
        link = 'https://www.walmart.com/search/?page=' + str(x) + '&ps=40&query='  + search_inp + '#searchProductResult'
        search_pages.append(link)
        x += 1
    return search_pages

def walmart_product_url(keyword):
    """
    Function:
    ---------

        (1) walmart_search gathers all the links we are able to access to get product data

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        search_pages (list): List of URLs
    """
    search_url = walmart_search(keyword)
    id_url = []
    #get the product id
    for i in range(len(search_url)):
        uClient = uReq(search_url[i])
        page_html = uClient.read()
        uClient.close()
    
        page_soup = soup(page_html, 'html.parser')

        id_url_raw = page_soup.findAll("a",{"class":"product-title-link line-clamp line-clamp-2"})
        for j in range(len(id_url_raw)):
            if id_url_raw[j].get('href').split('/')[-1] not in id_url:
                id_url.append(id_url_raw[j].get('href').split('/')[-1])
    #get the review page of the product using the id
    product_url = []
    for k in range(len(id_url)):
        url = 'https://www.walmart.com/reviews/product/' + str(id_url[k])
        product_url.append(url)
    return product_url

def walmart_get_brand_name(id_url_element):
    """
    Function:
    ---------

        (1) Functions get the brand name for the "id_url_element"

    Args:
    -----
        (1) id_url_element (str): Unique url identifier of product

    Returns:
    --------
        brands[0] (str): Brand Name for the "id_url_element" passed in.
    """
    walmart_product_url = "https://www.walmart.com/ip/{}".format(id_url_element)
    uClient = uReq(walmart_product_url)
    page_html = uClient.read()
    uClient.close()
    page_soup = soup(page_html, 'html.parser')
    brand = page_soup.findAll("span",{"itemprop":"brand"}, text = True)
    brands = []
    for node in brand:
        brands.append(''.join(node.findAll(text=True)))
    return brands[0]

def walmart_scrape_to_df(keyword):
    """
    Function:
    ---------

        (1) walmart_scrape_to_df iterates through all the product urls to retrive the relevant information, stores it in a pandas DataFrame, and returns the Pandas DataFrame

        (2) Output DataFrame is also saved as a pickle file, for caching purposes.

        (3) NO MULTIPROCESSING

    Args:
    -----
        (1) keyword (str): Search term defined by the user

    Returns:
    --------
        reviews_df (pandas DataFrame): pandas DataFrame with the following columns:
            (a) Name
            
            (b) Rating
            
            (c) User Comment
            
            (d) Date
            
            (e) Brand
            
            (f) Usefulness
            
            (g) Source
    """
    my_url_all = walmart_product_url(keyword)

    reviews_df = pd.DataFrame()
    for n in range(len(my_url_all)):
        try:
            my_url = my_url_all[n]
            print("Scraping from..." + my_url)
            uClient = uReq(my_url)
            page_html = uClient.read()
            uClient.close()

            page_soup = soup(page_html, 'html.parser')
        
            #get product name
            product_name = page_soup.find('div', {'class':"product-summary-wrapper"}).a.h1.div.div.text.replace(",",'')

            #get review title
            review_title_raw = page_soup.findAll("div",{"itemprop":"name"},{"class":"Grid ReviewList-content"})
            review_title = []
            for i in range(len(review_title_raw)):
                review_title.append(review_title_raw[i].text)

            #get star ratings
            star_raw = page_soup.findAll("div",{"class":"stars stars-small"})
            star = []
            for j in range(len(star_raw)):
                y = str(star_raw[j].span["alt"])[16]
                star.append(y)

            #get review details
            review_details_raw = page_soup.findAll("div",{"class":"review-body-text"})
            review_details = []
            for k in range(len(review_details_raw)):
                review_details.append(review_details_raw[k].text.replace('\n\n',''))
                
            #get date
            date_raw = page_soup.findAll('span', {'itemprop':"datePublished"})
            date = []
            for l in range(len(date_raw)):
                date.append(date_raw[l].text.replace(",",''))
            
            #get brand
            my_product_url = my_url.replace("https://www.walmart.com/reviews/product/", "")
            brand_name = walmart_get_brand_name(my_product_url)

            #formatting for excel file
            for m in range(len(star)):
                try: #some reviews have missing info which would cause error, this is to exclude such review
                    review_dict = {'Name': product_name,
                            'Rating': star[m],
                            'User Comment': str(review_title[m].replace(",","/") + review_details[m].replace('\n','').replace(',','/')),
                            'Date': str(date[m]),
                            'Brand': brand_name,
                            # Unable to get "Usefulness" label, set to 0
                            'Usefulness': 0,
                            'Source': "Walmart"}
                    reviews_df = reviews_df.append(review_dict, ignore_index=True)
                except:
                    continue
        except:
            continue
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    with open('pickle_files/walmart_web_scrape.pickle', 'wb') as handle:
        pickle.dump(reviews_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return reviews_df