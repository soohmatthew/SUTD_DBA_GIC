# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 01:11:55 2018

@author: Sp_ceinvader
"""

# =============================================================================
# Change the following parameters of variables in order to get different results (CAUTION: Larger numbers require higher processing power):
#     1. x in while loop of get_id function (change this number based on how many pages of products to scrape)
#     2. x in while loop of product_review_url function (change this number based on how many pages of reviews to scrape)
# =============================================================================

#Standard library imports
from urllib.request import urlopen as uReq
import pickle
import os

#Third party imports
import pandas as pd
from bs4 import BeautifulSoup as soup

#Function that generates the url for the given product
def bestbuy_search(keyword):
    product = keyword
    product = product.split()
    search_inp = ''
    for i in range(len(product)-1):
        search_inp += str(product[i])
        search_inp += '%20'
    search_inp += str(product[-1])
    x = 1
    search_pages = []
    while x <= 1: #change this number based on how many pages of products to scrape
        link = 'https://www.bestbuy.com/site/searchpage.jsp?cp=' + str(x) + '&intl=nosplash&st='  + search_inp 
        search_pages.append(link)
        x += 1
    return search_pages

#Function that generates the product name/product id for URL
def bestbuy_get_id(keyword):
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

#Function that generates the product review url
def bestbuy_product_review_url(keyword):
    product_id = bestbuy_get_id(keyword)
    product_url = []
    for i in range(len(product_id)):
        x = 1
        while x <= 2: #change this number based on how many pages of reviews to scrape
            urlWithPages = 'https://www.bestbuy.com/site/reviews/' + str(product_id[i]) + '?page=' + str(x)
            product_url.append(urlWithPages)
            x += 1
    return product_url

#Function that extracts the Brand, Product Name, Review Date, Review Ratings and Product Reviews in pandas and csv format
def bestbuy_scrape_to_df(keyword):
    reviews_df = pd.DataFrame()
    my_url_all = bestbuy_product_review_url(keyword)
    
    for n in range(len(my_url_all)):
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
            
            #get ratings
            reviewRating = productcontainer.findAll("p", {"class":"sr-only"})
            rating = reviewRating[0].text
            
            #get review details
            reviewDetailsRaw = productcontainer.findAll("p",{"class":"pre-white-space"})
            review = reviewDetailsRaw[0].text.replace('\n\n','')

            review_dict = {'Name' : productName,
                           'Rating' : rating,
                           'User Comment' : review,
                           'Date' : date,
                           'Brand' : brand,
                           'Source' : "Best Buy"}
            
            reviews_df = reviews_df.append(review_dict, ignore_index=True)
    if not os.path.exists("pickle_files"):
        os.mkdir("pickle_files")
    with open('pickle_files/bestbuy_web_scrape.pickle', 'wb') as handle:
        pickle.dump(reviews_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return reviews_df       