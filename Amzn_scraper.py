from bs4 import BeautifulSoup
import requests
import pandas as pd
import secret
review_list = []

# Connects to splash to render website as javascript
def get_soup(url):
    r = requests.get(secret.splash, params= {'url' : url, 'wait' : 2})
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup
 
# Initialises a dictionary for all relevant attributes of a review
# Adds desired data to review_list
def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    try: 
        for item in reviews:
            review = {
            'product': soup.title.text.replace('Amazon.co.uk:Customer reviews:', '').strip(),
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            'rating': float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            review_list.append(review)
    except:
        pass

# Loops through each review in the set range and adds data to the review_list.
# Contains Amazon product example
for x in range(1,100):
    soup = get_soup(f'https://www.amazon.co.uk/Invicta-8926OB-Unisex-Stainless-Automatic/product-reviews/B000JQFX1G/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    print(f'Getting page {x}.')
    get_reviews(soup)
    print(len(review_list))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break

# Convertrs review_list into an excel file ready for data cleaning
df = pd.DataFrame(review_list)
df.to_excel('test.xlsx', index = False)
print('Completed.')