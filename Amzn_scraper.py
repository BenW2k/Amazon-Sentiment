from bs4 import BeautifulSoup
import requests
import pandas as pd
import pandasql as pdsql
import secret
from textblob import Word
from nltk.corpus import stopwords
review_list = []
stop_words = stopwords.words('english')
more_stop_words = ['invicta', 'would', 'one', 'got', 'two',]

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
for x in range(1,5):
    soup = get_soup(f'https://www.amazon.co.uk/Invicta-8926OB-Unisex-Stainless-Automatic/product-reviews/B000JQFX1G/ref=cm_cr_getr_d_paging_btm_prev_1?ie=UTF8&reviewerType=all_reviews&pageNumber={x}')
    print(f'Getting page {x}.')
    get_reviews(soup)
    print(len(review_list))
    if not soup.find('li', {'class': 'a-disabled a-last'}):
        pass
    else:
        break

# Converts review_list into an excel file ready for data cleaning
df = pd.DataFrame(review_list)
# Drops any rows that have blank 'body' columns
df.dropna(subset=['body'])

def average_words(x):
    words = x.split()
    return sum(len(word) for word in words) / len(words)

# Additional dataset features that could be useful in the future
df['word_count'] = df['body'].apply(lambda x : len(x.split()))
df['char_count'] = df['body'].apply(lambda x : len(x))
df['average_word_length'] = df['body'].apply(lambda x : average_words(x))
df['stopword_count'] = df['body'].apply(lambda x : len([word for word in x.split() if word.lower() in stop_words]))
df['stopword_rate'] = df['stopword_count'] / df['word_count']

# Data cleaning functions (Keeping original data to compare performance between clean and unclean data when modelling)

# Converts all text to lower case
df['lowercase_body'] = df['body'].apply(lambda x: " ".join(word.lower() for word in x.split()))
# Removes all punctuation from text
df['punc_body'] = df['lowercase_body'].str.replace('[^\w\s]', '', regex=True)
# Removes stopwords found in stop_words from text
df['no_stopword_body'] = df['punc_body'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
# Removes additional stopwords found in more_stop_words from text
df['clean_body'] = df['no_stopword_body'].apply(lambda x: " ".join(word for word in x.split() if word not in more_stop_words))

# Lemmatization
df['lemmatized'] = df['clean_body'].apply(lambda x: "" .join(Word(word).lemmatize() for word in x.split()))


