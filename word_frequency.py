import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.probability import FreqDist

df = pd.read_csv('Invicta-8926OB-Unisex-Stainless-Automatic_sentiment.csv')

df['general sentiment'] = 'Positive'
df.loc[df['base sentiment'] < 3, 'general sentiment'] = 'Negative'

# Define a function to copy clean_body to a new column only if general sentiment is positive
def copy_if_positive(row):
    if row['general sentiment'] == 'Positive':
        return row['clean_body']
    else:
        return ''
# Define a function to copy clean_body to a new column only if general sentiment is negative
def copy_if_negative(row):
    if row['general sentiment'] == 'Negative':
        return row['clean_body']
    else:
        return ''

# Apply the function to create a new column called positive_clean_body
df['positive_clean_body'] = df.apply(copy_if_positive, axis=1)
df['negative_clean_body'] = df.apply(copy_if_negative, axis=1)


def positive_word_count():
    # Concatenate all entries into a single string
    all_entries = ' '.join(df['positive_clean_body'].astype(str).tolist())

    # Tokenize the concatenated string into words
    words = nltk.word_tokenize(all_entries)

    # Create a frequency distribution object
    fdist = FreqDist(words)

    # Print the 10 most common words and their frequencies
    pos_df = pd.DataFrame(fdist.most_common(100), columns=['word', 'frequency'])
    pos_df.to_csv('pos_word_freq.csv')

def negative_word_count():
    # Concatenate all entries into a single string
    all_entries = ' '.join(df['negative_clean_body'].astype(str).tolist())

    # Tokenize the concatenated string into words
    words = nltk.word_tokenize(all_entries)

    # Create a frequency distribution object
    fdist = FreqDist(words)

    # Print the 10 most common words and their frequencies
    neg_df = pd.DataFrame(fdist.most_common(100), columns=['word', 'frequency'])
    neg_df.to_csv('neg_word_freq.csv')

positive_word_count()
negative_word_count()
