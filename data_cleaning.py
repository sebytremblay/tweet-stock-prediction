import string
import re
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import hstack

def preprocess_tweet(tweet, lemmatizer=WordNetLemmatizer()):
    """Apply NLP preprocessing to a given tweet, preserving stock tickers and URLs.

    Args:
        tweet (str): The tweet to be preprocessed.
        lemmatizer (WordNetLemmatizer): The lemmatizer to be used.
    """
    # Define patterns for stock tickers and URLs
    ticker_pattern = r'\$\w+'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    mention_pattern = r'@\w+'
    
    # Find all stock tickers and URLs
    tickers = re.findall(ticker_pattern, tweet)
    urls = re.findall(url_pattern, tweet)
    mentions = re.findall(mention_pattern, tweet)
    
    # Replace tickers, URLs and mentions with placeholders and keep a map for restoration
    ticker_placeholder_map = {}
    url_placeholder_map = {}
    mentions_placeholder_map = {}
    for i, ticker in enumerate(tickers):
        placeholder = f"__ticker{i}__"
        ticker_placeholder_map[placeholder] = ticker
        tweet = tweet.replace(ticker, placeholder)
    
    for i, url in enumerate(urls):
        placeholder = f"__url{i}__"
        url_placeholder_map[placeholder] = url
        tweet = tweet.replace(url, placeholder)
        
    for i, mention in enumerate(mentions):
        placeholder = f"__mention{i}__"
        mentions_placeholder_map[placeholder] = mention
        tweet = tweet.replace(mention, placeholder)
        
    # Convert to lower case
    tweet = tweet.lower()

    # Tokenize tweet
    tweet_tokens = word_tokenize(tweet)
    
    # Remove stopwords
    tweet_tokens = [word for word in tweet_tokens if word not in stopwords.words('english')]
    
    # Remove punctuation from tokens that are not placeholders
    extended_punctuation = string.punctuation + '“”‘’—'
    tweet_tokens = [word for word in tweet_tokens 
                    if word in ticker_placeholder_map 
                        or word in url_placeholder_map 
                        or word in mentions_placeholder_map
                        or all(char not in extended_punctuation for char in word)]
    
    # Perform lemmatization on tokens that are not placeholders
    tweet_tokens = [lemmatizer.lemmatize(word) 
                        if word not in ticker_placeholder_map 
                            and word not in url_placeholder_map 
                            and word not in mentions_placeholder_map
                        else word for word in tweet_tokens]

    # Restore stock tickers and URLs from placeholders
    final_tokens = []
    for token in tweet_tokens:
        if token in ticker_placeholder_map:
            final_tokens.append(ticker_placeholder_map[token])
        elif token in url_placeholder_map:
            final_tokens.append(url_placeholder_map[token])
        elif token in mentions_placeholder_map:
            final_tokens.append(mentions_placeholder_map[token])
        else:
            final_tokens.append(token)

    return final_tokens

def prepare_features(df, text_column, categorical_columns, numeric_columns, target_column, vectorizer, encoder, scaler):
    """Prepare the features for training and evaluation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The names of the text column.
        categorical_columns (list): The names of the categorical columns.
        numeric_columns (list): The names of the numeric columns.
        target_column (str): The name of the target column.
        vectorizer (TfidfVectorizer): The fitted text vectorizer.
        encoder (OneHotEncoder): The fitted categorical encoder.
        scaler (StandardScaler): The fitted numeric scaler.
    
    Returns:
        X: The input features.
        y: The target variable.
    """
    # Vectorize the text column
    text_features = vectorizer.fit_transform(df[text_column].astype('U'))
    
    # Encode the categorical columns
    categorical_features = encoder.fit_transform(df[categorical_columns])
    
    # Scale the numeric columns
    numeric_features = scaler.fit_transform(df[numeric_columns])
    
    # Concatenate the features
    X = hstack([text_features, categorical_features, numeric_features])
    
    # Extract the target variable
    y = df[target_column].values
    
    return X, y