import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack

def preprocess_tweet(tweet, lemmatizer=WordNetLemmatizer(),
                     keep_urls=True, keep_mentions=True, keep_stock_tickers=True):
    """Apply NLP preprocessing to a given tweet, preserving stock tickers, URLs, and mentions.

    Args:
        tweet (str): The tweet to be preprocessed.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        keep_urls (bool): Whether to keep URLs in the tweet.
        keep_mentions (bool): Whether to keep mentions in the tweet.
        

    Returns:
        str: The preprocessed tweet.
    """
    patterns = {
        'ticker': r'\$\w+',
        'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        'mention': r'@\w+',
    }
    placeholder_map = {}

    # Replace special patterns with placeholders
    for key, pattern in patterns.items():
        for i, match in enumerate(re.findall(pattern, tweet)):
            placeholder = f"__{key}{i}__"
            placeholder_map[placeholder] = match
            tweet = tweet.replace(match, placeholder)

    # Processing steps: tokenize, lower, remove stopwords, remove punctuation, and lemmatize
    tokens = word_tokenize(tweet.lower())
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation + '“”‘’—')

    filtered_tokens = [
        lemmatizer.lemmatize(token) if token not in placeholder_map else token
        for token in tokens
        if token not in stop_words and (token in placeholder_map or not any(char in punctuation for char in token))
    ]

    # Restore placeholders with original content
    final_text = ' '.join(placeholder_map.get(token, token) for token in filtered_tokens)
    
    return final_text


def prepare_features(df, text_column, categorical_columns, numeric_columns, target_column, vectorizer, encoder, scaler, training_data=False):
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
        training_data (bool): Whether the data is for training or evaluation.
    
    Returns:
        X: The input features.
        y: The target variable.
    """
    # Fit the vectorizer, encoder, and scaler if this is training data
    if training_data:
        vectorizer.fit(df[text_column].astype('U'))
        encoder.fit(df[categorical_columns])
        scaler.fit(df[numeric_columns])

    # Transform the features
    text_features = vectorizer.transform(df[text_column].astype('U')) 
    categorical_features = encoder.transform(df[categorical_columns])
    numeric_features = scaler.transform(df[numeric_columns])
    
    # Concatenate the features
    X = hstack([text_features, categorical_features, numeric_features])
    
    # Extract the target variable
    y = df[target_column].values
    
    return X, y