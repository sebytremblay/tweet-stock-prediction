import pandas as pd
import os

def save_tweets(tweets, ticker, save_directory="./"):
    """Save the tweets to a CSV file.

    Args:
        tweets (list): a list of twint.tweet objects.
        ticker (str): the stock ticker.
        save_directory (str): the directory where the CSV file will be saved.
    """
    # Extract the relevant information from each tweet
    cleaned_tweets = [{'id': tweet.id_str,
                    'url': tweet.url,
                    'symbol': ticker[1:],
                    'user': tweet.user.username,
                    'verifiedUser': tweet.user.blue,
                    'quotedUser': tweet.quotedTweet.user.username if tweet.quotedTweet else pd.NA,
                    'verifiedQuotedUser': tweet.quotedTweet.user.blue if tweet.quotedTweet else pd.NA,
                    'date': tweet.date,
                    'rawContent': tweet.rawContent,
                    'quotedContent': tweet.quotedTweet.rawContent if tweet.quotedTweet else pd.NA,
                    'retweetCount': tweet.retweetCount,
                    'likeCount': tweet.likeCount}
                    for tweet in tweets]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(cleaned_tweets)
    df = df.sort_values(by=['date'], ascending=False)
    save_to_csv(f"{save_directory}/{ticker}_tweets.csv", df)
    
def save_to_csv(file_path, df):
    """Save the DataFrame to a CSV file, appending to the file if it already exists.
    
    Args:
        file_path (str): the path to the CSV file.
        df (pd.DataFrame): the DataFrame to save."""
    if os.path.exists(file_path):
        # Read the existing data from the file
        existing_df = pd.read_csv(file_path)
        
        # Combine the data and remove duplicates
        combined_df = pd.concat([existing_df, df])
        final_df = combined_df.drop_duplicates()
        
        # Save the combined DataFrame back to the file
        final_df.to_csv(file_path, index=False)
    else: 
        # If the file does not exist, save the DataFrame directly
        df.to_csv(file_path, index=False)