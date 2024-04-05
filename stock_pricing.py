import yfinance as yf
import pandas as pd

def get_stock_data(ticker):
    """Gets the stock data for the given ticker.

    Args:
        ticker (str): The stock ticker to get data for.

    Returns:
        pandas.DataFrame: The stock data for the given ticker.
    """
    return yf.download(ticker)

def create_df(ticker_lst):
    """Builds a dictionary of stock dataframes for the given list of tickers.

    Args:
        ticker_lst (list): A list of stock tickers.

    Returns:
        dict: A dictionary of stock dataframes, where the keys are the stock tickers.
    """
    ticker_dfs = {}
    for ticker in ticker_lst:
        # Get stock data
        stock = yf.Ticker(ticker)
        stock_d = stock.history(period="10y")
        stock_df = stock_d.loc['2018-01-01':'2018-12-31']
        
        # Add stock data to dictionary
        ticker_dfs[ticker] = stock_df
        ticker_dfs[ticker].reset_index(inplace=True)
        #ticker_dfs[ticker]['Date'] = pd.to_datetime(ticker_dfs[ticker]['Date'], format='%a %b %d %H:%M:%S +0000 %Y')
    
    return ticker_dfs

def mostRecentStockDate(tweet_date, when):
    """
    Returns the most recent stock date
    
    """
    if when == 'day before':
        day_before = str((tweet_date - pd.Timedelta(days=1)).date())
        if day_before is None:
            day_before = str((tweet_date - pd.Timedelta(days=2)).date())
            if day_before is None:
                day_before = str((tweet_date - pd.Timedelta(days=3)).date())
        return day_before

    elif when == 'day after':
        day_after = str((tweet_date + pd.Timedelta(days=1)).date())
        if day_after is None:
            day_after = str((tweet_date + pd.Timedelta(days=3)).date())
        return day_after
    
    elif when == 'day of':
        day_of = str((tweet_date).date())
        
        return day_of

def add_price_data(stock_dict, tweet_df, ticker_lst):
    """Add stock price data to the tweet DataFrame.

    Args:
        stock_dict (dict): A dictionary of stock dataframes.
        tweet_df (pandas.DataFrame): The tweet DataFrame.
        ticker_lst (list): A list of stock tickers.

    Returns:
        pandas.DataFrame: The tweet DataFrame with stock price data added.
    """
    for ticker in ticker_lst:
        stock_df = stock_dict[ticker]
        for index, row in tweet_df.iterrows():
            # Check if the stock ticker is 'AAPL'
            if ticker == row['symbols']:
                tweet_date = row['timestamp']
                
                # Extract the date string
                day_before = mostRecentStockDate(tweet_date, 'day before')
                day_of = mostRecentStockDate(tweet_date, 'day of')
                day_after = mostRecentStockDate(tweet_date, 'day after')
                
                # Filter the DataFrame based on the date string
                filtered_df_before = stock_df.loc[stock_df['Date'] == day_before]
                filtered_df_of = stock_df.loc[stock_df['Date'] == day_of]
                filtered_df_after = stock_df.loc[stock_df['Date'] == day_after]
                
                if filtered_df_before.empty:
                    price_day_before = "N/A"
                elif filtered_df_of.empty:
                    price_day_of = 'N/A'
                elif filtered_df_after.empty:
                    price_day_afterafter = 'N/A'
                
                else:
                    # Select the 'Close' column from the filtered DataFrame
                    close_series_before = filtered_df_before['Close']
                    close_series_of = filtered_df_of['Close']
                    close_series_after = filtered_df_after['Close']
                    
                    # Convert the Series to a numpy array
                    close_array_before = close_series_before.values
                    close_array_of = close_series_of.values
                    close_array_after = close_series_after.values
                    
                    # Access the first element of the numpy array
                    price_day_before = close_array_before[0]
                    price_day_of = close_array_of[0]
                    price_day_after = close_array_after[0]
                
                #MOST RECENT DAY AFTER
                tweet_df.at[index, 'Price Day Before Tweet'] = price_day_before
                tweet_df.at[index, 'Price Day of Tweet'] = price_day_of
                tweet_df.at[index, 'Price Day After Tweet'] = price_day_after
            
    return tweet_df