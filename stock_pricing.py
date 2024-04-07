import yfinance as yf
import pandas as pd
import pandas_market_calendars as mcal

def isTradingDay(date, nyse=mcal.get_calendar('NYSE')):
    """Check if the given date is a trading day for the NYSE.

    Args:
        date (str): The date to check in 'YYYY-MM-DD' format.
        nyse (pandas_market_calendars.exchange_calendar.ExchangeCalendar): The NYSE calendar.
        
    Returns:
        bool: True if the date is a trading day, False otherwise.
    """
    # Use the date parameter to create a date range where the start and end are the same date
    schedule = nyse.schedule(start_date=date, end_date=date)
    
    # If the schedule DataFrame is empty, the date is not a trading day
    return not schedule.empty

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

def mostRecentStockDate(tweet_date, when, nyse):
    """
    Returns the most recent trading day given the conditions.
    
    Args:
        tweet_date (pandas.Timestamp): The date of the tweet.
        when (str): The condition to check.
        nyse (pandas_market_calendars.exchange_calendar.ExchangeCalendar): The NYSE calendar.
        
    Returns:
        str: The most recent trading day.
    """
    if when == 'day before':
        day = tweet_date - pd.Timedelta(days=1)
        while not isTradingDay(day.strftime('%Y-%m-%d'), nyse):
            day = day - pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')
    elif when == 'day of':
        day = tweet_date
        while not isTradingDay(day.strftime('%Y-%m-%d'), nyse):
            day = day - pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')
    elif when == 'day after':
        day = tweet_date + pd.Timedelta(days=1)
        while not isTradingDay(day.strftime('%Y-%m-%d'), nyse):
            day = day + pd.Timedelta(days=1)
        return day.strftime('%Y-%m-%d')

def add_price_data(stock_dict, tweet_df, ticker_lst):
    """Add stock price data to the tweet DataFrame.

    Args:
        stock_dict (dict): A dictionary of stock dataframes.
        tweet_df (pandas.DataFrame): The tweet DataFrame.
        ticker_lst (list): A list of stock tickers.

    Returns:
        pandas.DataFrame: The tweet DataFrame with stock price data added.
    """
    # Load the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    for ticker in ticker_lst:
        stock_df = stock_dict[ticker]
        for index, row in tweet_df.iterrows():
            # Check if the stock ticker is 'AAPL'
            if ticker == row['symbols']:
                tweet_date = row['timestamp']
                
                # Extract the date string
                day_before = mostRecentStockDate(tweet_date, 'day before', nyse)
                day_of = mostRecentStockDate(tweet_date, 'day of', nyse)
                day_after = mostRecentStockDate(tweet_date, 'day after', nyse)
                
                print(f"Day Of: {day_of}")
                
                # Filter the DataFrame based on the date string
                filtered_df_before = stock_df.loc[stock_df['Date'] == day_before]
                filtered_df_of = stock_df.loc[stock_df['Date'] == day_of]
                filtered_df_after = stock_df.loc[stock_df['Date'] == day_after]
                
                if filtered_df_before.empty:
                    price_day_before = "N/A"
                elif filtered_df_of.empty:
                    price_day_of = 'N/A'
                elif filtered_df_after.empty:
                    price_day_after = 'N/A'
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

def preprocess_nasdaq_df(size=-1):     
    """Loads the stockerbot-export.csv file and preprocesses the data.
    
    Args:
        size (int): The number of rows to load from the CSV file.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    # Load the data frame  
    df = pd.read_csv("stockerbot-export.csv", on_bad_lines='skip')
    
    # If size is provided, subset the dataframe
    if size > 0:
        df = df.sample(size)
    
    # Drop the last column
    df = df.iloc[:, :8]
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
    df = df.dropna(subset=['timestamp'])

    # Access date components for each timestamp
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['year'] = df['timestamp'].dt.year

    # Add new columns to the DataFrame
    df['Price Day Before Tweet'] = pd.NA
    df['Price Day of Tweet'] = pd.NA
    df['Price Day After Tweet'] = pd.NA

    # Get the unique stock tickers
    ticker_lst = df['symbols'].unique().tolist()

    # Create a dictionary of stock dataframes
    stock_data = create_df(ticker_lst)
    
    # Merge the stock data with the tweet data
    return add_price_data(stock_data, df, ticker_lst)