import pandas as pd  
import yfinance as yf
import stock_pricing as sp
     
# making data frame  
df = pd.read_csv("stockerbot-export.csv", on_bad_lines='skip')  
   
df = df.iloc[:, :8]
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%a %b %d %H:%M:%S +0000 %Y', errors='coerce')
df = df.dropna(subset=['timestamp'])

# Access date components for each timestamp
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['year'] = df['timestamp'].dt.year

df['Price Day Before Tweet'] = pd.NA
df['Price Day of Tweet'] = pd.NA
df['Price Day After Tweet'] = pd.NA

df.head(5)

pd.set_option('display.max_rows', None)
ticker_lst = df['symbols'].unique().tolist()
print(len(ticker_lst))

stock = yf.Ticker('M')
stock_df = stock.history(period="10y")

import stock_pricing as sp

hi = sp.create_df(ticker_lst)

result_df = sp.add_price_data(hi, df, ticker_lst)

result_df.head(20)