# Tweet Stock Prediction

## Overview

This repository hosts the Tweet Stock Prediction project, which aims to predict stock prices based on the analysis of tweets from various financial influencers. The project utilizes machine learning techniques with Python and the `sklearn` library to analyze sentiment and correlate it with stock price movements.

## Features

- **Data Analysis**: Analyzes tweets for sentiment using natural language processing techniques.
- **Stock Price Prediction**: Uses machine learning models to predict stock prices based on sentiment derived from tweets, among other factors.
- **Dataset**: Uses the [Financial Tweets](https://www.kaggle.com/datasets/davidwallach/financial-tweets) dataset by David Wallach on Kaggle.
- **Models Used**:
  - Ridge Regression
  - Lasso Regression
  - Random Forest
  - Linear Regression
  
  The Random Forest model notably outperformed the others, achieving a Mean Absolute Error (MAE) of ~$0.05.

## How to Use

1. Clone the repository with:
   
   ```git clone git@github.com:sebytremblay/tweet-stock-prediction.git```   
2. Unzip the data and models zip files. Verify they are properly named `./data` and `./models`.
3. Navigate to `./main.ipynb` and click run all. By default, model re-training and data pre-processing regeneration are disabled. To enable this, set `force_data_regeneration` and `force_retrain` respectively to true.

## Requirements

- Python 3.6+
- Install requirements in `requirements.txt`
