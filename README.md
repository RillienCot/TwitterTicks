# TwitterTicks
Pulls data from twitter and yahoo finance to create a pandas dataframe of tweet sentiment scores correlated with stock price on a daily frequency.

Sample Usage
```
from SentAnal import *

# Get tweets mentioning MSFT for a date range
tweet_folder = saveTweets('MSFT', '2019-07-01', '2022-07-31')

# Combine tweets from saveTweets with stock market price data into dataframe of scores and average price
data = getDataMatrix('MSFT', tweet_folder)
```
