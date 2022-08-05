# TwitterTicks
Pulls data from twitter and yahoo finance to create a pandas dataframe of tweet sentiment scores correlated with stock price on a daily frequency.

Sample Usage
```
from TwitterQueryScoring import *

# Get tweets mentioning MSFT for a date range
tweet_folder = saveTweets('MSFT', '2019-07-01', '2022-07-31')

# Combine tweets from saveTweets with stock market price data into dataframe of scores and average price
data = getDataMatrix('MSFT', tweet_folder)
```
Sample Output
```
              Compound Score  Neutral Score  Positive Score  Negative Score  Average
Date
2022-06-01        1.483098       6.170265        1.011307        0.533926  273.865005
2022-06-02       -0.581242       7.750766        1.321250        2.563927  268.125000
2022-06-03        2.096768       8.712888        2.193828        1.443769  270.930008
2022-06-06        2.300438      13.029564        1.528589        0.183123  270.699997
2022-06-07        2.111138      10.098315        2.371916        0.812490  269.535004
2022-06-08        0.824045       8.613651        0.915860        0.443218  271.304993
```

Github: https://github.com/RillienCot/TwitterTicks
