from math import sqrt
import os
import datetime as dt
import pandas as pd
from pandas_datareader import data
import snscrape.modules.twitter as sntwit
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Create a container class for functions used to transform results from saveTweets into a n x 4 array of floats representing the average sentiment scores for the query
class TwitterQuery:

    def __init__(self, data):
        self.df = pd.read_csv(data)
    
    def head(self):
        print(self.df.head(30))

    # Clean tweets to allow for effective segmentation and analysis
    def clean(self):
        #Remove Capitilization
        self.df['Tweet'] = self.df['Tweet'].str.lower()

        #Remove Usernames
        self.df['Tweet'].replace(r'@[\w]*', '', True, regex=True)
        # Tickers
        self.df['Tweet'].replace(r'\$[\w]*', '', True, regex=True)
        # Numbers,
        self.df['Tweet'].replace(r'\d*', '', True, regex=True)
        # Single Characters,
        self.df['Tweet'].replace(r'\s[\w][\W]', '', True, regex=True)
        # Long Whitespaces,
        self.df['Tweet'].replace(r'\s[\s]+', ' ', True, regex=True)
        # Punctuation,
        self.df['Tweet'].replace(r'[^\w\s]', '', True, regex=True)
        # Unicode,
        self.df['Tweet'] = self.df['Tweet'].apply(lambda x: x.encode('ascii', 'ignore').strip())
    
    # Remove stopwords to lessen time spent processing non-meaningful words
    def stopRemoval(self):
        stops = set(stopwords.words('english'))
        def stopper(tweet):
            return [word for word in tweet if word not in stops]
        
        self.df['Tweet'] = self.df['Tweet'].apply(stopper)
    
    # Perform lemmetization to reduce words of similar meaning to a common root (eg. liking, liked -> like)
    def lemmetize(self):
        def lemm(tweet):
            return [WordNetLemmatizer().lemmatize(word) for word in tweet]
        
        self.df['Tweet'] = self.df['Tweet'].apply(lemm)

    # Segment tweet by word
    def tokenize(self):
        self.df['Tweet'] = self.df['Tweet'].apply(TweetTokenizer().tokenize)
    
    def join(self):
        self.df['Tweet'] = self.df['Tweet'].str.join(" ")
    
    def score(self):
        sia = SentimentIntensityAnalyzer()

        # Compose unweighted scores of tweets and drop tweet row
        self.df['Compound Score'] = self.df['Tweet'].apply(lambda tweet: sia.polarity_scores(tweet)['compound'])
        self.df['Positive Score'] = self.df['Tweet'].apply(lambda tweet: sia.polarity_scores(tweet)['pos'])
        self.df['Negative Score'] = self.df['Tweet'].apply(lambda tweet: sia.polarity_scores(tweet)['neg'])
        self.df['Neutral Score'] = self.df['Tweet'].apply(lambda tweet: sia.polarity_scores(tweet)['neu'])
        self.df.drop(columns='Tweet', inplace=True)

        # Weight Scores with the hypotenuse of the popularity metrics, likes and retweets
        self.df['hype'] = (self.df['Like Count']*self.df['Like Count']) + (self.df['Retweet Count']*self.df['Retweet Count']).astype(float)
        self.df['hype'] = self.df['hype'].apply(sqrt)
        self.df['Compound Score'] = self.df['Compound Score'] * self.df['hype']
        self.df['Positive Score'] = self.df['Positive Score'] * self.df['hype']
        self.df['Negative Score'] = self.df['Negative Score'] * self.df['hype']
        self.df['Neutral Score'] = self.df['Neutral Score'] * self.df['hype']
        self.df.drop(columns=['Like Count', 'Retweet Count', 'hype'], inplace=True)

    # Compute average score to reduce activity noise
    def averageScores(self):
        averageCompound = sum(self.df['Compound Score'])/len(self.df.index)
        averageNeutral = sum(self.df['Neutral Score'])/len(self.df.index)
        averagePositive = sum(self.df['Positive Score'])/len(self.df.index)
        averageNegative = sum(self.df['Negative Score'])/len(self.df.index)
        scores = [averageCompound, averageNeutral, averagePositive, averageNegative]
        return scores
    
    # Return the date associated with the query
    def getTweetDate(self):
        tweetDate = dt.datetime.strptime(self.df.loc[0, 'Date Posted'], '%Y-%m-%d %H:%M:%S+00:%f')
        tweetDate = tweetDate.strftime('%Y-%m-%d')
        return tweetDate

# Save tweets mentioning ticker from each day between start_date(yyy-mm-dd) and end_date(yyy-mm-dd) in 'Tweets/[ticker]'
# Returns path to location of saved files
def saveTweets(ticker, start_date, end_date):
    tweets_container = []

    # Transform given dates into pandas daterange object to iterate over
    dateRange = pd.date_range(start=start_date, end=end_date)

    # For each day in dateRange, return a dataframe of all the tweets that day mentioning given ticker
    # Dataframe contains Date Posted, Tweet, Like Count, and Retweet Count
    for day in dateRange:
        day0 = day.strftime('%Y-%m-%d')
        day1 = (day + dt.timedelta(days=1)).strftime('%Y-%m-%d')
        # The query to search
        searchQuery = ticker + ' lang:en since:' + day0 + ' until:' + day1 + ' -filter:links -filter:replies'

        # Append each tweet from search results to empty tweet_container array
        for i,tweet in enumerate(sntwit.TwitterSearchScraper(searchQuery).get_items()):
            tweets_container.append([tweet.date, tweet.content, tweet.likeCount, tweet.retweetCount])
        
        # Dataframe from tweets_container
        tweets_df = pd.DataFrame(tweets_container, columns=['Date Posted', 'Tweet', 'Like Count', 'Retweet Count'])

        # If folder doesn't exist for results already, make one
        if not os.path.exists('Tweets/' + ticker):
            os.makedirs('Tweets/' + ticker)

        # Save results as csv file
        tweets_df.to_csv('Tweets/' + ticker +'/' + day0 + '.csv')

        # Reset dataframe and tweet_container array
        tweets_container.clear()
        tweets_df.iloc[0:0]
    
    # Return path to location of saved files
    path = 'Tweets/' + ticker
    return path

# Call functions from TwitterQuery to return sentiment scores for files in path
# Returns array of scores with the date that gave those scores
def getScoreMatrix(path):
    scoreMatrix = []
    # For each file in directory, clean and score tweets from file and append results to scoreMatrix array
    for file in os.scandir(path):
        query = TwitterQuery(path + '/' + file.name)
        query.clean()
        query.tokenize()
        query.lemmetize()
        query.stopRemoval()
        query.join()
        query.score()
        scoreVector = query.averageScores()
        scoreVector.insert(0, query.getTweetDate())
        scoreMatrix.append(scoreVector)
    
    return scoreMatrix

# Add average daily stock price to sentiment score matrix and return complete dataframe
def getDataMatrix(ticker, path):

    # Create dataframe to contain score matrix
    scoreDF = pd.DataFrame(getScoreMatrix(path), columns=['Date', 'Compound Score', 'Neutral Score', 'Positive Score', 'Negative Score'])
    # Get start and end date of scores
    startDate = scoreDF.loc[0, 'Date']
    endDate = scoreDF.loc[len(scoreDF.index)-1, 'Date']

    # Index dataframe by date column
    scoreDF.set_index('Date', drop=True, inplace=True)
    scoreDF.index = pd.to_datetime(scoreDF.index)

    # Get HLOCVA price data from yahoo finance as dataframe indexed by date
    # Compute day's averaged from high and low values and append results to dataframe
    HLOCVA_DF = data.get_data_yahoo(ticker, startDate, endDate)
    HLOCVA_DF['Average'] = (HLOCVA_DF['High'] + HLOCVA_DF['Low'])/2

    # Join average price to tweet score matrix dataframe on date index, drop values that can't be matched
    scoreDF = scoreDF.join(HLOCVA_DF['Average'])
    scoreDF.dropna('index', inplace=True)
    
    return scoreDF
