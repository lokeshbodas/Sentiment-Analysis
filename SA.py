import yfinance as yf
import plotly.graph_objs as go
import streamlit as st
from gnews import GNews
from datetime import datetime
from dateutil import tz
import numpy as np
from datetime import timedelta
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from twython import Twython
import time
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

def get_ticker_data(ticker_symbol, data_period, data_interval):
    ticker_data = yf.download(tickers=ticker_symbol,
                              period=data_period, interval=data_interval)

    if len(ticker_data) == 0:
        st.write("no issuer data found")
    else:
        ticker_data.index = ticker_data.index.strftime("%d-%m-%Y %H:%M")
    
    return ticker_data

def search_key(word, period):
    google_news = GNews(language='id', country='ID', period=period, exclude_websites=None)

    news = google_news.get_news(word+'%20')

    my_bar = st.progress(0)

    for i in range (len(news)):
        time.sleep(0.1)
        article = google_news.get_full_article(news[i]['url'])
        news[i]['description'] = article.text
        my_bar.progress(i + 1)
    return news

def convert_date(gmt_date):
    from_zone = tz.gettz('GMT')
    #to_zone = tz.gettz('US/Eastern')
    gmt = datetime.strptime(gmt_date, '%a, %d %b %Y %H:%M:%S GMT')
    gmt = gmt.replace(tzinfo=from_zone)
    gmt = gmt.strftime('%Y-%m-%d')
    
    return gmt

def format_date(df):
    issuer_date = []
    for i in range(len(df.index)):
        tgl = df.index[i].split(' ')[0].split('-')
        tgl = tgl[2] + '-' + tgl[1] + '-' + tgl[0]
        issuer_date.append(tgl)

    return issuer_date  

def plot(df, column_name1, column_name2):
    df['upper_limit'] = df[column_name1].mean()+(1.64*df[column_name1].std())
    df['middle_value'] = df[column_name1].mean()
    df['lower_limit'] = df[column_name1].mean()-(1.64*df[column_name1].std())
    df[column_name1] = df[column_name1]*2

    layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df[column_name1], 
                        name='Issuer'))
    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df['upper_limit'], 
                        marker=dict(color="green"), 
                        name='Upper_Limit'))
    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df['middle_value'], 
                        marker=dict(color="red"), 
                        name='Middle Value'))
    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df['lower_limit'],  
                        marker=dict(color="green"), 
                        name='Lower Limit'))
    fig.update_layout(height=540)
    fig.update_layout(width=960)

    return fig
     
def plot_normal(df, column_name1, column_name2):
    df['middle_value'] = df[column_name1].mean()

    layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)')

    fig = go.Figure(layout=layout)

    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df[column_name1], 
                        name='Issuer'))
    fig.add_trace(go.Scatter(x=df[column_name2], 
                        y=df['middle_value'], 
                        marker=dict(color="red"), 
                        name='Middle Value'))
    fig.update_layout(height=540)
    fig.update_layout(width=960)
    
    return fig

def create_sentiment(df, column_name):
    sentiments = []
    for i in range (len(df)):
        if(df[column_name].iloc[i] > df['upper_limit'].iloc[i]):
            sentiments.append('positive')
        elif(df[column_name].iloc[i] < df['lower_limit'].iloc[i]):
            sentiments.append('negative')
        else:
            sentiments.append('neutral')

    return sentiments

def form_date_weekly(df, start_date, column_name):
    tgl = []
    val = []

    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    delta = timedelta(days=365)
    end_date = start_date + delta

    delta = timedelta(days=1)
    
    while (start_date <= end_date):
        if (not start_date.strftime('%Y-%m-%d') in list(df[column_name])):
            tgl.append(start_date.strftime('%Y-%m-%d'))
            val.append(np.NaN)
        start_date += delta

    return tgl, val

def calculate_weekly_berita(df1, df2 , column_name1, column_name2):
    # df1 = news
    # df2 = stock
  
    totals = []
    dates = []
  
    for i in range(len(df1) - 7):
        tgl = df1[column_name1].iloc[i].split('-')
        if ((datetime(int(tgl[0]), int(tgl[1]), int(tgl[2])).isoweekday() < 6) and (df1[column_name1].iloc[i+7] in list(df2[column_name2]))):
            
            total = 0     
            
            for j in range(i,i+7):
                if (not np.isnan(df1['sentiment_value'].iloc[j])):
                    total += df1['sentiment value'].iloc[j]
            totals.append(total)
            dates.append(df1[column_name1].iloc[i])

    return totals, dates

def calculate_weekly_share(df, column_name):
    weekly_shares = []
    dates = []
    
    for i in range(len(df)-5):
        if ((df[column_name].iloc[i] == 0)): # x/0
            weekly_stock = -1
        else:
            weekly_stock = ((df[column_name].iloc[i+5]-df[column_name].iloc[i])/df[column_name].iloc[i])
            
        dates.append(df['date'].iloc[i+5])
        weekly_shares.append(weekly_stock)

    return dates, weekly_shares

def calculate_score(df, column_name1, column_name2):
    cocok = 0

    for i in range (len(df)):
        if (df[column_name1].iloc[i] == df[column_name2].iloc[i]):
            suitable += 1

    value = (suitable/len(df))*100

    return value

def stemmingText(text): 
    factory = StemmerFactory()
    voter = factory.create_voter()
    text = voter.stem(text)
    return text

def filteringText(text):
    listStopwords = set(stopwords.words('english'))
    filtered = ''
    for txt in text:
        if txt not in listStopwords:
            #filtered.append(txt)
            filtered+=txt
    text = filtered 
    return text

def get_access_token():
    APP_KEY = 'jDoiK1NQq8BvLfGKxZOmRlCq2'
    APP_SECRET = 'rJSajv6auDx9SAOyktZLgN9JJq4rSqgxKPlFBWST7hT1MgbE3d'
    twitter = Twython(APP_KEY, APP_SECRET, oauth_version=2)
    ACCESS_TOKEN = twitter.obtain_access_token()
    twitter = Twython(APP_KEY, access_token=ACCESS_TOKEN)

    return twitter

def search_tweets(keyword):
    twitter = get_access_token()
    search_result = twitter.search(q=keyword, count=2000)

    return search_result

def process_tweets(search_result):
    tweets = search_result['statuses']

    ids = []

    ids = [tweet['id_str'] for tweet in tweets]
    texts = [tweet['text'] for tweet in tweets]
    times = [tweet['retweet_count'] for tweet in tweets]
    favtimes = [tweet['favorite_count'] for tweet in tweets]
    follower_count = [tweet['user']['followers_count'] for tweet in tweets]
    location = [tweet['user']['location'] for tweet in tweets]
    lang = [tweet['lang'] for tweet in tweets]
    date = [tweet['created_at'] for tweet in tweets]

    pl = pd.DataFrame(
        {'id': ids,
        'Tweet': texts,
        'Date':date,
        'Total Retweets': times,
        'Total Favorites':favtimes,
        'Location':location,
        'Language':lang
        }
    )

    return pl