#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[62]:


import pandas as pd
data=pd.read_csv("IndianFinancialNews.csv")
data


# In[ ]:





# In[ ]:





# In[ ]:






# In[63]:


# Import necessary libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon (if not already downloaded)
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_csv('IndianFinancialNews.csv')

# Initialize the VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get sentiment scores
def get_sentiment_scores(text):
    scores = sid.polarity_scores(text)
    return scores['compound']  # Return the compound score

# Apply the sentiment analyzer to each text
data['sentiment_score'] = data['Title'].apply(get_sentiment_scores)

# Save the updated data with sentiment scores to a new CSV file
data.to_csv('IndianFinancialNews_with_sentiment.csv', index=False)

# Display the DataFrame with sentiment scores
print(data[['Title', 'sentiment_score']].head())


# In[ ]:





# In[ ]:





# In[ ]:





# In[64]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import yfinance as yf
import plotly.graph_objects as go


# In[65]:


get_ipython().system('pip install yfinance')


# In[ ]:





# In[66]:


# Define the ticker symbol for HDFC Bank
ticker_symbol = 'HDFCBANK.NS'  # HDFC Bank stock ticker symbol for NSE (National Stock Exchange)

# Define the date range
start_date = '2010-05-01'
end_date = '2020-05-01'

# Fetch historical data from Yahoo Finance
stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Save the fetched data to a CSV file
csv_filename = 'stock_data.csv'
stock_data.to_csv(csv_filename)

print(f"Stock data saved to {csv_filename}")


# In[67]:


df=pd.read_csv("hdfc_stock_data.csv")
df


# In[ ]:





# In[75]:


# Import necessary libraries
import pandas as pd

# Load the dataset
data = pd.read_csv('IndianFinancialNews_with_sentiment.csv')

# Filter news articles containing the keyword "HDFC"
hdfc_news = data[data['Title'].str.contains('HDFC''Private Bank', case=False) | data['Title'].str.contains('HDFC', case=False)]

# Display the filtered news articles with Date and Sentiment Score
print(hdfc_news[['Date', 'Title', 'sentiment_score']])

# Store the filtered news articles with Date and Sentiment Score in a new CSV file
hdfc_news[['Date', 'Title', 'sentiment_score']].to_csv('hdfc_news_filtered.csv', index=False)



# In[76]:


import pandas as pd

# Read the CSV file
data = pd.read_csv('hdfc_news_filtered.csv')

# Convert data types
# For example, converting 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Write to a new CSV file with updated data format
data.to_csv('updated_file.csv', index=False)


# In[79]:


dff=pd.read_csv('updated_file.csv')
dff


# In[84]:


import pandas as pd

# Assuming you have two datasets: stock_data and news_data
# stock_data contains stock prices with a 'Date' column
# news_data contains news articles with a 'Date' column

# Load stock_data and news_data
stock_data = pd.read_csv('stock_data.csv')
news_data = pd.read_csv('updated_file.csv')

# Merge the two datasets on the 'Date' column
merged_data = pd.merge(stock_data, news_data, on='Date', how='inner')

merged_data.to_csv('merged_data.csv', index=False)

# Display the merged dataset
print(merged_data)


# In[86]:


dfff=pd.read_csv('merged_data.csv')
dfff


# In[89]:


import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('merged_data.csv')

# Convert 'Date' column to datetime and set it as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Ensure data is sorted chronologically
data.sort_index(inplace=True)

# Prepare the data for autoregression
target_variable = 'Close'
lag = 7  # Number of lag observations to include as features

# Create lagged dataset
lagged_data = pd.concat([data[target_variable].shift(i) for i in range(lag + 1)], axis=1)
lagged_data.columns = ['t'] + [f't-{i}' for i in range(1, lag + 1)]

# Drop rows with NaN values (due to shifting)
lagged_data.dropna(inplace=True)

# Split the data into train and test sets
train_size = int(len(lagged_data) * 0.8)
train, test = lagged_data[:train_size], lagged_data[train_size:]

# Fit autoregressive model
model = AutoReg(train['t'], lags=lag)
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

# Calculate RMSE
rmse = mean_squared_error(test['t'], predictions, squared=False)
print('Test RMSE:', rmse)

# Plot predictions vs actual
plt.plot(test.index, test['t'], label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('AR Model Predictions')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




