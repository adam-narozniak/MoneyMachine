<img src="images/money-machine-logo.png" alt="money machine logo" width="200" />

# Money Machine: Predicting stock prices
# Aims
Develop machine learning models and trading strategies for stock price prediction.
Also, to investigate the efficiency of stock predictions described in papers/other people work
# Work
* dataset created based on paper 1
* LSTM model (general idea paper 1)
* walk forward validation strategy
* trading simulation
* trading strategies 

## Google Trend dataset
Google trends normalizes the results to \[0,... , 100]. Proper preprocessing is needed before using the data predictions.
Google provides a detailed information about this process [(link)](https://support.google.com/trends/answer/4365533?hl=en-GB&ref_topic=6248052).

The possible proposed solutions:
* a blog post [(Using Google Trends data to leverage your predictive model)](https://towardsdatascience.com/using-google-trends-data-to-leverage-your-predictive-model-a56635355e3d)
describes how to anchor the data, it's in R
  
* an article [Calibration of Google Trends Time Series](https://arxiv.org/abs/2007.13861) described the deatials how to compare the data and provides the library to do so

* a blog post [Reconstruct Google Trends Daily Data for Extended Period](https://towardsdatascience.com/reconstruct-google-trends-daily-data-for-extended-period-75b6ca1d3420)
It compares a few described methods. The best thing so far. It also provides addtional references to other posts
  
Cons:
* author doesn't take into account the fact that the data is a random sample and doesn't average it therefore the method
  might be further improved
  

* a blog post 



# Examined papers

## Useful
### 1. Deep Learning for Stock Market Prediction, M. Nabipour, 2020 [(link)](https://www.mdpi.com/1099-4300/22/8/840) <br>

Pros:
* detailed information on how to create a dataset (features engineering) for stock prediction using  past stock data
* detailed information about creating the datasets for DNN (multiple timestamps)
* detailed information about ML models, general information about the DNN models (still beneficial for further investigation)
* huge grid search of model params = throughout comparison of models prediction ability (different errors, different number of days ahead prediction, different params)

Cons:
* no specific architecture for LSTM given (if one assumes one-layer architecture, the predictions are not accurate with the given config and the data is mostly "copied" from the past)
* no description of LSTM retraining strategy

---
### 2. Cryptocurrency Price Prediction Using Tweet Volumes and Sentiment Analysis, J. Abraham 2018 [(link)](https://scholar.smu.edu/datasciencereview/vol1/iss3/1/)

Pros:
* idea for a  new dataset using the Google Trend data and  Tweet Volume using https://bitinfocharts.com/
* idea how to adjust search volume index (SVI) of Google Trend data for the ML purposes based on [(blog post)](https://erikjohansson.blogspot.com/2014/12/creating-daily-search-volume-data-from.html)


Cons:
* sentiment analysis using only VADER (lexicon and rule-based sentiment analysis tool); BERT needs to be examined
* poor tweet collection method
* not clear sentiment analysis and result presentation

I felt the need for further examination of the Google Trends.
I leaned towards the following papers:
1) The Proper Use of Google Trends in Forecasting Models, C. Marcelo, 2021 [(link)](https://arxiv.org/abs/2104.03065)
2) Assessing the Methods, Tools, and Statistical Approaches in Google Trends Research: Systematic Review, A. Mavragani, 2018 [(link)](https://www.jmir.org/2018/11/e270)
### 3. The Proper Use of Google Trends in Forecasting Models, C. Marcelo, 2021 [(link)](https://arxiv.org/abs/2104.03065)

Pros:
* you should take the average of many results of Google Trends to improve your model performance
* in reference one approach to google data preprocessing is (page 44) [(link)](https://www.oecd-ilibrary.org/economics/tracking-activity-in-real-time-with-google-trends_6b9c7518-en)
Cons:
* no description how to adjust the data for the daily forecast purposes



## Hold judgement
### 1. Stock Closing Price Prediction using Machine Learning Techniques, M. Vijh [(link)](https://www.sciencedirect.com/science/article/pii/S1877050920307924)

Paper not implemented. The description of the model used is not detailed. Dataset creation is simple (the highest - the lowest prices of the day's stock price it predicts, moving average and the standard deviation of the prices from the past days).
