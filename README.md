<img src="images/money-machine-logo.png" alt="money machine logo" width="200" />

# Money Machine: Predicting stock prices
# Aims
Develop machine learning models and trading strategies for stock price prediction.
Also, to investigate the efficiency of stock predictions described in papers/other people work/
# My work
* dataset created based on paper 1
* LSTM model (general idea paper 1)
* walk forward validation strategy
* trading simulation
* trading strategies 

# Examined papers

## Useful
1. Deep Learning for Stock Market Prediction, M. Nabipour, 2020 [(link)](https://www.mdpi.com/1099-4300/22/8/840) <br>

Pros:
* detailed information on how to create a dataset (features engineering) for stock prediction using  past stock data
* detailed information about creating the datasets for DNN (multiple timestamps)
* detailed information about ML models, general information about the DNN models (still beneficial for further investigation)
* huge grid search of model params = throughout comparison of models prediction ability (different errors, different number of days ahead prediction, different params)

Cons:
* no specific architecture for LSTM given (if one assumes one-layer architecture, the predictions are not accurate with the given config and the data is mostly "copied" from the past)
* no description of LSTM retraining strategy

2. Cryptocurrency Price Prediction Using Tweet Volumes and Sentiment Analysis, J. Abraham 2018 [(link)](https://scholar.smu.edu/datasciencereview/vol1/iss3/1/)

Pros:
* idea for a  new dataset using the Google Trend data and  Tweet Volume using https://bitinfocharts.com/
* idea how to adjust search volume index (SVI) of Google Trend data for the ML purposes

Cons:
* sentiment analysis using only VADER (lexicon and rule-based sentiment analysis tool); BERT needs to be examined
* poor tweet collection method
* not clear sentiment analysis and result presentation


## Hold judgement
1. Stock Closing Price Prediction using Machine Learning Techniques, M. Vijh [(read it here)](https://www.sciencedirect.com/science/article/pii/S1877050920307924) <br>

Paper not implemented. The description of the model used is not detailed. Dataset creation is simple (the highest - the lowest prices of the day's stock price it predicts, moving average and the standard deviation of the prices from the past days).
