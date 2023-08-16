# Stock Price Prediction
The goal of this project is to develop a model capable of predicting the stock price for a specified future date. The model is trained using financial data obtained from Yahoo Finance and is built upon various machine learning algorithms. After thorough evaluation, the Long Short-Term Memory (LSTM) model emerged as the top performer and was chosen as the final model with.

## Feature Selection
The dataset downloaded from Yahoo Finance comprises a time series of daily stock trading metrics, including Open, High, Low, Close, Adjusted Close (Adj Close), and Volume. The target variable for prediction is the Adjusted Close column.

At the outset, the features considered for training included Open, High, Low, and Volume. However, upon analysis, Close was excluded due to its high correlation with Adj Close. Surprisingly, employing only Adj Close as both the feature and target variable resulted in improved prediction accuracy. This decision was motivated by the observed enhancement in prediction accuracy and the recognition that the other features are price-related and remain unknown before the date of interest.

## Data Preprocessing
The data from Yahoo Finance is naturally formatted as a time series with dates as the index. While this initial format is suitable for analysis, it's important to note that stock markets operate only on business days. As a result, there are gaps in the data on weekends and holidays. To address this, the .asfreq() method was utilized to convert the irregularly spaced data into a regular frequency. This transformation contributed to enhanced prediction accuracy compared to using the original data. The method introduces NaN values for the days without data, necessitating the use of forward filling to populate these gaps.

## Methodology
To explore the best model for the task, a comparison was made between the Gated Recurrent Unit (GRU) and LSTM models. GridSearch was applied to both models to identify optimal parameters. Following parameter tuning, the LSTM model demonstrated superior accuracy compared to the GRU model.

## Metrics Used: Mean Absolute Percentage Error (MAPE)
For the evaluation of our stock price prediction model, we adopted the Mean Absolute Percentage Error (MAPE) metric. MAPE is a common performance measure in time series forecasting tasks, including stock price prediction. It quantifies the accuracy of predictions by computing the absolute percentage difference between predicted and actual values, providing insight into the magnitude of prediction errors relative to the actual values.

Throughout this project, the developed LSTM model achieved a Mean Absolute Percentage Error (MAPE) of 0.58. This MAPE value serves as an indication of the model's performance, showcasing its ability to predict stock prices with an average error of 0.58% relative to the actual values in test set.

## Conclusion
In conclusion, the task of predicting stock prices is a complex and challenging endeavor. The stock market is influenced by a multitude of factors, including economic indicators, geopolitical events, market sentiment, and unforeseen occurrences. While this project has successfully implemented a Long Short Term Memory (LSTM) model to predict stock prices based on historical data, it's important to acknowledge the limitations of such models.

It's important to approach stock market prediction with a comprehensive understanding of its intricacies and the many factors that can affect it.

## How to Use
This script serves as a template that can be customized for any stock of interest. To obtain the desired price prediction for a particular stock, simply replace the following variables with the values corresponding to your target stock and desired date range:

stock_symbol = 'AAPL'<br>
start_date = '2022-01-01'<br>
end_date = '2023-07-30'<br>

## Included Files
Stock_price_prediction.ipynb: Jupyter Notebook containing the project code.
Stock_data.csv: Sample CSV file containing stock data.

## Installation
To get started, clone this GitHub repository and ensure you have the required libraries installed, as listed below.

## Libraries Used
Yfinance: Data collection from Yahoo Finance API.<br>
Numpy and Pandas: Data manipulation and analysis.<br>
Matplotlib: Visualization of data and results.<br>
Scikit-learn: Data modeling and machine learning.<br>
TensorFlow: Model training and evaluation.<br>

## Acknowledgements
We extend our gratitude to Yahoo Finance for providing a free API that supplies daily trading data for numerous publicly traded companies. This data source has been instrumental in the development of this project.

## Blog post link
https://medium.com/@miryam.ychen/predicting-stock-prices-with-machine-learning-a-deep-dive-8f80bfd585cd
