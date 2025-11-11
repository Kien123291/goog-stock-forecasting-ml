Google Stock Price Prediction (LSTM Time Series)
This repository contains an end-to-end time series modeling project that predicts Google (GOOG) stock prices using deep learning, with a focus on LSTM-based recurrent neural networks and simple technical indicators.
The project is implemented in a single Jupyter Notebook:
Google stock prediction.ipynb
It walks through data collection, exploratory analysis, feature engineering, model building (TensorFlow & PyTorch LSTMs), and short-horizon forecasting.
1. Project Overview
The goal of this project is to:
Download historical daily price data for GOOG using yfinance
Explore and visualize trends, distributions, and relationships in the data
Engineer features such as Simple Moving Averages (SMA) and daily returns
Frame the problem as a supervised time series prediction task:
Use the previous N days of prices/indicators to predict the next day’s closing price
Build and compare LSTM-based models to capture temporal dependencies
Generate a 10-day ahead forecast based on trained models
This project is educational / experimental and not financial advice.
2. Data
Source: Yahoo Finance via the yfinance API
Ticker: GOOG
Start date: 2015-10-22 (configurable in the notebook)
Raw columns used:
Open
High
Low
Close
Volume
Dividends
Stock Splits
Key preprocessing steps:
Re-ordered columns to emphasize Close
Converted Volume to numeric type
Verified:
No missing values (visual & programmatic checks)
No duplicate rows
Basic quality diagnostics via:
.info(), .describe()
Missing value heatmap
3. Exploratory Data Analysis (EDA)
The notebook includes rich visual analysis to understand GOOG’s behavior:
Correlation matrix & heatmap across numeric features
Distribution plots (histograms, KDE) for price and volume
Time series plots of:
Open, High, Low, Close, Volume
Daily returns
Candlestick chart (Plotly) for price action
Resampled yearly averages to observe long-run trends
These plots help check for trends, volatility, and feature relevance before modeling.
4. Feature Engineering
To improve predictive power, the notebook constructs:
Daily Return:
Percentage change in Close
Simple Moving Averages (SMA):
SMA_10, SMA_20, SMA_50 on closing prices
Scaled versions of features using MinMaxScaler:
Applied to price and volume columns (and SMA features)
For supervised learning:
Sliding window approach:
Use a sequence of past days (e.g. 10 days) as input
Predict the next day’s Close (or related target)
Datasets are split into:
Train (80%)
Validation (10%)
Test (10%)
Split is chronological (no shuffling) to respect time order
5. Modeling Approach
The notebook experiments with LSTM-based models implemented in both:
TensorFlow / Keras
PyTorch
5.1 TensorFlow LSTM
Input: sequences of past days (multivariate features)
Architecture:
Stacked LSTM layers
Dense layers on top for regression output
Loss: Mean Squared Error (MSE)
Metrics:
Test loss on unseen time window
Visualization:
Plot of actual vs predicted normalized closing prices
5.2 PyTorch LSTM (Multiple Feature Sets)
A custom NeuralNetwork class (LSTM + Dropout + Linear) is trained on:
Features with 10-day SMA
Features with 20-day SMA
Features with both SMAs / all features
Key components:
TensorDataset + DataLoader for efficient batching
Training loop with:
Forward pass
Backpropagation & optimizer steps (Adam)
Epoch-level train & validation MSE
Comparison plot:
Actual vs predicted test set for all three PyTorch models
This setup shows how technical indicators and different input designs affect predictive performance.
6. Next 10-Day Forecast
Using the best-performing PyTorch LSTM configuration:
Take the last available sequence from the dataset
Iteratively generate predictions for the next 10 days
Inverse-transform scaled outputs back to price space
Store results in a DataFrame for inspection and plotting
This demonstrates practical sequence-to-sequence style forecasting using an autoregressive loop.
7. Tech Stack
Main libraries used:
Data & Visualization
pandas, numpy
matplotlib, seaborn
plotly
tqdm
Data Source
yfinance
Preprocessing & Utilities
scikit-learn (MinMaxScaler, train_test_split)
Deep Learning
TensorFlow, Keras
PyTorch, torchvision (if installed), torch.utils.data
8. How to Run
Clone the repository (or download the notebook):
git clone <your-repo-url>.git
cd <your-repo-name>
Create & activate a virtual environment (recommended).
Install dependencies (adjust as needed):
pip install numpy pandas matplotlib seaborn yfinance plotly scikit-learn torch torchvision torchaudio tensorflow
Launch Jupyter Notebook:
jupyter notebook
Open Google stock prediction.ipynb and run all cells in order.
Note: The notebook fetches live market data with yfinance, so results can slightly change over time depending on the latest GOOG prices.
9. Disclaimer
This project is for learning and experimentation in time series modeling and deep learning.
It is not investment advice and should not be used directly for live trading decisions.
If you’d like, I can now:
Add a short “About Me” / “Author” section tailored to your CV
Adjust the README tone (more formal/academic or more portfolio-style)
Add a requirements.txt or environment.yml snippet based exactly on this notebook.
