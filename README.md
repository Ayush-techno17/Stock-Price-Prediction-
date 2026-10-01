# Stock Market Prediction using ML, ANN & LSTM

A stock market prediction project focused on forecasting **Reliance Industries (RELIANCE.NS)** closing prices using historical market data, technical indicators, news sentiment, traditional Machine Learning, an Artificial Neural Network (ANN), and a Long Short-Term Memory (LSTM) neural network.

## Project Overview

The project combines two types of information:

- **Market data:** Open, High, Low, Close and Volume
- **News sentiment:** sentiment labels derived from financial news

Multiple predictive approaches are explored:

1. Traditional Machine Learning
2. Artificial Neural Network (ANN)
3. LSTM for sequential/time-series learning

The objective is to investigate how historical price patterns, technical indicators and sentiment information can be used to model stock-price behavior.

> **Note:** Stock prices are highly volatile and influenced by many external factors. Model predictions should not be interpreted as financial advice or guaranteed future prices.

## Dataset

### Stock Market Data

The notebook downloads historical Reliance Industries stock data using `yfinance`.

- **Ticker:** `RELIANCE.NS`
- **Period:** December 20, 2016 to October 3, 2021
- **Features:**
  - Open
  - High
  - Low
  - Close
  - Volume

The original dataset contains approximately **1,183 observations and 5 market features** before additional engineered features are considered.

### News Sentiment

Financial news is incorporated into the dataset using a sentiment label:

- `-1` → Negative
- `0` → Neutral
- `1` → Positive

This allows the project to investigate whether market sentiment can provide additional information for price prediction.

## Features & Preprocessing

The project uses historical market variables along with engineered/derived information such as:

- Open price
- High price
- Low price
- Closing price
- Trading volume
- Moving averages such as MA5 and MA10
- News sentiment

Typical preprocessing steps include:

- Selecting relevant columns
- Handling missing values
- Creating technical indicators
- Scaling numerical features where required
- Splitting the data into training and testing sets
- Creating sequential windows for the LSTM model

## Models

### 1. Machine Learning

Traditional regression models are used as baseline approaches for stock-price prediction.

These models provide a reference point before moving to neural-network-based approaches.

### 2. Artificial Neural Network (ANN)

An ANN is used to learn nonlinear relationships between the input features and the target stock price.

The ANN works with tabular feature representations and provides a comparison against traditional ML models.

### 3. LSTM

The project also includes an LSTM-based time-series model.

Unlike a standard ANN, LSTM is designed to process sequential data and retain information from previous time steps.

The implemented LSTM uses a **30-day lookback window**, meaning each prediction is generated using a sequence of the previous 30 observations.

Conceptually:

```text
Previous 30 days
      ↓
LSTM Layer (64 units)
      ↓
Dropout
      ↓
LSTM Layer (32 units)
      ↓
Dropout
      ↓
Dense Output
      ↓
Predicted Closing Price
```

The LSTM input is represented as a 3-dimensional tensor:

```text
(samples, time_steps, features)
```

where:

- `samples` = number of training sequences
- `time_steps` = 30
- `features` = number of input variables

## Evaluation

The models can be evaluated using regression metrics such as:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R² Score**

The notebook also includes an **Actual vs Predicted** visualization to compare model predictions with observed closing prices.

A model-comparison section is included to compare the ANN and LSTM results.

## Project Workflow

```text
Historical Stock Data
        +
Financial News
        ↓
Data Collection
        ↓
Data Cleaning & Preprocessing
        ↓
Feature Engineering
        ↓
Technical Indicators
        +
News Sentiment
        ↓
Train/Test Split
        ↓
 ┌───────────────┬───────────────┬───────────────┐
 │ Traditional   │      ANN      │     LSTM      │
 │ ML Models     │               │ Time Series   │
 └───────────────┴───────────────┴───────────────┘
        ↓
Model Evaluation
        ↓
MSE / MAE / R²
        ↓
Actual vs Predicted Analysis
```

## Tech Stack

### Programming

- Python

### Data Analysis

- NumPy
- Pandas

### Visualization

- Matplotlib
- Seaborn

### Machine Learning

- Scikit-learn

### Deep Learning

- TensorFlow
- Keras

### Financial Data

- yfinance

### NLP / Sentiment

- Financial news sentiment data

## Project Structure

```text
Stock-Market-Prediction/
│
├── Stock_Market_Prediction_with_LSTM.ipynb
├── README.md
└── data/
    └── news sentiment data (if required)
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Stock-Market-Prediction
```

Install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow yfinance
```

If you are using Google Colab, most of the required Python libraries are already available.

## Running the Project

Open the notebook:

```bash
jupyter notebook Stock_Market_Prediction_with_LSTM.ipynb
```

or upload the notebook to Google Colab.

Run the notebook cells sequentially.

The stock-market data is downloaded through `yfinance`. The news-sentiment portion requires the corresponding sentiment dataset used by the notebook.

## Key Learning Outcomes

This project demonstrates:

- Financial time-series data analysis
- Feature engineering for stock prediction
- Technical indicator generation
- News sentiment integration
- Regression-based Machine Learning
- Artificial Neural Networks
- LSTM-based sequence modelling
- Time-series window creation
- Model evaluation using regression metrics
- Actual vs predicted visualization
- Comparison of ANN and LSTM approaches

## Future Improvements

Possible extensions include:

- Add GRU and Bidirectional LSTM models
- Incorporate more technical indicators such as RSI, MACD and Bollinger Bands
- Use larger and more recent historical datasets
- Improve news sentiment using FinBERT or another financial language model
- Perform systematic hyperparameter tuning
- Use walk-forward validation instead of a single train/test split
- Predict returns or direction in addition to absolute price
- Build an interactive Streamlit dashboard
- Deploy the trained model as an API
- Add live market-data ingestion for experimentation

## Disclaimer

This project is developed for **educational and research purposes**. Stock-market prediction is an inherently uncertain problem, and historical patterns do not guarantee future performance. Nothing in this repository constitutes financial, investment, or trading advice.
