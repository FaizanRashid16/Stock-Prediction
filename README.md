# Stock-Prediction
Here's a comprehensive README for your project:

---

# Stock Trend Prediction

This project leverages an LSTM (Long Short-Term Memory) neural network to predict stock price trends based on historical data. Users can input a stock ticker symbol, and the app retrieves historical stock data from Yahoo Finance. It preprocesses the data, scales it, and uses the LSTM model to predict future stock prices. The app visualizes both the original and predicted stock prices, providing an intuitive interface for analyzing stock trends.

## Features

- **User Input**: Enter any stock ticker symbol.
- **Data Retrieval**: Automatically fetch historical stock data from Yahoo Finance.
- **Data Preprocessing**: Scale and transform the data for model prediction.
- **Prediction**: Use an LSTM model to predict future stock prices.
- **Visualization**: Display original and predicted stock prices on an interactive chart.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/FaizanRashid16/stock-Prediction.git
   cd stock-trend-prediction
   ```

2. Install the required Dependencies
  

3. Download the pre-trained LSTM model and place it in the project directory. Ensure the file is named `model.h5`.

## Usage

1. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Enter the stock ticker symbol (e.g., `AAPL` for Apple Inc.) and press Enter.

4. View the historical data summary, the closing price vs. time chart, and the predictions vs. original prices chart.



## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- keras
- streamlit
- yfinance
- scikit-learn



