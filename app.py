from datetime import date, timedelta
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
# Set page config
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# Stock list
stocks = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Function to create sequences


def create_sequences(dataset, lookback=60):
    x, y = [], []
    for i in range(lookback, len(dataset)):
        x.append(dataset[i-lookback:i, :])
        y.append(dataset[i, 3])  # Close price is at index 3
    return np.array(x), np.array(y)

# Function to test a model


def test_model(ticker):
    try:
        # Load the saved model and scaler
        model = load_model(f"{ticker}_lstm_model.h5")
        scaler = joblib.load(f"{ticker}_scaler.save")

        # Download test data (last 6 months)
        end_date = date.today()
        start_date = end_date - timedelta(days=180)
        test_data = yf.download(ticker, start=start_date, end=end_date)
        test_data = test_data[['Open', 'High', 'Low', 'Close']]

        # Add SMA30
        test_data["SMA30"] = test_data["Close"].rolling(
            window=30).mean().fillna(method="bfill")

        # Scale the test data
        scaled_test_data = scaler.transform(test_data)

        # Create sequences for testing
        x_test, y_test = create_sequences(scaled_test_data, 60)

        if len(x_test) == 0:
            return None, None, None

        # Make predictions
        predictions = model.predict(x_test)

        # Inverse transform predictions and actual values
        dummy_array_pred = np.zeros(
            (len(predictions), scaled_test_data.shape[1]))
        dummy_array_pred[:, 3] = predictions.flatten()
        predictions_actual = scaler.inverse_transform(dummy_array_pred)[:, 3]

        dummy_array_actual = np.zeros((len(y_test), scaled_test_data.shape[1]))
        dummy_array_actual[:, 3] = y_test
        y_test_actual = scaler.inverse_transform(dummy_array_actual)[:, 3]

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, predictions_actual)
        mae = mean_absolute_error(y_test_actual, predictions_actual)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_actual, predictions_actual)

        return y_test_actual, predictions_actual, {
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2
        }

    except Exception as e:
        st.error(f"Error testing {ticker}: {str(e)}")
        return None, None, None

# Function to predict future prices


def predict_future(ticker, days=30):
    try:
        # Load the saved model and scaler
        model = load_model(f"{ticker}_lstm_model.h5")
        scaler = joblib.load(f"{ticker}_scaler.save")

        # Download recent data
        end_date = date.today()
        start_date = end_date - timedelta(days=120)
        recent_data = yf.download(ticker, start=start_date, end=end_date)
        recent_data = recent_data[['Open', 'High', 'Low', 'Close']]

        # Add SMA30
        recent_data["SMA30"] = recent_data["Close"].rolling(
            window=30).mean().fillna(method="bfill")

        # Get the last 60 days of data
        last_60_days = recent_data.iloc[-60:].copy()
        scaled_data = scaler.transform(last_60_days)

        future_predictions = []
        current_sequence = scaled_data.copy()

        for _ in range(days):
            x = current_sequence[-60:].reshape(1, 60, scaled_data.shape[1])
            next_pred = model.predict(x, verbose=0)

            new_row = current_sequence[-1].copy()
            new_row[3] = next_pred[0, 0]  # Update Close price

            # Simple assumptions for other price fields
            new_row[0] = new_row[3] * 0.99  # Open
            new_row[1] = new_row[3] * 1.01  # High
            new_row[2] = new_row[3] * 0.98  # Low

            current_sequence = np.vstack([current_sequence, new_row])
            future_predictions.append(new_row)

        # Inverse transform
        future_predictions = np.array(future_predictions)
        future_prices = scaler.inverse_transform(future_predictions)[:, 3]

        # Create future dates
        last_date = recent_data.index[-1]
        future_dates = [last_date + timedelta(days=i)
                        for i in range(1, days+1)]

        return recent_data, future_dates, future_prices

    except Exception as e:
        st.error(f"Error predicting future for {ticker}: {str(e)}")
        return None, None, None


# Main app
st.title("📈 Stock Prediction Dashboard")

# Sidebar
st.sidebar.header("Settings")
selected_stock = st.sidebar.selectbox("Select Stock", stocks)
prediction_days = st.sidebar.slider("Days to Predict", 7, 90, 30)

# Tabs
tab1, tab2 = st.tabs(["Model Performance", "Future Prediction"])

with tab1:
    st.header("Model Performance Evaluation")

    if st.button("Test Model Performance"):
        with st.spinner("Testing model..."):
            actual, predicted, metrics = test_model(selected_stock)

            if actual is not None:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE", f"{metrics['rmse']:.2f}")
                col2.metric("MAE", f"{metrics['mae']:.2f}")
                col3.metric("MSE", f"{metrics['mse']:.2f}")
                col4.metric("R²", f"{metrics['r2']:.4f}")

                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(actual, label='Actual', linewidth=2)
                ax.plot(predicted, label='Predicted', linewidth=2)
                ax.set_title(f'{selected_stock} - Actual vs Predicted Prices')
                ax.set_xlabel('Time')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

with tab2:
    st.header("Future Price Prediction")

    if st.button("Predict Future Prices"):
        with st.spinner("Making predictions..."):
            historical_data, future_dates, future_prices = predict_future(
                selected_stock, prediction_days)

            if historical_data is not None and future_prices is not None:
                # Ensure we are using floats, not Series
                current_price = float(historical_data['Close'].iloc[-1])
                predicted_price = float(future_prices[-1])
                change_percent = ((predicted_price - current_price) / current_price) * 100

                # Display metrics
                st.metric(
                    label=f"Current {selected_stock} Price",
                    value=f"${current_price:.2f}",
                    delta=f"{change_percent:+.2f}% predicted change"
                )

                # Plot historical and future predictions
                fig, ax = plt.subplots(figsize=(12, 6))

                # Plot historical data (last 60 days)
                ax.plot(historical_data.index[-60:], historical_data['Close'][-60:],
                        label='Historical', linewidth=2, color='blue')

                # Plot future predictions
                ax.plot(future_dates, future_prices,
                        label='Predicted', linewidth=2, color='red', linestyle='--')

                ax.set_title(
                    f'{selected_stock} - {prediction_days}-Day Price Prediction')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # Show prediction table
                st.subheader("Prediction Details")
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_prices
                })
                st.dataframe(future_df)


# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    This app uses LSTM neural networks to predict stock prices.
    Predictions are for educational purposes only and should not
    be considered financial advice.
    """
)
