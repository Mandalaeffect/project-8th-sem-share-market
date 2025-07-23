import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import base64
from io import BytesIO
import requests
from datetime import datetime

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def prediction_accuracy(y_true, y_pred, threshold=0.05):
    correct = np.abs((y_true - y_pred) / y_true) <= threshold
    return np.mean(correct) * 100, correct

def get_nepse_today_price(symbol=None):
    """
    Fetch today's NEPSE data. Optionally filter by symbol.
    """
    url = "https://nepse-data-api.vercel.app/api/today"
    resp = requests.get(url)
    if resp.status_code != 200:
        print("Failed to fetch NEPSE data.")
        return None
    data = resp.json()['data']
    df_today = pd.DataFrame(data)
    if symbol:
        df_today = df_today[df_today['symbol'] == symbol]
    return df_today

def run_forecast_on_df(df, date_col='Date', price_col='Ltp', future_days=1, verify_next_day=True, symbol=None):
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, price_col])

    df_daily = df.set_index(date_col).resample('D').ffill().reset_index()
    df_daily = df_daily[df_daily[date_col].dt.dayofweek != 5]
    df_daily['is_friday'] = (df_daily[date_col].dt.dayofweek == 4).astype(int)

    split_idx = int(len(df_daily) * 0.8)
    train_df = df_daily.iloc[:split_idx]
    test_df = df_daily.iloc[split_idx:]
    last_price = df_daily[price_col].iloc[-1]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df[[price_col]])
    test_scaled = scaler.transform(test_df[[price_col]])

    def create_sequences(data, seq_len):
        x, y = [], []
        for i in range(len(data)-seq_len):
            x.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        return np.array(x), np.array(y)

    seq_length = 10
    x_train, y_train = create_sequences(train_scaled, seq_length)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, y_train, epochs=20, verbose=0)

    lstm_preds_test = []
    last_seq = train_scaled[-seq_length:]
    for i in range(len(test_df)):
        pred = model.predict(last_seq.reshape(1, seq_length, 1), verbose=0)
        lstm_preds_test.append(pred[0,0])
        last_seq = np.append(last_seq, test_scaled[i]).reshape(-1,1)[-seq_length:]
    lstm_preds_test = scaler.inverse_transform(np.array(lstm_preds_test).reshape(-1,1)).flatten()

    prophet_train = train_df[[date_col, price_col]].rename(columns={date_col: 'ds', price_col: 'y'})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(prophet_train)

    future_test = test_df[[date_col]].rename(columns={date_col: 'ds'})
    prophet_forecast = prophet.predict(future_test)
    prophet_preds_test = prophet_forecast['yhat'].values

    actual_test = test_df[price_col].values

    lstm_rmse = np.sqrt(mean_squared_error(actual_test, lstm_preds_test))
    prophet_rmse = np.sqrt(mean_squared_error(actual_test, prophet_preds_test))
    lstm_mae = mean_absolute_error(actual_test, lstm_preds_test)
    prophet_mae = mean_absolute_error(actual_test, prophet_preds_test)
    lstm_mape = mape(actual_test, lstm_preds_test)
    prophet_mape = mape(actual_test, prophet_preds_test)

    lstm_acc_pct, lstm_acc_flags = prediction_accuracy(actual_test, lstm_preds_test)
    prophet_acc_pct, prophet_acc_flags = prediction_accuracy(actual_test, prophet_preds_test)

    test_df = test_df.copy()
    test_df['LSTM_Pred'] = lstm_preds_test
    test_df['Prophet_Pred'] = prophet_preds_test
    test_df['LSTM_Accurate'] = np.where(lstm_acc_flags, "✅", "❌")
    test_df['Prophet_Accurate'] = np.where(prophet_acc_flags, "✅", "❌")

    # Predict future
    last_seq_full = scaler.transform(df_daily[[price_col]])[-seq_length:]
    future_preds_lstm = []
    for _ in range(future_days):
        pred = model.predict(last_seq_full.reshape(1, seq_length, 1), verbose=0)
        future_preds_lstm.append(pred[0,0])
        last_seq_full = np.append(last_seq_full, pred).reshape(-1,1)[-seq_length:]
    future_preds_lstm = scaler.inverse_transform(np.array(future_preds_lstm).reshape(-1,1)).flatten()

    last_date = df_daily[date_col].max()
    future_dates = []
    current_date = last_date + pd.Timedelta(days=1)
    while len(future_dates) < future_days:
        if current_date.dayofweek != 5:
            future_dates.append(current_date)
        current_date += pd.Timedelta(days=1)

    future_df = pd.DataFrame({'ds': future_dates})
    forecast = prophet.predict(future_df)

    combined_signal = (0.5 * future_preds_lstm) + (0.5 * forecast['yhat'].values)

    signals = []
    for price in combined_signal:
        if price > last_price:
            signals.append("Profit ↑")
        elif price < last_price:
            signals.append("Loss ↓")
        else:
            signals.append("Neutral →")

    result_df = pd.DataFrame({
        'Date': future_dates,
        'LSTM_Pred': future_preds_lstm,
        'Prophet_Pred': forecast['yhat'].values,
        'Combined': combined_signal,
        'Signal': signals
    })

    # Optional: verify next day
    verification_result = None
    if verify_next_day and future_days == 1:
        today_data = get_nepse_today_price(symbol=symbol)
        if today_data is not None and not today_data.empty:
            if symbol:
                actual_price = float(today_data.iloc[0]['closingPrice'])
            else:
                actual_price = float(today_data['closingPrice'].mean())
            predicted_price = result_df['Combined'].iloc[0]
            error_pct = abs(predicted_price - actual_price) / actual_price * 100
            accurate = "✅ Accurate" if error_pct <= 5 else "❌ Not Accurate"
            verification_result = {
                'Predicted': predicted_price,
                'Actual': actual_price,
                'Error%': error_pct,
                'Result': accurate
            }

    # Plot
    plt.figure(figsize=(12,6))
    plt.plot(df_daily[date_col], df_daily[price_col], label='Actual', color='blue')
    plt.plot(test_df[date_col], lstm_preds_test, label='LSTM Test', color='green')
    plt.plot(test_df[date_col], prophet_preds_test, label='Prophet Test', color='orange')
    plt.plot(result_df['Date'], result_df['Combined'], label='Future Forecast', color='red')
    plt.axvline(x=last_date, linestyle='--', color='gray')
    plt.title("NEPSE Forecast: Next {} Trading Days".format(future_days))
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    plot_data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plot_url = f"data:image/png;base64,{plot_data}"

    metrics = {
        'LSTM': {'RMSE': lstm_rmse, 'MAE': lstm_mae, 'MAPE': lstm_mape, 'Accuracy_5pct': lstm_acc_pct},
        'Prophet': {'RMSE': prophet_rmse, 'MAE': prophet_mae, 'MAPE': prophet_mape, 'Accuracy_5pct': prophet_acc_pct}
    }

    return result_df, plot_url, metrics, test_df, verification_result
