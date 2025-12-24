# train_baseline.py

import pandas as pd
import numpy as np
import os
import random
import warnings

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import Config

def set_seed(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed_value)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass
    print(f"--- Random seed set to {seed_value} for baseline model ---")

# --- 1. CẤU HÌNH VÀ THIẾT LẬP ---
set_seed(42)
warnings.filterwarnings('ignore')
config = Config()

# --- 2. TẢI VÀ CHUẨN HÓA DỮ LIỆU ---
try:
    df = pd.read_csv(config.DATA_PATH)
    print("Data loaded successfully.")

    # =============================================================
    #  Chuyển tất cả tên cột thành chữ thường
    # =============================================================
    df.columns = df.columns.str.lower()
    
    # Đổi tên cột 'symbol' thành 'code' để nhất quán với file train.py
    if 'symbol' in df.columns:
        df.rename(columns={'symbol': 'code'}, inplace=True)

    # xử lý cột 'date' đã được chuyển thành chữ thường
    df['date'] = pd.to_datetime(df['date'])

except FileNotFoundError:
    print(f"Error: The file '{config.DATA_PATH}' was not found.")
    exit()
except KeyError as e:
    print(f"Error: Column {e} not found after attempting to standardize. Please check your CSV file.")
    exit()

# --- 3. HÀM HỖ TRỢ ---
def create_multivariate_sequences(dataset, target_col_idx, seq_length):
    X, y = [], []
    for i in range(seq_length, len(dataset)):
        X.append(dataset[i-seq_length:i, :])
        y.append(dataset[i, target_col_idx])
    return np.array(X), np.array(y)

# --- 4. VÒNG LẶP XỬ LÝ CHÍNH ---
tickers_to_run = config.TICKER_LIST
results_list = []

sequence_length = 30
lstm_units = 32
dropout_rate = 0.2
dense_units = 25
batch_size_param = 32
epochs_param = 50
features = ['open', 'high', 'low', 'close', 'volume'] 
target_col_index = features.index('close')

for ticker_code in tickers_to_run:
    print(f"\n{'='*25} Processing Baseline LSTM for: {ticker_code} {'='*25}")

    df_ticker = df[df['code'] == ticker_code].copy()
    df_ticker = df_ticker.sort_values('date')
    if df_ticker.empty or len(df_ticker) < sequence_length + 50:
        print(f"--- Ticker {ticker_code} has insufficient data. Skipping. ---")
        continue
    df_ticker['volume'] = df_ticker['volume'].replace(0, 1)

    data_original = df_ticker[features].values
    training_data_len_idx = int(np.ceil(len(data_original) * 0.8))
    train_data_original = data_original[0:training_data_len_idx, :]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_scaled = scaler.fit_transform(train_data_original)
    
    X_train, y_train = create_multivariate_sequences(train_data_scaled, target_col_index, sequence_length)
    
    full_data_scaled = scaler.transform(data_original)
    input_test_pool_scaled = full_data_scaled[training_data_len_idx - sequence_length:, :]
    X_test, _ = create_multivariate_sequences(input_test_pool_scaled, target_col_index, sequence_length)
    
    if X_train.size == 0 or X_test.size == 0:
        print(f"--- Could not create sequences for {ticker_code}. Skipping. ---")
        continue

    num_features = X_train.shape[2]

    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(X_train.shape[1], num_features)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=dense_units),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print(f"Training model for {ticker_code}...")
    model.fit(X_train, y_train, batch_size=batch_size_param, epochs=epochs_param, verbose=0)
    print("Training complete.")

    print("Evaluating model...")
    predictions_scaled = model.predict(X_test, verbose=0)
    
    dummy_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_predictions[:, target_col_index] = predictions_scaled.flatten()
    predictions_unscaled = scaler.inverse_transform(dummy_predictions)[:, target_col_index]
    
    y_test_unscaled = data_original[training_data_len_idx:, target_col_index]

    # Cần đảm bảo độ dài khớp nhau để đánh giá
    min_len = min(len(y_test_unscaled), len(predictions_unscaled))
    y_test_unscaled = y_test_unscaled[:min_len]
    predictions_unscaled = predictions_unscaled[:min_len]
    
    rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions_unscaled))
    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    r2 = r2_score(y_test_unscaled, predictions_unscaled)

    print(f"Results for {ticker_code}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    results_list.append({'Ticker': ticker_code, 'RMSE': rmse, 'MAE': mae, 'R2': r2})

# --- 5. TỔNG KẾT VÀ LƯU KẾT QUẢ ---
print(f"\n{'='*25} FINAL SUMMARY OF BASELINE RESULTS {'='*25}")
if not results_list:
    print("No results to display.")
else:
    results_df = pd.DataFrame(results_list)
    results_df.insert(0, 'STT', range(1, len(results_df) + 1))
    print(results_df.to_string(index=False))

    save_dir = os.path.join("results", "baseline_lstm")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "summary.csv")
    results_df.to_csv(save_path, index=False, float_format='%.4f')
    print(f"\nBaseline results saved to: {save_path}")