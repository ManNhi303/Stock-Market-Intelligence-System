# predict.py

import pandas as pd
import numpy as np
import torch
import os
import pickle
import json
from datetime import timedelta

from src.config import Config
from src.model import DA_RNN_MultiTask
from src.data_processing import (
    calculate_sma, calculate_ema, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_stochastic_k, calculate_adx
)

TICKER_TO_PREDICT = 'BTC-USD' 

def load_artifacts(ticker, config):
    print(f"--- Loading artifacts for {ticker} from new structure ---")
    model_dir_base = config.MODEL_SAVE_PATH
    safe_ticker = ticker.replace('.VN','_VN').replace('^','')
    model_dir_specific = os.path.join(model_dir_base, "by_ticker", safe_ticker)
    
    model_path = os.path.join(model_dir_specific, "model.pth")
    scaler_X_path = os.path.join(model_dir_specific, "scaler_X.pkl")
    scaler_y_path = os.path.join(model_dir_specific, "scaler_y_pct.pkl")
    features_path = os.path.join(model_dir_specific, "features.json")

    for path in [model_path, scaler_X_path, scaler_y_path, features_path]:
        if not os.path.exists(path):
            print(f"ERROR: Artifact not found at {path}. Please run train.py for this ticker first.")
            return None, None, None, None
            
    with open(features_path, 'r') as f:
        feature_list = json.load(f)['features']
        
    model = DA_RNN_MultiTask(config, feature_list)
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval()
    
    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y_pct = pickle.load(f)
        
    print("--- Artifacts loaded successfully ---")
    return model, scaler_X, scaler_y_pct, feature_list

def prepare_inference_data(ticker, config, feature_list):
    print(f"--- Preparing latest data for {ticker} for inference ---")
    df_full = pd.read_csv(config.DATA_PATH)

    # chuẩn hóa dữ liệu
    cols_to_drop = ['Dividends', 'Stock Splits']
    df_full.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    rename_map = {'Date': 'date', 'Symbol': 'code'}
    df_full.rename(columns=rename_map, inplace=True)
    df_full.columns = df_full.columns.str.lower()

    # Chuyển đổi cột 'date', sau đó loại bỏ các dòng có ngày không hợp lệ
    df_full['date'] = pd.to_datetime(df_full['date'], utc=True, errors='coerce')
    df_full.dropna(subset=['date'], inplace=True) # Xóa các dòng có ngày tháng không hợp lệ (NaT)
    df_full['date'] = df_full['date'].dt.date 

    df = df_full[df_full['code'] == ticker].copy().sort_values('date').reset_index(drop=True)
    
    df = df.tail(config.T + 50).reset_index(drop=True)
    df['SMA_10'] = calculate_sma(df['close'], 10); df['EMA_10'] = calculate_ema(df['close'], 10); df['RSI_14'] = calculate_rsi(df['close'], 14); df['MACD'] = calculate_macd(df['close']); df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['close']); df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close']); df['ADX_14'] = calculate_adx(df['high'], df['low'], df['close']); df['STOCHk_14_3_3'] = calculate_stochastic_k(df['high'], df['low'], df['close'])
    
    df_X = pd.DataFrame(index=df.index)
    for col in config.PCT_CHANGE_FEATURES:
        df_X[f'{col}_pct_change'] = df[col].pct_change()
    for col in config.STATIONARY_FEATURES:
        df_X[col] = df[col]
        
    last_known_date = df['date'].iloc[-1]
    prediction_date = last_known_date + timedelta(days=1)
    
    df_final_features = df_X.tail(config.T)
    if len(df_final_features) < config.T:
        print(f"ERROR: Not enough data to form a full window of size {config.T}.")
        return None, None, None
        
    scaled_features = scaler_X.transform(df_final_features[feature_list])
    input_tensor = torch.FloatTensor(scaled_features).unsqueeze(0).to(config.DEVICE)
    last_close_price = df['close'].iloc[-1]
    
    return input_tensor, last_close_price, prediction_date

def main():
    config = Config()
    ticker = TICKER_TO_PREDICT
    artifacts = load_artifacts(ticker, config)
    if artifacts[0] is None: return
    model, scaler_X, scaler_y_pct, feature_list = artifacts
    inference_data = prepare_inference_data(ticker, config, feature_list)
    if inference_data[0] is None: return
    input_tensor, last_close_price, prediction_date = inference_data
    
    print("\n" + "="*50 + "\n--- MAKING PREDICTION ---")
    with torch.no_grad():
        pct_pred_scaled, trend_logits = model(input_tensor)
        
    predicted_pct_scaled = pct_pred_scaled.cpu().numpy()
    predicted_pct = scaler_y_pct.inverse_transform(predicted_pct_scaled).flatten()[0]
    predicted_price = last_close_price * (1 + predicted_pct)
    
    trend_pred_index = torch.argmax(trend_logits, dim=1).cpu().item()
    trend_labels = {0: 'Down', 1: 'Stable', 2: 'Up'}
    predicted_trend = trend_labels[trend_pred_index]
    
    print(f"Prediction for Ticker: {ticker}"); print(f"Prediction Date: {prediction_date.strftime('%Y-%m-%d')}"); print(f"Last Known Close Price: {last_close_price:,.2f}"); print("-" * 30); print(f"Predicted Trend: {predicted_trend}"); print(f"Predicted Next Day's Close Price: {predicted_price:,.2f}"); print("="*50 + "\n")
    
    results_dir = os.path.join('results', 'latest_prediction')
    os.makedirs(results_dir, exist_ok=True)
    safe_ticker = ticker.replace('.VN','_VN').replace('^','')
    output_path = os.path.join(results_dir, f"{safe_ticker}_prediction.csv")
    prediction_df = pd.DataFrame([{'Ticker': ticker, 'Last_Date': last_known_date.strftime('%Y-%m-%d'), 'Prediction_Date': prediction_date.strftime('%Y-%m-%d'), 'Last_Close_Price': last_close_price, 'Predicted_Price': predicted_price, 'Predicted_Trend': predicted_trend}])
    prediction_df.to_csv(output_path, index=False)
    print(f"Prediction saved to: {output_path}")

if __name__ == '__main__':
    main()