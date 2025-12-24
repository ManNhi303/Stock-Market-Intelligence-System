# src/data_processing.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- 1. Technical Indicator Calculation Functions ---
def calculate_sma(series, window): return series.rolling(window=window).mean()
def calculate_ema(series, span): return series.ewm(span=span, adjust=False).mean()
def calculate_rsi(series, window=14): delta = series.diff(1); gain = (delta.where(delta > 0, 0)).rolling(window=window).mean(); loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean(); rs = gain / (loss + 1e-9); return 100 - (100 / (1 + rs))
def calculate_macd(series, span1=12, span2=26): return calculate_ema(series, span1) - calculate_ema(series, span2)
def calculate_bollinger_bands(series, window=20): sma = calculate_sma(series, window); std = series.rolling(window=window).std(); return sma + (std * 2), sma - (std * 2)
def calculate_atr(high, low, close, window=14): tr = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1); return calculate_ema(tr, window)
def calculate_stochastic_k(high, low, close, k_window=14, smooth_window=3): low_k = low.rolling(window=k_window).min(); high_k = high.rolling(window=k_window).max(); k_percent = 100 * ((close - low_k) / (high_k - low_k + 1e-9)); return k_percent.rolling(window=smooth_window).mean()
def calculate_adx(high, low, close, window=14): plus_dm = high.diff(); minus_dm = low.diff(); plus_dm[plus_dm < 0] = 0; minus_dm[minus_dm > 0] = 0; tr = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1); atr = tr.ewm(span=window, adjust=False).mean(); plus_di = 100 * (plus_dm.ewm(span=window, adjust=False).mean() / (atr + 1e-9)); minus_di = 100 * (np.abs(minus_dm.ewm(span=window, adjust=False).mean()) / (atr + 1e-9)); dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)); return dx.ewm(span=window, adjust=False).mean()

# --- 2. Data Preparation Function ---
def prepare_data_for_multitask_pct(config, ticker):
    print(f"\n--- Preparing data for ticker: {ticker} using QUANTILE-BASED labeling ---")
    try:
        df_full = pd.read_csv(config.DATA_PATH)
        
        df_full.columns = df_full.columns.str.lower()
        if 'symbol' in df_full.columns:
            df_full.rename(columns={'symbol': 'code'}, inplace=True)
            
    except FileNotFoundError:
        print(f"ERROR: File not found at '{config.DATA_PATH}'. Please check path in config.py.")
        return [None] * 9

    df_full['date'] = pd.to_datetime(df_full['date'], utc=True, errors='coerce')
    df_full.dropna(subset=['date'], inplace=True)
    df_full['date'] = df_full['date'].dt.date

    df = df_full[df_full['code'] == ticker].copy().sort_values('date').reset_index(drop=True)

    if len(df) < 100:
        print(f"Warning: Insufficient data for ticker '{ticker}' after processing. Skipping.")
        return [None] * 9
        
    df['SMA_10'] = calculate_sma(df['close'], 10); df['EMA_10'] = calculate_ema(df['close'], 10); df['RSI_14'] = calculate_rsi(df['close'], 14); df['MACD'] = calculate_macd(df['close']); df['BB_upper'], df['BB_lower'] = calculate_bollinger_bands(df['close']); df['ATR_14'] = calculate_atr(df['high'], df['low'], df['close']); df['ADX_14'] = calculate_adx(df['high'], df['low'], df['close']); df['STOCHk_14_3_3'] = calculate_stochastic_k(df['high'], df['low'], df['close'])
    df['price_pct_change'] = df[config.TARGET_COLUMN].pct_change()
    df['target_pct_change'] = df['price_pct_change'].shift(-1)
    df.dropna(subset=['target_pct_change'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    train_end_idx = int(len(df) * config.TRAIN_RATIO)
    target_pct_train = df['target_pct_change'].iloc[:train_end_idx]
    
    if len(target_pct_train) == 0:
        print(f"Warning: Not enough training data to calculate quantiles for ticker '{ticker}'. Skipping.")
        return [None] * 9
        
    lower_bound = target_pct_train.quantile(config.LOWER_QUANTILE); upper_bound = target_pct_train.quantile(config.UPPER_QUANTILE)
    print(f"Dynamic Quantile Thresholds: Down < {lower_bound:.4f}, Up > {upper_bound:.4f}")
    def get_label_dynamic(change_pct):
        if change_pct > upper_bound: return 2
        elif change_pct < lower_bound: return 0
        else: return 1
    df['target_trend'] = df['target_pct_change'].apply(get_label_dynamic)
    print("Class distribution:", df['target_trend'].value_counts(normalize=True).sort_index().to_dict())
    
    df_X = pd.DataFrame(index=df.index)
    for col in config.PCT_CHANGE_FEATURES:
        df_X[f'{col}_pct_change'] = df[col].pct_change()
    for col in config.STATIONARY_FEATURES:
        df_X[col] = df[col]
        
    df_processed = pd.concat([df['date'], df_X, df[['target_pct_change', 'target_trend', 'close']]], axis=1)
    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_processed.dropna(inplace=True)
    df_processed.reset_index(drop=True, inplace=True)
    
    if len(df_processed) < config.T:
        print(f"Warning: Insufficient data after processing for ticker '{ticker}'. Skipping.")
        return [None] * 9
        
    y_target_pct = df_processed['target_pct_change'].values
    y_target_trend = df_processed['target_trend'].values
    last_close_prices = df_processed['close'].values
    dates_for_output = df_processed['date'].values
    feature_list = df_processed.drop(['date', 'target_pct_change', 'target_trend', 'close'], axis=1).columns.tolist()
    df_features = df_processed[feature_list]
    
    train_end = int(len(df_features) * config.TRAIN_RATIO)
    scaler_X = StandardScaler().fit(df_features.iloc[:train_end])
    data_scaled = scaler_X.transform(df_features)
    
    scaler_y_pct = StandardScaler().fit(y_target_pct[:train_end].reshape(-1, 1))
    y_target_pct_scaled = scaler_y_pct.transform(y_target_pct.reshape(-1, 1)).flatten()
    
    def create_windows(features, target_reg, target_cls, last_prices, dates, config):
        X, y_reg, y_cls, last_p, win_dates = [], [], [], [], []
        for i in range(len(features) - config.T):
            X.append(features[i:i + config.T])
            y_reg.append(target_reg[i + config.T - 1])
            y_cls.append(target_cls[i + config.T - 1])
            last_p.append(last_prices[i + config.T - 1])
            win_dates.append(dates[i + config.T - 1])
        return np.array(X), np.array(y_reg), np.array(y_cls, dtype=np.int64), np.array(last_p), np.array(win_dates)
        
    X_final, y_reg_final, y_cls_final, last_p_final, dates_final = create_windows(data_scaled, y_target_pct_scaled, y_target_trend, last_close_prices, dates_for_output, config)
    
    train_end_win = int(len(X_final) * config.TRAIN_RATIO)
    val_end_win = int(len(X_final) * (config.TRAIN_RATIO + config.VALIDATION_RATIO))
    
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_final[:train_end_win]), torch.FloatTensor(y_reg_final[:train_end_win]), torch.LongTensor(y_cls_final[:train_end_win])), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_final[train_end_win:val_end_win]), torch.FloatTensor(y_reg_final[train_end_win:val_end_win]), torch.LongTensor(y_cls_final[train_end_win:val_end_win])), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_final[val_end_win:]), torch.FloatTensor(y_reg_final[val_end_win:]), torch.LongTensor(y_cls_final[val_end_win:])), batch_size=config.BATCH_SIZE, shuffle=False)
    
    last_p_test = last_p_final[val_end_win:]
    dates_test = dates_final[val_end_win:]
    
    y_cls_train_win = y_cls_final[:train_end_win]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_cls_train_win), y=y_cls_train_win)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
    
    print("--- Data preparation complete ---")
    return train_loader, val_loader, test_loader, feature_list, class_weights, scaler_X, scaler_y_pct, last_p_test, dates_test