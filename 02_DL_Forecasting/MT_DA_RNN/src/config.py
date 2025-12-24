# src/config.py

import torch

class Config:
    DATA_PATH = 'data/all_symbols_daily.csv' 
    MODEL_SAVE_PATH = 'models/'
    RESULTS_PATH = 'results/by_ticker/'

    # --- DATA & FEATURES ---
    TICKER_LIST = [
        'BID.VN', 'BNB-USD', 'BTC-USD', 'CTG.VN', 'ETH-USD', 'FPT.VN',
        'GAS.VN', 'HPG.VN', 'NQ=F', 'SOL-USD', 'TCB.VN','USDT-USD',
        'VCB.VN', 'VHM.VN', 'VIC.VN', 'VPB.VN', '^DJI', '^GSPC',
        '^NYA', '^RUT'
    ]
    TARGET_COLUMN = 'close'
    PCT_CHANGE_FEATURES = ['open', 'high', 'low', 'close', 'volume', 'SMA_10', 'EMA_10', 'MACD', 'BB_upper', 'BB_lower', 'ATR_14']
    STATIONARY_FEATURES = ['RSI_14', 'ADX_14', 'STOCHk_14_3_3']
    LOWER_QUANTILE = 0.33
    UPPER_QUANTILE = 0.67
    T = 15 # Time steps (lookback window)
    TRAIN_RATIO = 0.7
    VALIDATION_RATIO = 0.15

    # --- MODEL HYPERPARAMETERS ---
    LEARNING_RATE = 0.0008
    WEIGHT_DECAY = 0.0006
    N_HIDDEN_ENCODER = 32
    N_HIDDEN_DECODER = 32
    DROPOUT_RATE = 0.18
    LOSS_WEIGHT_CLASSIFICATION = 5.0
    LOSS_WEIGHT_REGRESSION = 1.0
    NUM_CLASSES = 3

    # --- TRAINING ---
    BATCH_SIZE = 128
    EPOCHS = 150
    PATIENCE = 20
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- BACKTESTING PARAMETERS ---
    INITIAL_CAPITAL = 100_000_000  # Vốn ban đầu là 100 triệu
    TRANSACTION_FEE_PCT = 0.001   # Phí giao dịch: 0.1% mỗi chiều