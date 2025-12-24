# train.py

import time
import copy
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import traceback
import pandas as pd
import os
import random
import numpy as np

# Import from project source files
from src.config import Config
from src.data_processing import prepare_data_for_multitask_pct
from src.model import DA_RNN_MultiTask
from src.utils import (evaluate_and_collect_results, save_ticker_specific_files,
                       save_model_artifacts, run_backtest_simulation)

def set_seed(seed_value=42):
    """Thiết lập seed để đảm bảo kết quả nhất quán."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"--- Random seed set to {seed_value} ---")

def main():
    """Main function to execute the training and evaluation pipeline."""
    set_seed(42) # Đặt seed ở đầu
    config = Config()
    print(f"Using device: {config.DEVICE}")

    # Tạo các list để lưu kết quả tổng hợp
    all_regression_metrics = []
    all_classification_metrics = []
    all_backtest_performances = []

    for ticker in config.TICKER_LIST:
        print("\n" + "#"*80 + f"\n### STARTING PROCESSING FOR TICKER: {ticker} ###\n" + "#"*80)
        try:
            # 1. Prepare Data
            data_payload = prepare_data_for_multitask_pct(config, ticker)
            if data_payload[0] is None:
                continue

            (train_loader, val_loader, test_loader, feature_list, class_weights,
             scaler_X, scaler_y_pct, last_p_test, test_dates) = data_payload

            # 2. Initialize Model and Training Components
            model = DA_RNN_MultiTask(config, feature_list).to(config.DEVICE)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, 'min', patience=config.PATIENCE//2, factor=0.5, verbose=False)
            criterion_reg = nn.MSELoss()
            criterion_cls = nn.CrossEntropyLoss(weight=class_weights)

            # 3. Training Loop
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_wts = None
            start_time = time.time()
            train_losses, val_losses = [], []

            for epoch in range(config.EPOCHS):
                model.train()
                total_train_loss = 0
                for X_batch, y_batch_reg, y_batch_cls in train_loader:
                    X_batch, y_batch_reg, y_batch_cls = X_batch.to(config.DEVICE), y_batch_reg.to(config.DEVICE), y_batch_cls.to(config.DEVICE)
                    optimizer.zero_grad()
                    pct_pred, trend_logits = model(X_batch)
                    loss_reg = criterion_reg(pct_pred.squeeze(-1), y_batch_reg)
                    loss_cls = criterion_cls(trend_logits, y_batch_cls)
                    total_loss = config.LOSS_WEIGHT_REGRESSION * loss_reg + config.LOSS_WEIGHT_CLASSIFICATION * loss_cls
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_train_loss += total_loss.item()
                avg_train_loss = total_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_batch_val, y_batch_val_reg, y_batch_val_cls in val_loader:
                        X_batch_val, y_batch_val_reg, y_batch_val_cls = X_batch_val.to(config.DEVICE), y_batch_val_reg.to(config.DEVICE), y_batch_val_cls.to(config.DEVICE)
                        pct_pred_val, trend_logits_val = model(X_batch_val)
                        val_loss_reg = criterion_reg(pct_pred_val.squeeze(-1), y_batch_val_reg)
                        val_loss_cls = criterion_cls(trend_logits_val, y_batch_val_cls)
                        val_loss_total = config.LOSS_WEIGHT_REGRESSION * val_loss_reg + config.LOSS_WEIGHT_CLASSIFICATION * val_loss_cls
                        total_val_loss += val_loss_total.item()
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                scheduler.step(avg_val_loss)

                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d}/{config.EPOCHS} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= config.PATIENCE:
                        print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                        break
            
            print(f"--- Training for {ticker} completed in {time.time() - start_time:.2f}s ---")

            # 4. Evaluation, Backtest, and Saving
            if best_model_wts:
                model.load_state_dict(best_model_wts)

                model.eval()
                all_val_errors_scaled = []
                with torch.no_grad():
                    for X_b, y_b_reg, _ in val_loader:
                        X_b, y_b_reg = X_b.to(config.DEVICE), y_b_reg.to(config.DEVICE)
                        pct_pred_val, _ = model(X_b)
                        errors_scaled = y_b_reg - pct_pred_val.squeeze(-1)
                        all_val_errors_scaled.append(errors_scaled.cpu())
                error_std = torch.std(torch.cat(all_val_errors_scaled)).item()
                
                ticker_metrics = evaluate_and_collect_results(
                    ticker, model, test_loader, config, scaler_y_pct,
                    last_p_test, feature_list, error_std, test_dates
                )
                
                # --- THU THẬP KẾT QUẢ VÀO CÁC LIST TỔNG HỢP ---
                all_regression_metrics.append({
                    'Ticker': ticker,
                    'RMSE': ticker_metrics['rmse'],
                    'MAE': ticker_metrics['mae'],
                    'R2': ticker_metrics['r2']
                })
                
                report = ticker_metrics.get('class_report_dict', {})
                up_metrics = report.get('Up', {})
                down_metrics = report.get('Down', {})
                neutral_metrics = report.get('Neutral', {})
                all_classification_metrics.append({
                    'Ticker': ticker,
                    'Accuracy': ticker_metrics['accuracy'],
                    'MCC': ticker_metrics['mcc'],
                    'Precision_Up': up_metrics.get('precision'), 'Recall_Up': up_metrics.get('recall'), 'F1_Score_Up': up_metrics.get('f1-score'),
                    'Precision_Down': down_metrics.get('precision'), 'Recall_Down': down_metrics.get('recall'), 'F1_Score_Down': down_metrics.get('f1-score'),
                    'Precision_Neutral': neutral_metrics.get('precision'), 'Recall_Neutral': neutral_metrics.get('recall'), 'F1_Score_Neutral': neutral_metrics.get('f1-score')
                })

                backtest_performance = run_backtest_simulation(ticker_metrics, config)
                if backtest_performance:
                    all_backtest_performances.append(backtest_performance)

                # Lưu các file riêng của ticker
                loss_history = {'train_loss': train_losses, 'val_loss': val_losses}
                save_ticker_specific_files(ticker_metrics, loss_history, config)
                save_model_artifacts(ticker, model, scaler_X, scaler_y_pct, feature_list, config)

            else:
                print(f"No best model found for {ticker} after training.")

        except Exception as e:
            print(f"!!!!!! AN ERROR OCCURRED WHILE PROCESSING TICKER {ticker}: {e} !!!!!")
            traceback.print_exc()
            continue

    # --- LƯU CÁC FILE TỔNG HỢP SAU KHI VÒNG LẶP KẾT THÚC ---
    print("\n" + "="*80 + "\n### SAVING OVERALL RESULT FILES ###\n" + "="*80)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    if all_regression_metrics:
        df_reg = pd.DataFrame(all_regression_metrics)
        df_reg.insert(0, 'STT', range(1, len(df_reg) + 1))
        reg_path = os.path.join(results_dir, 'all_regression_metrics.csv')
        df_reg.to_csv(reg_path, index=False, float_format='%.4f')
        print(f"Overall regression metrics saved to '{reg_path}'")

    if all_classification_metrics:
        df_cls = pd.DataFrame(all_classification_metrics)
        df_cls.insert(0, 'STT', range(1, len(df_cls) + 1))
        cls_path = os.path.join(results_dir, 'all_classification_metrics.csv')
        df_cls.to_csv(cls_path, index=False, float_format='%.4f')
        print(f"Overall classification metrics saved to '{cls_path}'")

    if all_backtest_performances:
        df_perf = pd.DataFrame(all_backtest_performances)
        df_perf.insert(0, 'STT', range(1, len(df_perf) + 1))
        perf_path = os.path.join(results_dir, 'all_strategy_performance.csv')
        df_perf.to_csv(perf_path, index=False, float_format='%.4f')
        print(f"Overall strategy performance saved to '{perf_path}'")


if __name__ == '__main__':
    main()