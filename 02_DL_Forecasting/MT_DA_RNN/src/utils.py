
# src/utils.py
import pandas as pd
import numpy as np
import torch
import os
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    matthews_corrcoef # add for MCC
)

def evaluate_and_collect_results(ticker, model, loader, config, scaler_y_pct, last_close_prices, feature_list, error_std, test_dates):
    print("\n" + "="*60 + f"\n--- EVALUATING MODEL ON TEST SET ({ticker}) ---\n" + "="*60)
    model.eval()
    all_preds_reg, all_true_reg, all_preds_cls, all_true_cls = [], [], [], []
    all_input_attentions, all_temporal_attentions = [], []
    with torch.no_grad():
        for X_batch, y_batch_reg, y_batch_cls in loader:
            X_batch, y_batch_reg, y_batch_cls = X_batch.to(config.DEVICE), y_batch_reg.to(config.DEVICE), y_batch_cls.to(config.DEVICE)
            pct_pred, trend_logits, input_attns, temporal_attns = model(X_batch, return_attentions=True)
            trend_pred_labels = torch.argmax(trend_logits, dim=1)
            all_preds_reg.append(pct_pred.cpu().numpy()); all_true_reg.append(y_batch_reg.cpu().numpy())
            all_preds_cls.append(trend_pred_labels.cpu().numpy()); all_true_cls.append(y_batch_cls.cpu().numpy())
            all_input_attentions.append(input_attns.cpu().numpy()); all_temporal_attentions.append(temporal_attns.cpu().numpy())
            
    preds_reg_scaled = np.concatenate(all_preds_reg).flatten(); true_reg_scaled = np.concatenate(all_true_reg).flatten()
    upper_bound_scaled = preds_reg_scaled + 1.96 * error_std; lower_bound_scaled = preds_reg_scaled - 1.96 * error_std
    preds_pct = scaler_y_pct.inverse_transform(preds_reg_scaled.reshape(-1, 1)).flatten(); true_pct = scaler_y_pct.inverse_transform(true_reg_scaled.reshape(-1, 1)).flatten()
    upper_pct = scaler_y_pct.inverse_transform(upper_bound_scaled.reshape(-1, 1)).flatten(); lower_pct = scaler_y_pct.inverse_transform(lower_bound_scaled.reshape(-1, 1)).flatten()
    true_prices = last_close_prices * (1 + true_pct); preds_prices = last_close_prices * (1 + preds_pct)
    upper_prices = last_close_prices * (1 + upper_pct); lower_prices = last_close_prices * (1 + lower_pct)
    
    rmse = np.sqrt(mean_squared_error(true_prices, preds_prices)); mae = mean_absolute_error(true_prices, preds_prices); r2 = r2_score(true_prices, preds_prices)
    print(f"Regression Metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.4f}")

    preds_cls = np.concatenate(all_preds_cls); true_cls = np.concatenate(all_true_cls)
    mcc = matthews_corrcoef(true_cls, preds_cls) # calculate MCC
    class_report_dict = classification_report(true_cls, preds_cls, target_names=['Down', 'Neutral', 'Up'], zero_division=0, digits=4, output_dict=True)
    print(f"Classification Metrics:\n{classification_report(true_cls, preds_cls, target_names=['Down', 'Neutral', 'Up'], zero_division=0, digits=4)}")
    print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")
    
    accuracy = class_report_dict.get('accuracy', 0)
    conf_matrix = confusion_matrix(true_cls, preds_cls)
    
    avg_temporal_weights = np.mean(np.concatenate(all_temporal_attentions, axis=0), axis=0) if all_temporal_attentions else None
    avg_input_attentions = np.mean(np.concatenate(all_input_attentions, axis=0), axis=0) if all_input_attentions else None

    return {
        'ticker': ticker, 'rmse': rmse, 'mae': mae, 'r2': r2, 'accuracy': accuracy, 'mcc': mcc,
        'class_report_dict': class_report_dict, 'confusion_matrix': conf_matrix, 'preds_prices': preds_prices,
        'true_prices': true_prices, 'upper_prices': upper_prices, 'lower_prices': lower_prices,
        'avg_temporal_attentions': avg_temporal_weights, 'avg_input_attentions': avg_input_attentions,
        'feature_list': feature_list, 'config': config, 'test_dates': test_dates, 'preds_cls': preds_cls,
        'true_prices_for_trading': last_close_prices
    }

def calculate_mdd(portfolio_values):
    if len(portfolio_values) < 2: return 0.0
    peak = portfolio_values[0]; max_drawdown = 0.0
    for value in portfolio_values:
        if value > peak: peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown: max_drawdown = drawdown
    return max_drawdown * 100

def calculate_sharpe_ratio(portfolio_values, trading_days_per_year=252):
    if len(portfolio_values) < 2: return 0.0
    portfolio_values = pd.Series(portfolio_values)
    daily_returns = portfolio_values.pct_change().dropna()
    if daily_returns.std() == 0 or len(daily_returns) == 0: return 0.0
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trading_days_per_year)
    return sharpe_ratio

def run_backtest_simulation(ticker_results, config):
    print(f"\n--- Running Backtest Simulation for {ticker_results['ticker']} (with Transaction Fee) ---")
    try: initial_capital, fee_pct = config.INITIAL_CAPITAL, config.TRANSACTION_FEE_PCT
    except AttributeError: print("Warning: Backtest parameters not in config. Using default values."); initial_capital, fee_pct = 100_000_000, 0.0015
    trade_prices = ticker_results['true_prices_for_trading']
    if len(trade_prices) < 2: print("Not enough data for backtest simulation."); return None
    final_price, predicted_trends = ticker_results['true_prices'][-1], ticker_results['preds_cls']
    bnh_portfolio_values, model_portfolio_values = [initial_capital], [initial_capital]
    buy_fee_bnh = initial_capital * fee_pct; capital_after_buy_fee = initial_capital - buy_fee_bnh; bnh_shares = capital_after_buy_fee / trade_prices[0]
    for i in range(len(trade_prices)): bnh_portfolio_values.append(bnh_shares * trade_prices[i])
    final_sale_value_bnh = bnh_shares * final_price; sell_fee_bnh = final_sale_value_bnh * fee_pct; bnh_final_capital = final_sale_value_bnh - sell_fee_bnh
    model_capital, model_shares, num_trades = initial_capital, 0, 0
    for i in range(len(trade_prices) - 1):
        today_price, predicted_trend_for_tomorrow = trade_prices[i], predicted_trends[i]
        if predicted_trend_for_tomorrow == 2 and model_capital > 1: buy_fee = model_capital * fee_pct; model_shares = (model_capital - buy_fee) / today_price; model_capital = 0; num_trades += 1
        elif predicted_trend_for_tomorrow == 0 and model_shares > 0: sell_value = model_shares * today_price; sell_fee = sell_value * fee_pct; model_capital = sell_value - sell_fee; model_shares = 0; num_trades += 1
        model_portfolio_values.append(model_capital + (model_shares * today_price))
    if model_shares > 0: final_sale_value_model = model_shares * final_price; sell_fee_model = final_sale_value_model * fee_pct; model_final_capital = final_sale_value_model - sell_fee_model; num_trades += 1
    else: model_final_capital = model_capital
    model_portfolio_values.append(model_final_capital)
    bnh_return_pct, bnh_mdd, bnh_sharpe = (bnh_final_capital / initial_capital - 1) * 100, calculate_mdd(bnh_portfolio_values), calculate_sharpe_ratio(bnh_portfolio_values)
    model_return_pct, model_mdd, model_sharpe = (model_final_capital / initial_capital - 1) * 100, calculate_mdd(model_portfolio_values), calculate_sharpe_ratio(model_portfolio_values)
    print("\n" + "-"*25 + " STRATEGY PERFORMANCE " + "-"*25); print(f"{'Metric':<25} | {'Buy and Hold':<20} | {'Model Strategy'}"); print("-"*70); print(f"{'Final Capital (VND)':<25} | {bnh_final_capital:,.2f}"); print(f"{'':<25} | {'':<20} | {model_final_capital:,.2f}"); print(f"{'Net Return (%)':<25} | {bnh_return_pct:.2f}%"); print(f"{'':<25} | {'':<20} | {model_return_pct:.2f}%"); print(f"{'Max Drawdown (%)':<25} | {bnh_mdd:.2f}%"); print(f"{'':<25} | {'':<20} | {model_mdd:.2f}%"); print(f"{'Sharpe Ratio':<25} | {bnh_sharpe:.2f}"); print(f"{'':<25} | {'':<20} | {model_sharpe:.2f}"); print(f"{'Number of Trades':<25} | 2"); print(f"{'':<25} | {'':<20} | {num_trades}"); print("-"*70 + "\n")
    return {'ticker': ticker_results['ticker'], 'bnh_return_pct': bnh_return_pct, 'model_return_pct': model_return_pct, 'bnh_final_value': bnh_final_capital, 'model_final_value': model_final_capital, 'bnh_max_drawdown': bnh_mdd, 'model_max_drawdown': model_mdd, 'bnh_sharpe_ratio': bnh_sharpe, 'model_sharpe_ratio': model_sharpe, 'model_num_trades': num_trades}


def save_ticker_specific_files(ticker_results, loss_history, config):
    """Lưu các file riêng cho từng ticker (biểu đồ và giá dự báo)."""
    ticker = ticker_results['ticker']
    safe_ticker = ticker.replace('.VN','_VN').replace('^','')
    save_dir = os.path.join("results", "by_ticker", safe_ticker)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n--- Saving charts and price data for {ticker} to {save_dir} ---")

    # 1. Lưu file 03_predict_prices.csv
    prices_df = pd.DataFrame({
        'Date': ticker_results['test_dates'], 'Ticker': ticker,
        'Actual_Price': ticker_results['true_prices'],
        'Predicted_Price': ticker_results['preds_prices'],
        'Lower_CI_Price': ticker_results['lower_prices'],
        'Upper_CI_Price': ticker_results['upper_prices']
    })
    prices_df.to_csv(os.path.join(save_dir, '03_predict_prices.csv'), index=False, float_format='%.4f')

    # 2. Vẽ và lưu các biểu đồ (đã bỏ logic tạo file 04_attention)
    plt.style.use('seaborn-whitegrid')
    # Chart 01: Price Prediction
    fig1 = plt.figure(figsize=(15, 7)); plt.plot(ticker_results['test_dates'], ticker_results['true_prices'], label='Actual Prices', color='royalblue', linewidth=2); plt.plot(ticker_results['test_dates'], ticker_results['preds_prices'], label='Predicted Prices', color='darkorange', linestyle='--'); plt.fill_between(ticker_results['test_dates'], ticker_results['lower_prices'], ticker_results['upper_prices'], color='orange', alpha=0.2, label='95% Confidence Band'); plt.title(f'Actual vs. Predicted Prices - {ticker}', fontsize=16); plt.xlabel('Date'); plt.ylabel('Stock Price'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'chart_01_price_prediction.png'), dpi=300); plt.close(fig1)
    
    # Chart 02: Loss Curve
    fig2 = plt.figure(figsize=(10, 5)); plt.plot(loss_history['train_loss'], label='Training Loss'); plt.plot(loss_history['val_loss'], label='Validation Loss'); plt.title(f'Training & Validation Loss - {ticker}', fontsize=14); plt.xlabel('Epoch'); plt.ylabel('Total Loss'); plt.legend(); plt.grid(True); plt.savefig(os.path.join(save_dir, 'chart_02_loss_curve.png'), dpi=300); plt.close(fig2)
    
    # Chart 03: Confusion Matrix
    fig3 = plt.figure(figsize=(6, 5)); sns.heatmap(ticker_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Neutral', 'Up'], yticklabels=['Down', 'Neutral', 'Up']); plt.title(f'Confusion Matrix - {ticker}', fontsize=14); plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.savefig(os.path.join(save_dir, 'chart_03_confusion_matrix.png'), dpi=300, bbox_inches='tight'); plt.close(fig3)

    # Chart 04: Temporal Attention
    if ticker_results['avg_temporal_attentions'] is not None:
        fig4 = plt.figure(figsize=(15, 4)); T = ticker_results['config'].T; time_labels = [f'T-{T-i-1}' for i in range(T)]; plt.bar(time_labels, ticker_results['avg_temporal_attentions'], color='lightcoral'); plt.title(f'Average Temporal Attention Weights - {ticker}', fontsize=14); plt.xlabel('Past Time Step'); plt.ylabel('Average Weight'); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.savefig(os.path.join(save_dir, 'chart_04_temporal_attention.png'), dpi=300); plt.close(fig4)

    # Chart 05: Input Attention (đã sắp xếp)
    if ticker_results['avg_input_attentions'] is not None:
        # Tạo DataFrame để dễ dàng sắp xếp
        input_attn_df = pd.DataFrame({
            'Feature': ticker_results['feature_list'],
            'Weight': ticker_results['avg_input_attentions'].mean(axis=0) # Lấy trung bình theo seq_len
        }).sort_values(by='Weight', ascending=False)
        
        fig5 = plt.figure(figsize=(15, 8)); sns.barplot(x='Weight', y='Feature', data=input_attn_df, palette='viridis', orient='h'); plt.title(f'Average Input Attention Weights - {ticker}', fontsize=14); plt.xlabel('Average Attention Weight'); plt.ylabel('Feature'); plt.tight_layout(); plt.savefig(os.path.join(save_dir, 'chart_05_input_attention.png'), dpi=300); plt.close(fig5)
        
    print(f"--- Files for {ticker} saved successfully. ---")

def save_model_artifacts(ticker, model, scaler_X, scaler_y_pct, feature_list, config):
    safe_ticker = ticker.replace('.VN','_VN').replace('^','')
    save_dir = os.path.join(config.MODEL_SAVE_PATH, "by_ticker", safe_ticker) 
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
    with open(os.path.join(save_dir, "scaler_X.pkl"), 'wb') as f: pickle.dump(scaler_X, f)
    with open(os.path.join(save_dir, "scaler_y_pct.pkl"), 'wb') as f: pickle.dump(scaler_y_pct, f)
    with open(os.path.join(save_dir, "features.json"), 'w') as f: json.dump({'features': feature_list}, f)
    print(f"--- Model artifacts for {ticker} saved to {save_dir} ---")