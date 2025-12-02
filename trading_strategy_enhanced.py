import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from config import Config

class EnhancedTFTTradingStrategy:
    def __init__(self, model, config: Config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        
        # --- STRATEGY PARAMETERS ---
        # Minimum expected return to enter a trade (e.g., 0.05%)
        self.min_profit_threshold = 0.05 
        
        self.stop_loss_pct = 0.002    # 0.2% hard stop
        self.take_profit_pct = 0.004  # 0.4% take profit
        self.position = 0             # Current position: 0 (flat), 1 (long), -1 (short)
        self.entry_price = 0.0
        self.entry_step = 0
        self.entry_time = None        # Initialize entry_time
        self.last_pred_return = 0.0   # Initialize last_pred_return

        self.last_p10 = 0.0  # <--- NEW
        self.last_p50 = 0.0  # <--- NEW
        self.last_p90 = 0.0  # <--- NEW
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'long_signals': 0,
            'short_signals': 0,
            'avg_predicted_return': 0.0
        }

    def get_signal(self, sequences: torch.Tensor) -> Tuple[int, float]:
        """
        Input: sequences [batch_size, lookback, features]
        Output: signal (-1, 0, 1), predicted_return (float)
        """
        with torch.no_grad():
            sequences = sequences.to(self.device)
            
            # Model outputs: [batch_size, 3] -> (P10, P50, P90)
            predictions = self.model(sequences)
            
            # Take the last prediction in the batch
            latest_pred = predictions[-1].cpu().numpy()

            
            p10 = latest_pred[0] # 10th percentile (Conservative Lower Bound)
            p50 = latest_pred[1] # 50th percentile (Median/Expected Return)
            p90 = latest_pred[2] # 90th percentile (Conservative Upper Bound)

            self.last_p10 = float(p10)
            self.last_p50 = float(p50)
            self.last_p90 = float(p90)
            
            self.stats['total_predictions'] += 1
            self.stats['avg_predicted_return'] += abs(p50)

            # --- QUANTILE STRATEGY LOGIC ---
            
            # LONG CONDITION:
            # We are 90% confident that the return will be > min_profit_threshold
            if p10 > self.min_profit_threshold:
                self.stats['long_signals'] += 1
                return 1, float(p50)
            
            # SHORT CONDITION:
            # We are 90% confident that the return will be < -min_profit_threshold
            elif p90 < -self.min_profit_threshold:
                self.stats['short_signals'] += 1
                return -1, float(p50)
            
            # FLAT CONDITION:
            else:
                return 0, 0.0

    def run_day(self, df, day_num: int) -> List[Dict]:
        """
        Simulate trading for a single day using the dataframe.
        """
        # --- FIX: Convert to Pandas (CPU) ---
        # If input is a cuDF DataFrame (GPU), move it to CPU first.
        # This fixes 'cupy does not support object' for the string 'Time' column
        # and speeds up the row-by-row iteration below.
        if hasattr(df, 'to_pandas'):
            df = df.to_pandas()
        # ------------------------------------

        trades = []
        feature_cols = self.config.get_feature_columns()
        lookback = self.config.LOOKBACK_WINDOW
        
        # Convert features to numpy for fast slicing
        features_data = df[feature_cols].values
        price_data = df['Price'].values
        time_data = df['Time'].values
        
        # Reset daily state
        self.position = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.last_pred_return = 0.0
        
        # Need at least lookback periods to make first prediction
        if len(df) <= lookback:
            return []

        # Iterate through the day
        for i in range(lookback, len(df) - 1):
            current_price = price_data[i]
            current_time = time_data[i]
            
            # Check Exit Conditions first (if we have a position)
            if self.position != 0:
                pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
                
                # Stop Loss or Take Profit
                if pnl_pct <= -self.stop_loss_pct or pnl_pct >= self.take_profit_pct:
                    trades.append({
                        'day': day_num,
                        'entry_time': self.entry_time,
                        'exit_time': current_time,
                        'type': 'LONG' if self.position == 1 else 'SHORT',
                        'entry_price': self.entry_price,
                        'exit_price': current_price,
                        'return': pnl_pct,
                        'predicted_return': self.last_pred_return,
                        'reason': 'TP' if pnl_pct > 0 else 'SL'
                    })
                    self.position = 0
                    continue # Wait for next signal
            
            # Check Entry Conditions (only if flat)
            if self.position == 0:
                # Prepare sequence
                seq = features_data[i-lookback : i]
                # Add batch dim: (1, lookback, features)
                # Ensure float32 for model
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
                
                signal, pred_return = self.get_signal(seq_tensor)

                # Print the first few predictions of the day, and then every 2000 steps
                if i < lookback + 5 or i % 2000 == 0:
                    print(f"  [Debug] P10:{self.last_p10:+.5f} | P50:{self.last_p50:+.5f} | P90:{self.last_p90:+.5f} | Sig:{signal}")
                # ---------------------------
                
                if signal != 0:
                    self.position = signal
                    self.entry_price = current_price
                    self.entry_time = current_time
                    self.last_pred_return = pred_return
                    print(f"  >>> TRADE SIGNAL TRIGGERED: {signal} @ {current_price:.2f}")
                    
        # EOD: Close any open position
        if self.position != 0:
            final_price = price_data[-1]
            pnl_pct = (final_price - self.entry_price) / self.entry_price * self.position
            trades.append({
                'day': day_num,
                'entry_time': self.entry_time,
                'exit_time': time_data[-1],
                'type': 'LONG' if self.position == 1 else 'SHORT',
                'entry_price': self.entry_price,
                'exit_price': final_price,
                'return': pnl_pct,
                'predicted_return': self.last_pred_return,
                'reason': 'EOD'
            })
            
        return trades

    def get_statistics(self):
        avg_ret = 0
        if self.stats['total_predictions'] > 0:
            avg_ret = self.stats['avg_predicted_return'] / self.stats['total_predictions']
            
        return {
            'total_predictions': self.stats['total_predictions'],
            'long_signals': self.stats['long_signals'],
            'short_signals': self.stats['short_signals'],
            'avg_predicted_return': avg_ret
        }


class PerformanceEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def calculate_metrics(self, trades: List[Dict], num_days: int) -> Dict:
        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0,
                'annual_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
                'calmar_ratio': 0.0
            }
            
        df_trades = pd.DataFrame(trades)
        
        # Basic Metrics
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades['return'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns
        df_trades['cumulative_return'] = (1 + df_trades['return']).cumprod()
        total_return = df_trades['cumulative_return'].iloc[-1] - 1
        
        # Risk Metrics
        returns = df_trades['return'].values
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Sharpe (Annualized assuming 252 days, simplified)
        sharpe = 0.0
        if std_return > 0:
            sharpe = (avg_return / std_return) * np.sqrt(252 * total_trades / num_days)
            
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar
        calmar = 0.0
        if max_drawdown < 0:
            calmar = total_return / abs(max_drawdown)
            
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': total_return * (252 / num_days),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar
        }

    def print_results(self, results: Dict):
        print("\n" + "="*40)
        print("       BACKTEST PERFORMANCE       ")
        print("="*40)
        print(f"Total Trades:    {results['total_trades']}")
        print(f"Win Rate:        {results['win_rate']*100:.2f}%")
        print(f"Total Return:    {results['total_return']*100:.2f}%")
        print(f"Annual Return:   {results.get('annual_return', 0)*100:.2f}%")
        print(f"Max Drawdown:    {results['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:    {results['sharpe_ratio']:.2f}")
        print("="*40)