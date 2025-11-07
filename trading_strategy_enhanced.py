import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    CUDF_AVAILABLE = False


class EnhancedTFTTradingStrategy:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self.position = 0
        self.entry_price = 0
        self.entry_time = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.signal_buffer = deque(maxlen=config.SIGNAL_SMOOTHING_WINDOW)
        self.trades = []
        self.current_day_trades = []
        self.total_predictions = 0
        self.confident_predictions = 0

    def predict(self, sequence: np.ndarray) -> Tuple[int, float, np.ndarray]:
        with torch.no_grad():
            seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            output = self.model(seq_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
            predicted_class = np.argmax(probabilities) - 1
            confidence = probabilities[np.argmax(probabilities)]
            self.total_predictions += 1
            return predicted_class, confidence, probabilities

    def smooth_signal(self, prediction: int, confidence: float) -> Tuple[int, float]:
        self.signal_buffer.append((prediction, confidence))
        if len(self.signal_buffer) < self.config.SIGNAL_SMOOTHING_WINDOW:
            return 0, 0.0
        weighted_sum = 0.0
        total_confidence = 0.0
        for sig, conf in self.signal_buffer:
            weighted_sum += sig * conf
            total_confidence += conf
        if total_confidence == 0:
            return 0, 0.0
        avg_signal = weighted_sum / total_confidence
        avg_confidence = total_confidence / len(self.signal_buffer)
        if avg_signal > 0.3:
            smoothed_signal = 1
        elif avg_signal < -0.3:
            smoothed_signal = -1
        else:
            smoothed_signal = 0
        return smoothed_signal, avg_confidence

    def should_enter_trade(self, prediction: int, confidence: float, current_price: float, current_time: int) -> bool:
        if self.position != 0:
            return False
        if confidence < self.config.MIN_PREDICTION_CONFIDENCE:
            return False
        if prediction == 0:
            return False
        self.confident_predictions += 1
        return True

    def should_exit_trade(self, current_price: float, current_time: int) -> Tuple[bool, str]:
        if self.position == 0:
            return False, ""
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price:
            self.lowest_price = current_price
        pnl_pct = (current_price - self.entry_price) / self.entry_price * self.position
        if pnl_pct >= self.config.PROFIT_THRESHOLD:
            return True, "PROFIT_TARGET"
        if pnl_pct <= -self.config.STOP_LOSS:
            return True, "STOP_LOSS"
        if self.position == 1:
            pullback = (self.highest_price - current_price) / self.highest_price
            if pullback >= self.config.TRAILING_STOP_PCT:
                return True, "TRAILING_STOP"
        elif self.position == -1:
            pullback = (current_price - self.lowest_price) / self.lowest_price
            if pullback >= self.config.TRAILING_STOP_PCT:
                return True, "TRAILING_STOP"
        if self.entry_time is not None:
            holding_time = (current_time - self.entry_time) if self.entry_time else 0
            if holding_time >= self.config.MAX_HOLDING_TIME:
                return True, "MAX_HOLDING_TIME"
        return False, ""

    def execute_trade(self, action: str, price: float, timestamp: int, reason: str = ""):
        if action == 'BUY':
            self.position = 1
            self.entry_price = price
            self.entry_time = timestamp
            self.highest_price = price
            self.lowest_price = price
        elif action == 'SELL':
            self.position = -1
            self.entry_price = price
            self.entry_time = timestamp
            self.highest_price = price
            self.lowest_price = price
        elif action == 'CLOSE':
            if self.position != 0:
                pnl = (price - self.entry_price) * self.position
                pnl_pct = pnl / self.entry_price
                holding_time = (current_time - self.entry_time) if self.entry_time else 0
                trade_record = {
                    'entry_index': self.entry_time,
                    'exit_index': timestamp,
                    'entry_price': self.entry_price,
                    'exit_price': price,
                    'position': self.position,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'holding_time': holding_time,
                    'exit_reason': reason
                }
                self.trades.append(trade_record)
                self.current_day_trades.append(trade_record)
                self.position = 0
                self.entry_price = 0
                self.entry_time = None
                self.highest_price = 0
                self.lowest_price = float('inf')

    def run_day(self, df, day_num: int = None) -> List[Dict]:
        self.current_day_trades = []
        lookback = self.config.LOOKBACK_WINDOW
        feature_cols = self.config.get_feature_columns()
        available_cols = [col for col in feature_cols if col in df.columns]
        if CUDF_AVAILABLE and isinstance(df, cudf.DataFrame):
            df_features = df[available_cols].to_numpy()
            df_prices = df['Price'].to_numpy()
        else:
            df_features = df[available_cols].values
            df_prices = df['Price'].values
        for i in range(lookback, len(df)):
            sequence = df_features[i - lookback:i]
            current_price = float(df_prices[i])
            current_time = i
            prediction, confidence, probs = self.predict(sequence)
            smoothed_pred, smoothed_conf = self.smooth_signal(prediction, confidence)
            should_exit, exit_reason = self.should_exit_trade(current_price, current_time)
            if should_exit:
                self.execute_trade('CLOSE', current_price, current_time, exit_reason)
            elif self.should_enter_trade(smoothed_pred, smoothed_conf, current_price, current_time):
                if smoothed_pred == 1:
                    self.execute_trade('BUY', current_price, current_time)
                elif smoothed_pred == -1:
                    self.execute_trade('SELL', current_price, current_time)
        if self.position != 0:
            final_price = float(df_prices[-1])
            final_time = len(df) - 1
            self.execute_trade('CLOSE', final_price, final_time, "END_OF_DAY")
        self.signal_buffer.clear()
        return self.current_day_trades

    def get_statistics(self) -> Dict:
        return {
            'total_predictions': self.total_predictions,
            'confident_predictions': self.confident_predictions,
            'confidence_rate': self.confident_predictions / max(self.total_predictions, 1),
            'total_trades': len(self.trades)
        }


class PerformanceEvaluator:
    def __init__(self, config):
        self.config = config

    def calculate_metrics(self, trades: List[Dict], days_traded: int) -> Dict:
        if not trades:
            return self._empty_metrics()
        returns = np.array([t['pnl_pct'] for t in trades])
        holding_times = np.array([t['holding_time'] for t in trades])
        cumulative_returns = (1 + returns).cumprod() - 1
        total_return = cumulative_returns[-1]
        annual_return = (1 + total_return) ** (self.config.TRADING_DAYS_PER_YEAR / days_traded) - 1
        cumulative_wealth = 1 + cumulative_returns
        running_max = np.maximum.accumulate(cumulative_wealth)
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        sharpe_ratio = self._calculate_sharpe(returns, self.config.TRADING_DAYS_PER_YEAR)
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        win_rate = len(winning_trades) / len(trades)
        avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
        total_wins = sum(t['pnl'] for t in winning_trades)
        total_losses = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'UNKNOWN')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        results = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_time': holding_times.mean(),
            'max_holding_time': holding_times.max(),
            'exit_reasons': exit_reasons,
            'days_traded': days_traded
        }
        return results

    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int) -> float:
        if len(returns) == 0:
            return 0
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0:
            return 0
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
        return sharpe

    def _empty_metrics(self) -> Dict:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'annual_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'calmar_ratio': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_holding_time': 0,
            'max_holding_time': 0,
            'exit_reasons': {},
            'days_traded': 0
        }

    def print_results(self, results: Dict):
        print("\n")
        print("TRADING PERFORMANCE REPORT")
        print("\nTRADE STATISTICS")
        print(f"Total Trades:          {results['total_trades']:,}")
        print(f"Winning Trades:        {results['winning_trades']:,} ({results['winning_trades']/max(results['total_trades'],1)*100:.1f}%)")
        print(f"Losing Trades:         {results['losing_trades']:,} ({results['losing_trades']/max(results['total_trades'],1)*100:.1f}%)")
        print(f"Win Rate:              {results['win_rate']*100:.2f}%")
        print("\nRETURNS")
        print(f"Total Return:          {results['total_return']*100:+.2f}%")
        print(f"Annualized Return:     {results['annual_return']*100:+.2f}%")
        print(f"Average Win:           {results['avg_win']*100:+.2f}%")
        print(f"Average Loss:          {results['avg_loss']*100:.2f}%")
        print("\nRISK METRICS")
        print(f"Maximum Drawdown:      {results['max_drawdown']*100:.2f}%")
        print(f"Sharpe Ratio:          {results['sharpe_ratio']:.3f}")
        print(f"Calmar Ratio:          {results['calmar_ratio']:.3f}")
        print(f"Profit Factor:         {results['profit_factor']:.2f}")
        print("\nHOLDING TIMES")
        print(f"Average:               {results['avg_holding_time']:.1f} seconds")
        print(f"Maximum:               {results['max_holding_time']:.1f} seconds")
        print("\nEXIT REASONS")
        for reason, count in results['exit_reasons'].items():
            pct = count / results['total_trades'] * 100
            print(f"{reason:20s}: {count:4d} ({pct:5.1f}%)")
        print("COMPETITION REQUIREMENTS")
        meets_return = results['annual_return'] >= self.config.MIN_ANNUAL_RETURN
        meets_drawdown = results['max_drawdown'] <= self.config.MAX_DRAWDOWN
        print(f"Annualized Return ≥ 20%:  {'PASS' if meets_return else 'FAIL'} ({results['annual_return']*100:.2f}%)")
        print(f"Max Drawdown ≤ 10%:       {'PASS' if meets_drawdown else 'FAIL'} ({results['max_drawdown']*100:.2f}%)")
        if meets_return and meets_drawdown:
            print("\nStrategy is simply lovely.")
        else:
            print("\nStrategy is not that good")
