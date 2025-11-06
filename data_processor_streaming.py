import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import cudf
    import dask_cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False

from config import Config

class StreamingDataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.use_gpu = config.USE_GPU and CUDF_AVAILABLE
        self.feature_stats = {}
        self.stats_computed = False
        print(f"StreamingDataProcessor initialized with GPU: {self.use_gpu}")
    
    def load_parquet_lazy(self, day_num: int):
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        if not os.path.exists(parquet_path):
            return None
        try:
            if self.use_gpu:
                ddf = dask_cudf.read_parquet(parquet_path)
            else:
                ddf = dd.read_parquet(parquet_path)
            return ddf
        except Exception as e:
            print(f"Error loading day{day_num}: {e}")
            return None
    
    def filter_stable_period(self, ddf):
        stable_time = self.config.STABLE_START_TIME
        if self.use_gpu:
            stable_dt = cudf.to_datetime(stable_time, format='%H:%M:%S')
        else:
            import pandas as pd
            stable_dt = pd.to_datetime(stable_time, format='%H:%M:%S')
        ddf_filtered = ddf[ddf['Time'] >= stable_dt]
        return ddf_filtered
    
    def compute_feature_statistics(self, day_numbers: List[int], sample_rate: float = 0.1):
        print("Computing feature statistics...")
        feature_cols = self.config.get_feature_columns()
        sums, sum_squares, counts = {}, {}, {}
        for col in feature_cols:
            sums[col] = 0.0
            sum_squares[col] = 0.0
            counts[col] = 0
        for day_num in day_numbers:
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            ddf = self.filter_stable_period(ddf)
            if sample_rate < 1.0:
                ddf = ddf.sample(frac=sample_rate)
            for col in feature_cols:
                if col in ddf.columns:
                    col_data = ddf[col].compute()
                    if self.use_gpu:
                        col_data = col_data.replace([np.inf, -np.inf], None)
                        col_data = col_data.dropna()
                    else:
                        col_data = col_data.replace([np.inf, -np.inf], np.nan)
                        col_data = col_data.dropna()
                    if len(col_data) > 0:
                        sums[col] += col_data.sum()
                        sum_squares[col] += (col_data ** 2).sum()
                        counts[col] += len(col_data)
            del ddf
            gc.collect()
        self.feature_stats = {}
        for col in feature_cols:
            if counts[col] > 0:
                mean = sums[col] / counts[col]
                variance = (sum_squares[col] / counts[col]) - (mean ** 2)
                std = np.sqrt(max(variance, 0)) + 1e-8
                self.feature_stats[col] = {'mean': float(mean), 'std': float(std)}
            else:
                self.feature_stats[col] = {'mean': 0.0, 'std': 1.0}
        self.stats_computed = True
        print(f"Statistics computed for {len(feature_cols)} features")
    
    def normalize_partition(self, partition):
        feature_cols = self.config.get_feature_columns()
        for col in feature_cols:
            if col in partition.columns and col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                partition[col] = (partition[col] - mean) / std
                if self.use_gpu:
                    partition[col] = partition[col].replace([np.inf, -np.inf], 0)
                    partition[col] = partition[col].fillna(0)
                else:
                    partition[col] = partition[col].replace([np.inf, -np.inf], np.nan)
                    partition[col] = partition[col].fillna(0)
                partition[col] = partition[col].clip(-5, 5)
        return partition
    
    def create_target_variable_partition(self, partition):
        horizon = self.config.PREDICTION_HORIZON
        for h in range(1, horizon + 1):
            partition[f'future_return_{h}'] = partition['Price'].pct_change(h).shift(-h)
        return_cols = [f'future_return_{h}' for h in range(1, horizon + 1)]
        partition['target_return'] = partition[return_cols].mean(axis=1)
        if self.use_gpu:
            partition['target_return'] = partition['target_return'].replace([np.inf, -np.inf], 0)
            partition['target_return'] = partition['target_return'].fillna(0)
        else:
            partition['target_return'] = partition['target_return'].replace([np.inf, -np.inf], np.nan)
            partition['target_return'] = partition['target_return'].fillna(0)
        partition['target_direction'] = 0
        partition.loc[partition['target_return'] > 0.0001, 'target_direction'] = 1
        partition.loc[partition['target_return'] < -0.0001, 'target_direction'] = -1
        partition = partition.drop(columns=return_cols)
        return partition
    
    def process_and_save_sequences(self, day_numbers: List[int], output_prefix: str):
        print(f"Processing days {day_numbers[0]} to {day_numbers[-1]}...")
        feature_cols = self.config.get_feature_columns()
        lookback = self.config.LOOKBACK_WINDOW
        horizon = self.config.PREDICTION_HORIZON
        total_sequences = 0
        max_sequences_estimate = len(day_numbers) * 15000
        X_memmap_path = f"{self.config.CACHE_DIR}/{output_prefix}_X.dat"
        y_memmap_path = f"{self.config.CACHE_DIR}/{output_prefix}_y.dat"
        X_memmap = np.memmap(X_memmap_path, dtype='float32', mode='w+', shape=(max_sequences_estimate, lookback, len(feature_cols)))
        y_memmap = np.memmap(y_memmap_path, dtype='int8', mode='w+', shape=(max_sequences_estimate,))
        sequence_idx = 0
        for day_num in day_numbers:
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            ddf = self.filter_stable_period(ddf)
            ddf = ddf.map_partitions(self.normalize_partition)
            ddf = ddf.map_partitions(self.create_target_variable_partition)
            df = ddf.compute()
            if self.use_gpu:
                df_features = df[feature_cols].to_numpy()
                df_targets = df['target_direction'].to_numpy()
            else:
                df_features = df[feature_cols].values
                df_targets = df['target_direction'].values
            for i in range(lookback, len(df) - horizon):
                seq = df_features[i - lookback:i]
                target = df_targets[i]
                if not np.isnan(target) and not np.isnan(seq).any():
                    X_memmap[sequence_idx] = seq
                    y_memmap[sequence_idx] = target
                    sequence_idx += 1
                    if sequence_idx >= max_sequences_estimate:
                        print("WARNING: Reached maximum sequence estimate!")
                        break
            del df, ddf
            gc.collect()
            print(f"  day{day_num}: {sequence_idx} total sequences")
        X_final = X_memmap[:sequence_idx]
        y_final = y_memmap[:sequence_idx]
        np.save(f"{self.config.MODEL_DIR}/{output_prefix}_X.npy", X_final)
        np.save(f"{self.config.MODEL_DIR}/{output_prefix}_y.npy", y_final)
        del X_memmap, y_memmap
        os.remove(X_memmap_path)
        os.remove(y_memmap_path)
        print(f"Saved {sequence_idx} sequences to {output_prefix}")
        return sequence_idx
    
    def prepare_training_data(self, train_days: List[int], val_days: List[int], test_days: List[int]):
        print("="*60)
        print("PREPARING TRAINING DATA")
        print("="*60)
        if not self.stats_computed:
            self.compute_feature_statistics(train_days, sample_rate=0.2)
        print("\n1. Processing training data...")
        train_count = self.process_and_save_sequences(train_days, "train")
        print("\n2. Processing validation data...")
        val_count = self.process_and_save_sequences(val_days, "val")
        print("\n3. Processing test data...")
        test_count = self.process_and_save_sequences(test_days, "test")
        print("DATA PREPARATION COMPLETE")
        print(f"Training sequences: {train_count}")
        print(f"Validation sequences: {val_count}")
        print(f"Test sequences: {test_count}")
        return train_count, val_count, test_count

def main():
    config = Config()
    processor = StreamingDataProcessor(config)
    train_days = list(range(0, 10))
    val_days = list(range(10, 12))
    test_days = list(range(12, 15))
    processor.prepare_training_data(train_days, val_days, test_days)

if __name__ == "__main__":
    main()
