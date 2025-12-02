import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json
from tqdm import tqdm

try:
    import cudf
    import dask_cudf
    import cupy
    from cupy.lib.stride_tricks import as_strided as cp_as_strided
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False
    cupy = None
    cp_as_strided = None
    
from numpy.lib.stride_tricks import as_strided as np_as_strided
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
        ddf_filtered = ddf[ddf['Time'] >= stable_time]
        return ddf_filtered
    
    def compute_feature_statistics(self, day_numbers: List[int], sample_rate: float = 0.1):
        print("Computing feature statistics...")
        feature_cols = self.config.get_feature_columns()
        sums, sum_squares, counts = {}, {}, {}
        for col in feature_cols:
            sums[col] = 0.0
            sum_squares[col] = 0.0
            counts[col] = 0
            
        for day_num in tqdm(day_numbers, desc="Computing Stats"):
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            ddf = self.filter_stable_period(ddf)
            if sample_rate < 1.0:
                ddf = ddf.sample(frac=sample_rate)
            
            relevant_cols = [col for col in feature_cols if col in ddf.columns]
            if not relevant_cols:
                del ddf; gc.collect(); continue
                
            day_stats_df = ddf[relevant_cols].compute()
            
            for col in relevant_cols:
                col_data = day_stats_df[col]
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
                    
            del ddf, day_stats_df
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
        
        stats_path = f"{self.config.CACHE_DIR}/feature_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent=4)
        print(f"Feature statistics saved to {stats_path}")

    def process_day_for_training(self, day_num: int):
        if not self.stats_computed:
            print("ERROR: Feature stats not loaded. Call load_feature_statistics() first.")
            return None, None
            
        lookback = self.config.LOOKBACK_WINDOW
        horizon = self.config.PREDICTION_HORIZON
        feature_cols = self.config.get_feature_columns()

        ddf = self.load_parquet_lazy(day_num)
        if ddf is None:
            return None, None
        
        ddf = self.filter_stable_period(ddf)

        # 1. Create Targets FIRST (on raw prices)
        ddf = ddf.map_partitions(self.create_target_variable_partition)
        # 2. THEN Normalize Features
        ddf = ddf.map_partitions(self.normalize_partition) 

        df = ddf.compute()

        n_rows = len(df)
        if n_rows <= lookback + horizon:
            del df, ddf; gc.collect()
            return None, None

        n_sequences = n_rows - lookback - horizon
        if n_sequences <= 0:
            del df, ddf; gc.collect()
            return None, None

        X_day_valid_cpu, y_day_valid_cpu, n_valid = None, None, 0

        try:
            if self.use_gpu and CUDF_AVAILABLE and cupy is not None:
                # Get Targets (Float)
                df_targets_gpu = df['target_return'].to_cupy()
                df_features_gpu = df[feature_cols].to_cupy()
                del df, ddf; gc.collect()
                
                y_data_gpu = df_targets_gpu[lookback : n_rows - horizon]
                itemsize = df_features_gpu.itemsize
                num_features = df_features_gpu.shape[1]
                shape = (n_sequences, lookback, num_features)
                strides = (itemsize * num_features, itemsize * num_features, itemsize)
                
                X_day_gpu = cp_as_strided(df_features_gpu, shape=shape, strides=strides)

                valid_targets = ~cupy.isnan(y_data_gpu)
                valid_seqs = ~cupy.isnan(X_day_gpu).any(axis=(1, 2))
                valid_mask = valid_targets & valid_seqs
                
                X_day_valid_gpu = X_day_gpu[valid_mask]
                y_day_valid_gpu = y_data_gpu[valid_mask].astype(cupy.float32)
                
                n_valid = len(X_day_valid_gpu)
                if n_valid > 0:
                    X_day_valid_cpu = X_day_valid_gpu.get()
                    y_day_valid_cpu = y_day_valid_gpu.get()
                
                # Cleanup GPU
                del df_features_gpu, df_targets_gpu, X_day_gpu, y_data_gpu, valid_mask
                cupy.get_default_memory_pool().free_all_blocks()
            
            else:
                # CPU Fallback
                if CUDF_AVAILABLE and hasattr(df, 'to_numpy'):
                    df_features = df[feature_cols].to_numpy()
                    df_targets = df['target_return'].to_numpy()
                else:
                    df_features = df[feature_cols].values
                    df_targets = df['target_return'].values
                del df, ddf; gc.collect()
                
                y_data = df_targets[lookback : n_rows - horizon]
                itemsize = df_features.itemsize
                num_features = df_features.shape[1]
                shape = (n_sequences, lookback, num_features)
                strides = (itemsize * num_features, itemsize * num_features, itemsize)
                X_day = np_as_strided(df_features, shape=shape, strides=strides)
                
                valid_targets = ~np.isnan(y_data)
                valid_seqs = ~np.isnan(X_day).any(axis=(1, 2))
                valid_mask = valid_targets & valid_seqs

                X_day_valid_cpu = X_day[valid_mask].astype(np.float32)
                y_day_valid_cpu = y_data[valid_mask].astype(np.float32)
                n_valid = len(X_day_valid_cpu)
            
            if n_valid == 0:
                 return None, None
                 
            return X_day_valid_cpu, y_day_valid_cpu

        except Exception as e:
            print(f"\nError processing day {day_num} for training: {e}")
            if self.use_gpu and CUDF_AVAILABLE and cupy is not None: 
                cupy.get_default_memory_pool().free_all_blocks()
            return None, None
        
    def load_feature_statistics(self):
        stats_path = f"{self.config.CACHE_DIR}/feature_stats.json"
        if not os.path.exists(stats_path):
            print(f"ERROR: Feature statistics file not found at {stats_path}")
            return False
        try:
            with open(stats_path, 'r') as f:
                self.feature_stats = json.load(f)
            self.stats_computed = True
            print(f"Feature statistics loaded from {stats_path}")
            return True
        except Exception as e:
            print(f"Error loading feature statistics: {e}")
            return False
        
    def normalize_partition(self, partition):
        feature_cols = self.config.get_feature_columns()
        for col in feature_cols:
            if col in partition.columns and col in self.feature_stats:
                mean = self.feature_stats[col]['mean']
                std = self.feature_stats[col]['std']
                partition[col] = (partition[col] - mean) / std
                
                if self.use_gpu:
                    partition[col] = partition[col].replace([np.inf, -np.inf], 0.0)
                    partition[col] = partition[col].fillna(0.0)
                else:
                    partition[col] = partition[col].replace([np.inf, -np.inf], 0.0)
                    partition[col] = partition[col].fillna(0.0)
                    
                partition[col] = partition[col].clip(-5, 5)
        return partition
    
    def create_target_variable_partition(self, partition):
        horizon = self.config.PREDICTION_HORIZON
        for h in range(1, horizon + 1):
            partition[f'future_return_{h}'] = partition['Price'].pct_change(h).shift(-h)
            
        return_cols = [f'future_return_{h}' for h in range(1, horizon + 1)]
        # --- SCALING FIX: Multiply by 100 to get Percentage Points ---
        partition['target_return'] = partition[return_cols].mean(axis=1) * 100.0
        # -------------------------------------------------------------
        
        if self.use_gpu:
            partition['target_return'] = partition['target_return'].replace([np.inf, -np.inf], 0.0)
            partition['target_return'] = partition['target_return'].fillna(0.0)
        else:
            partition['target_return'] = partition['target_return'].replace([np.inf, -np.inf], 0.0)
            partition['target_return'] = partition['target_return'].fillna(0.0)
            
        partition = partition.drop(columns=return_cols)
        return partition

    def prepare_training_data(self, train_days: List[int], val_days: List[int], test_days: List[int]):
        print("PREPARING TRAINING DATA (COMPUTING STATS ONLY)")
        # Force re-computation or ensure cache is deleted manually
        if not self.stats_computed:
            self.compute_feature_statistics(train_days, sample_rate=0.2)
        return 1, 1, 1    

def main():
    config = Config()
    processor = StreamingDataProcessor(config)
    train_days = list(range(0, 10))
    val_days = list(range(10, 12))
    test_days = list(range(12, 15))
    processor.prepare_training_data(train_days, val_days, test_days)

if __name__ == "__main__":
    main()
    
