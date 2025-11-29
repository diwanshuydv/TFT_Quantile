import numpy as np
import gc
import os
from pathlib import Path
from typing import List, Tuple, Optional
import json  # <-- ADD
from tqdm import tqdm  # <-- ADD

try:
    import cudf
    import dask_cudf
    import cupy  # <-- ADD
    from cupy.lib.stride_tricks import as_strided as cp_as_strided  # <-- ADD
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False
    cupy = None  # <-- ADD
    cp_as_strided = None  # <-- ADD
    
from numpy.lib.stride_tricks import as_strided as np_as_strided  # <-- ADD
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
            
        # Wrap the main loop with tqdm for a progress bar
        for day_num in tqdm(day_numbers, desc="Computing Stats"):
            ddf = self.load_parquet_lazy(day_num)
            if ddf is None:
                continue
            ddf = self.filter_stable_period(ddf)
            if sample_rate < 1.0:
                ddf = ddf.sample(frac=sample_rate)
            
            # --- Optimization ---
            relevant_cols = [col for col in feature_cols if col in ddf.columns]
            if not relevant_cols:
                del ddf
                gc.collect()
                continue
                
            # Compute ALL relevant columns at once
            day_stats_df = ddf[relevant_cols].compute()
            
            for col in relevant_cols:
                col_data = day_stats_df[col]
            # --- End Optimization ---
                
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
                    
            del ddf, day_stats_df # Clear memory
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
        
        # --- ADD THIS ---
        # Save stats to a file
        stats_path = f"{self.config.CACHE_DIR}/feature_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(self.feature_stats, f, indent=4)
        print(f"Feature statistics saved to {stats_path}")
        # --- END ADD ---

    def process_day_for_training(self, day_num: int):
        # This function loads, processes, and returns X, y for a single day
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
        ddf = ddf.map_partitions(self.normalize_partition) # Relies on self.feature_stats
        ddf = ddf.map_partitions(self.create_target_variable_partition)
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
                # print("hre")
                df_features_gpu = df[feature_cols].to_cupy() # <-- CHANGED
                # print("hre2")
                df_targets_gpu = df['target_direction'].to_cupy() # <-- CHANGED
                del df, ddf; gc.collect()
                # print("hree")
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
                y_day_valid_gpu = y_data_gpu[valid_mask].astype(cupy.int8) # y should be int
                
                n_valid = len(X_day_valid_gpu)
                if n_valid > 0:
                    X_day_valid_cpu = X_day_valid_gpu.get()
                    y_day_valid_cpu = y_day_valid_gpu.get()

                del df_features_gpu, df_targets_gpu, X_day_gpu, y_data_gpu, valid_mask, X_day_valid_gpu, y_day_valid_gpu
                cupy.get_default_memory_pool().free_all_blocks()
            
            else:
                # Fallback to NumPy (CPU processing)
                # Check if df is a cudf dataframe (which it is if use_gpu=True but cupy=None)
                if CUDF_AVAILABLE and hasattr(df, 'to_numpy'):
                    df_features = df[feature_cols].to_numpy()
                    df_targets = df['target_direction'].to_numpy()
                else:
                    # It's a pandas dataframe
                    df_features = df[feature_cols].values
                    df_targets = df['target_direction'].values
                
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
                y_day_valid_cpu = y_data[valid_mask].astype(np.int8) # y should be int
                n_valid = len(X_day_valid_cpu)
                del df_features, df_targets, X_day, y_data, valid_mask
            
            # y target in file is -1, 0, 1. For CrossEntropy it must be 0, 1, 2.
            if y_day_valid_cpu is not None and n_valid > 0:
                y_day_valid_cpu = y_day_valid_cpu + 1

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
            print("Please run data preparation (step 2) first.")
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
        print("PREPARING TRAINING DATA (COMPUTING STATS ONLY)")
        if not self.stats_computed:
            self.compute_feature_statistics(train_days, sample_rate=0.2)
        
        print("DATA PREPARATION COMPLETE (STATS SAVED)")
        print("Skipping sequence generation. This will be done in memory during training.")
        # Return dummy counts
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
    
