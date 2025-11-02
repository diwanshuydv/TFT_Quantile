import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import gc
from typing import List, Tuple
from config import Config
import os

class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.feature_columns = None
        
    def load_day_file(self, day_num: int) -> pd.DataFrame:
        file_path = f"{self.config.DATA_DIR}/day{day_num}.csv"
        try:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.config.CHUNK_SIZE):
                chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True)
            del chunks
            gc.collect()
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
            df['day'] = day_num
            return df
        except Exception as e:
            print(f"Error loading day{day_num}.csv: {e}")
            return None
    
    def filter_stable_period(self, df: pd.DataFrame) -> pd.DataFrame:
        stable_time = pd.to_datetime(self.config.STABLE_START_TIME, format='%H:%M:%S').time()
        df_filtered = df[df['Time'].dt.time >= stable_time].copy()
        return df_filtered
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col not in ['Time', 'Price', 'day']]
        valid_counts = df[self.feature_columns].notna().sum(axis=1)
        df = df[valid_counts >= self.config.MIN_VALID_FEATURES].copy()
        df[self.feature_columns] = df[self.feature_columns].ffill()
        df[self.feature_columns] = df[self.feature_columns].bfill()
        df[self.feature_columns] = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        df[self.feature_columns] = df[self.feature_columns].fillna(0)
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        for horizon in range(1, self.config.PREDICTION_HORIZON + 1):
            df[f'future_return_{horizon}'] = df['Price'].pct_change(horizon).shift(-horizon)
        return_cols = [f'future_return_{i}' for i in range(1, self.config.PREDICTION_HORIZON + 1)]
        df['target_return'] = df[return_cols].mean(axis=1)
        df['target_return'] = df['target_return'].replace([np.inf, -np.inf], np.nan)
        df['target_return'] = df['target_return'].fillna(0)
        df['target_direction'] = 0
        df.loc[df['target_return'] > 0.0001, 'target_direction'] = 1
        df.loc[df['target_return'] < -0.0001, 'target_direction'] = -1
        return df
    
    def create_sequences(self, df: pd.DataFrame, save_to_disk: bool = False, file_prefix: str = "sequences") -> Tuple[np.ndarray, np.ndarray]:
        sequences = []
        targets = []
        lookback = self.config.LOOKBACK_WINDOW
        feature_cols = [col for col in df.columns if col not in ['Time', 'day', 'target_return', 'target_direction'] and not col.startswith('future_return')]
        df_features = df[feature_cols].values
        df_targets = df['target_direction'].values
        for i in range(lookback, len(df) - self.config.PREDICTION_HORIZON):
            seq = df_features[i - lookback:i]
            target = df_targets[i]
            if not np.isnan(target) and not np.isnan(seq).any():
                sequences.append(seq)
                targets.append(target)
        X = np.array(sequences, dtype=np.float32)
        y = np.array(targets, dtype=np.int8)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        print(f"Created {len(X)} sequences (filtered from {len(df)} rows)")
        print(f"NaN count in sequences: {np.isnan(X).sum()}")
        print(f"Inf count in sequences: {np.isinf(X).sum()}")
        if save_to_disk:
            np.save(f"{self.config.MODEL_DIR}/{file_prefix}_X.npy", X)
            np.save(f"{self.config.MODEL_DIR}/{file_prefix}_y.npy", y)
            return None, None
        return X, y
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns if col not in ['Time', 'Price', 'day']]
        for col in self.feature_columns:
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                expanding_mean = df[col].expanding(min_periods=30).mean()
                expanding_std = df[col].expanding(min_periods=30).std()
                expanding_std = expanding_std + 1e-8
                df[col] = (df[col] - expanding_mean) / expanding_std
                df[col] = df[col].replace([np.inf, -np.inf], [5, -5])
                df[col] = df[col].clip(-5, 5)
                df[col] = df[col].fillna(0)
        return df
    
    def process_day_batch(self, day_numbers: List[int]) -> pd.DataFrame:
        dfs = []
        for day_num in day_numbers:
            df = self.load_day_file(day_num)
            if df is not None:
                df = self.filter_stable_period(df)
                df = self.handle_missing_values(df)
                df = self.normalize_features(df)
                df = self.create_target_variable(df)
                dfs.append(df)
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            del dfs
            gc.collect()
            return combined_df
        return None
    
    def prepare_training_data(self, start_day: int = 0, end_day: int = 50):
        batch_files = []
        for batch_start in range(start_day, end_day, self.config.MAX_DAYS_IN_MEMORY):
            batch_end = min(batch_start + self.config.MAX_DAYS_IN_MEMORY, end_day)
            day_numbers = list(range(batch_start, batch_end))
            print(f"Processing days {batch_start} to {batch_end-1}...")
            df_batch = self.process_day_batch(day_numbers)
            if df_batch is not None:
                file_prefix = f"batch_{batch_start}_{batch_end-1}"
                self.create_sequences(df_batch, save_to_disk=True, file_prefix=file_prefix)
                batch_files.append(file_prefix)
                del df_batch
                gc.collect()
        print("\nCombining all batches...")
        all_X = []
        all_y = []
        for file_prefix in batch_files:
            X_batch = np.load(f"{self.config.MODEL_DIR}/{file_prefix}_X.npy", mmap_mode='r')
            y_batch = np.load(f"{self.config.MODEL_DIR}/{file_prefix}_y.npy", mmap_mode='r')
            all_X.append(X_batch[:])
            all_y.append(y_batch[:])
            del X_batch, y_batch
            gc.collect()
            try:
                os.remove(f"{self.config.MODEL_DIR}/{file_prefix}_X.npy")
                os.remove(f"{self.config.MODEL_DIR}/{file_prefix}_y.npy")
            except PermissionError:
                print(f"Warning: Could not delete {file_prefix} files")
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        del all_X, all_y
        gc.collect()
        return X, y
