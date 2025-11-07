import os
import gc
from pathlib import Path
import numpy as np
try:
    import cudf
    import dask_cudf
    CUDF_AVAILABLE = True
except ImportError:
    import pandas as pd
    import dask.dataframe as dd
    CUDF_AVAILABLE = False
    print("WARNING: cuDF not available, falling back to pandas")

from config import Config

class CSVToParquetConverter:
    def __init__(self, config: Config):
        self.config = config
        self.use_gpu = config.USE_GPU and CUDF_AVAILABLE
        print(f"Converter initialized with GPU: {self.use_gpu}")
    
    def convert_single_file(self, day_num: int, verbose: bool = True):
        csv_path = f"{self.config.DATA_DIR}/day{day_num}.csv"
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        
        if not os.path.exists(csv_path):
            if verbose:
                print(f"File not found: {csv_path}")
            return False
        
        if os.path.exists(parquet_path):
            if verbose:
                print(f"Already exists: {parquet_path} (skipping)")
            return True
        
        try:
            if verbose:
                print(f"Converting day{day_num}.csv to parquet...")
            
            if self.use_gpu:
                ddf = dask_cudf.read_csv(
                    csv_path,
                    blocksize=f'{self.config.CHUNK_SIZE}KB',
                    dtype={'Time': 'str', 'Price': 'float32'}
                )
            else:
                ddf = dd.read_csv(
                    csv_path,
                    blocksize=f'{self.config.CHUNK_SIZE}KB',
                    dtype={'Time': 'str', 'Price': 'float32'}
                )
            
            ddf['day'] = day_num
            
            ddf.to_parquet(
                parquet_path,
                engine='pyarrow',
                compression='snappy',
                write_index=False,
                row_group_size=self.config.PARQUET_ROW_GROUP_SIZE
            )
            
            if verbose:
                size_mb = os.path.getsize(parquet_path) / (1024 * 1024)
                print(f"day{day_num} converted ({size_mb:.1f} MB)")
            
            del ddf
            gc.collect()
            
            return True
            
        except Exception as e:
            print(f"Error converting day{day_num}: {str(e)}")
            return False
    
    def convert_all_files(self, start_day: int = 0, end_day: int = 279):
        print("CONVERTING CSV FILES TO PARQUET")
        print(f"Range: day{start_day} to day{end_day}")
        print(f"GPU Acceleration: {self.use_gpu}")
        
        success_count = 0
        fail_count = 0
        
        for day_num in range(start_day, end_day):
            success = self.convert_single_file(day_num, verbose=True)
            if success:
                success_count += 1
            else:
                fail_count += 1
        
        print("Conversion complete!")
        print(f"Successful: {success_count}")
        print(f"Failed: {fail_count}")
        
        return success_count, fail_count
    
    def verify_conversion(self, day_num: int):
        parquet_path = f"{self.config.PARQUET_DIR}/day{day_num}.parquet"
        
        try:
            if self.use_gpu:
                ddf = dask_cudf.read_parquet(parquet_path)
            else:
                ddf = dd.read_parquet(parquet_path)
            
            nrows = len(ddf)
            ncols = len(ddf.columns)
            
            print(f"day{day_num}: {nrows} rows, {ncols} columns")
            print(f"Columns: {list(ddf.columns[:10])}...")
            
            return True
            
        except Exception as e:
            print(f"Verification failed for day{day_num}: {str(e)}")
            return False


def main():
    config = Config()
    converter = CSVToParquetConverter(config)
    converter.convert_all_files(start_day=0, end_day=279)
    print("\nVerifying sample files...")
    for day in [0, 100, 200]:
        converter.verify_conversion(day)


if __name__ == "__main__":
    main()
