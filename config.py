import os

class Config:
    DATA_DIR = "/data/quant14/EBY"
    PARQUET_DIR = "/data/quant14/EBY"
    MODEL_DIR = "./models"
    RESULTS_DIR = "./results"
    OPTUNA_DIR = "./optuna_studies"
    CACHE_DIR = "./cache"
    for directory in [MODEL_DIR, RESULTS_DIR, OPTUNA_DIR, CACHE_DIR, PARQUET_DIR]:
        os.makedirs(directory, exist_ok=True)
    USE_GPU = True
    GPU_DEVICE_ID = 0
    ENABLE_CUDF = True
    DASK_MEMORY_LIMIT = '20GB'
    DASK_N_WORKERS = 4
    DASK_THREADS_PER_WORKER = 2
    CHUNK_SIZE = 100000
    PARQUET_ROW_GROUP_SIZE = 50000
    LOOKBACK_WINDOW = 120
    PREDICTION_HORIZON = 10
    MIN_VALID_FEATURES = 100
    STABLE_START_TIME = "00:31:55"
    PRICE_FEATURES = [f'PB{i}_T{j}' for i in [10, 11, 6, 3, 2, 1, 7, 14, 15] for j in range(1, 13)]
    VOLUME_FEATURES = [f'VB{i}_T{j}' for i in [3] for j in range(1, 13)]
    BOLLINGER_FEATURES = [f'BB{i}_T{j}' for i in [13, 14, 15, 4, 5, 6] for j in range(1, 13)] + ['BB23']
    ALL_FEATURES = PRICE_FEATURES + VOLUME_FEATURES + BOLLINGER_FEATURES
    HIDDEN_SIZE = 128
    LSTM_LAYERS = 3
    ATTENTION_HEADS = 8
    DROPOUT = 0.3
    HIDDEN_CONTINUOUS_SIZE = 64
    FFN_HIDDEN_SIZE = 256
    VSN_HIDDEN_SIZE = 128
    NUM_QUANTILES = 7
    BATCH_SIZE = 512
    MAX_EPOCHS = 5
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-5
    GRADIENT_CLIP_VAL = 1.0
    LR_SCHEDULER = 'ReduceLROnPlateau'
    LR_PATIENCE = 5
    LR_FACTOR = 0.5
    LR_MIN = 1e-6
    EARLY_STOPPING_PATIENCE = 10
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    OPTUNA_N_TRIALS = 50
    OPTUNA_TIMEOUT = None
    OPTUNA_N_JOBS = 1
    HP_HIDDEN_SIZE = [64, 128, 256]
    HP_LSTM_LAYERS = [2, 3, 4]
    HP_ATTENTION_HEADS = [4, 8, 16]
    HP_DROPOUT = [0.2, 0.3, 0.4]
    HP_LEARNING_RATE = [1e-4, 3e-4, 1e-3]
    HP_BATCH_SIZE = [256, 512, 1024]
    POSITION_SIZE = 1
    MIN_PREDICTION_CONFIDENCE = 0.0
    SIGNAL_SMOOTHING_WINDOW = 5
    PROFIT_THRESHOLD = 0.025
    STOP_LOSS = 0.015
    TRAILING_STOP_PCT = 0.01
    MAX_HOLDING_TIME = 300
    ENABLE_DYNAMIC_SIZING = False
    MAX_POSITION_SIZE = 3
    RISK_PER_TRADE = 0.02
    MIN_ANNUAL_RETURN = 0.20
    MAX_DRAWDOWN = 0.10
    TARGET_SHARPE = 2.0
    TARGET_CALMAR = 3.0
    TRADING_DAYS_PER_YEAR = 279
    SECONDS_PER_DAY = 22438
    ENABLE_DERIVED_FEATURES = False
    RSI_PERIODS = [14, 28]
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ROLLING_WINDOWS = [30, 60, 120, 300]
    LOG_INTERVAL = 100
    SAVE_CHECKPOINT_EVERY = 5
    VERBOSE = True
    USE_WANDB = False
    WANDB_PROJECT = "tft-trading-strategy"
    WANDB_ENTITY = None
    RANDOM_SEED = 42
    DETERMINISTIC = True
    QUANTILES = [0.1, 0.5, 0.9] 
    OUTPUT_SIZE = len(QUANTILES)
    
    @classmethod
    def get_feature_columns(cls):
        return cls.ALL_FEATURES

    @classmethod
    def get_num_features(cls):
        return len(cls.ALL_FEATURES)

    @classmethod
    def print_config(cls):
        print("="*60)
        print("CONFIGURATION SUMMARY")
        print(f"GPU Enabled: {cls.USE_GPU}")
        print(f"Model Hidden Size: {cls.HIDDEN_SIZE}")
        print(f"LSTM Layers: {cls.LSTM_LAYERS}")
        print(f"Attention Heads: {cls.ATTENTION_HEADS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Lookback Window: {cls.LOOKBACK_WINDOW}")
        print(f"Total Features: {cls.get_num_features()}")
