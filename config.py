import os

class Config:
    DATA_DIR = "./EBY"
    MODEL_DIR = "./models"
    RESULTS_DIR = "./results"
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    CHUNK_SIZE = 50000
    MAX_DAYS_IN_MEMORY = 2
    USE_DISK_CACHE = True
    
    LOOKBACK_WINDOW = 60
    PREDICTION_HORIZON = 5
    MIN_VALID_FEATURES = 50
    
    PRICE_FEATURES = [f'PB{i}_T{j}' for i in range(1, 19) for j in range(1, 13)]
    VOLUME_FEATURES = [f'VB{i}_T{j}' for i in range(1, 7) for j in range(1, 13)]
    BOLLINGER_FEATURES = [f'BB{i}_T{j}' for i in range(4, 16) for j in range(1, 13)] + ['BB23', 'BB24', 'BB25']
    
    HIDDEN_SIZE = 32
    ATTENTION_HEADS = 2
    DROPOUT = 0.2
    HIDDEN_CONTINUOUS_SIZE = 16
    
    BATCH_SIZE = 128
    MAX_EPOCHS = 10
    LEARNING_RATE = 0.001
    GRADIENT_CLIP_VAL = 0.5
    
    POSITION_SIZE = 1
    PROFIT_THRESHOLD = 0.02
    STOP_LOSS = 0.01
    MIN_PREDICTION_CONFIDENCE = 0.1
    
    MIN_ANNUAL_RETURN = 0.20
    MAX_DRAWDOWN = 0.10
    
    STABLE_START_TIME = "00:31:55"
    
    USE_GPU = True
