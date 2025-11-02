import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
import gc
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from tft_model import TemporalFusionTransformer, TradingDataset, TFTTrainer
from trading_strategy import TFTTradingStrategy, PerformanceEvaluator

def main():
    config = Config()
    config.MAX_DAYS_IN_MEMORY = 2
    config.BATCH_SIZE = 64
    config.HIDDEN_SIZE = 24
    
    if config.USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    print("\n" + "="*60)
    print("STEP 1: DATA PREPARATION (MEMORY-OPTIMIZED)")
    print("="*60)
    
    data_processor = DataProcessor(config)
    TRAIN_DAYS_START = 150
    TRAIN_DAYS_END = 152
    TEST_DAYS_START = 152
    TEST_DAYS_END = 153
    
    print(f"\nTraining on days {TRAIN_DAYS_START}-{TRAIN_DAYS_END-1}")
    print(f"Testing on days {TEST_DAYS_START}-{TEST_DAYS_END-1}")
    print("\nNote: Using fewer days due to 8GB RAM constraint")
    print("For full dataset, use Google Colab with more RAM!\n")
    
    print("Loading and processing training data...")
    X_train, y_train = data_processor.prepare_training_data(
        start_day=TRAIN_DAYS_START, 
        end_day=TRAIN_DAYS_END
    )
    
    print(f"\nTraining sequences shape: {X_train.shape}")
    print(f"Training targets shape: {y_train.shape}")
    print(f"Target distribution: {np.bincount(y_train.astype(int) + 1)}")
    print(f"Memory usage: {X_train.nbytes / 1e9:.2f} GB")
    
    dataset = TradingDataset(X_train, y_train)
    del X_train, y_train
    gc.collect()
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.USE_GPU else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.USE_GPU else False
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    print("\nSTEP 2: MODEL TRAINING")
    
    sample_batch, _ = next(iter(train_loader))
    num_features = sample_batch.shape[2]
    
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=config.HIDDEN_SIZE,
        num_attention_heads=config.ATTENTION_HEADS,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"Input features: {num_features}")
    print(f"Hidden size: {config.HIDDEN_SIZE}")
    print(f"Attention heads: {config.ATTENTION_HEADS}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    trainer = TFTTrainer(model, config, device)
    print(f"\nStarting training for {config.MAX_EPOCHS} epochs...")
    trainer.train(train_loader, val_loader, config.MAX_EPOCHS)
    
    model.load_state_dict(torch.load(f"{config.MODEL_DIR}/best_tft_model.pt"))
    print("\nBest model loaded!")
    
    del train_loader, val_loader, train_dataset, val_dataset, dataset
    gc.collect()
    torch.cuda.empty_cache() if config.USE_GPU else None
    
    print("\nSTEP 3: STRATEGY BACKTESTING")
    
    print(f"\nTesting on days {TEST_DAYS_START}-{TEST_DAYS_END-1}...")
    
    strategy = TFTTradingStrategy(model, config, device)
    evaluator = PerformanceEvaluator(config)
    
    all_trades = []
    test_days = list(range(TEST_DAYS_START, TEST_DAYS_END))
    
    for day_num in test_days:
        print(f"\rProcessing day {day_num}...", end='')
        df = data_processor.load_day_file(day_num)
        if df is not None:
            df = data_processor.filter_stable_period(df)
            df = data_processor.handle_missing_values(df)
            df = data_processor.normalize_features(df, fit=False)
            day_trades = strategy.run_day(df)
            all_trades.extend(day_trades)
            del df
            gc.collect()
    
    print("\n\nBacktesting complete!")
    results = evaluator.evaluate(all_trades, days_traded=len(test_days))
    evaluator.print_results(results)
    
    results_file = f"{config.RESULTS_DIR}/strategy_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        results_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in results.items()}
        json.dump(results_json, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_file = f"{config.RESULTS_DIR}/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")
    
    print("EXECUTION COMPLETE!")

if __name__ == "__main__":
    main()
