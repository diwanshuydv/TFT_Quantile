import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_converter import CSVToParquetConverter
from data_processor_streaming import StreamingDataProcessor
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset, count_parameters
from trainer_enhanced import EnhancedTFTTrainer
from trading_strategy_enhanced import EnhancedTFTTradingStrategy, PerformanceEvaluator
from hyperparameter_tuning import OptunaHyperparameterTuner
import argparse
from datetime import datetime
from pathlib import Path
import warnings
import random  # <-- ADD
from tqdm import tqdm  # <-- ADD
import gc  # <-- ADD
warnings.filterwarnings('ignore')

def setup_environment(config: Config):
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    if config.USE_GPU and torch.cuda.is_available():
        torch.cuda.manual_seed(config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = config.DETERMINISTIC
        torch.backends.cudnn.benchmark = not config.DETERMINISTIC
    print("ENVIRONMENT SETUP")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"CUDA Available: No (using CPU)")


def step1_convert_to_parquet(config: Config):
    print("\n")
    print("STEP 1: CONVERT TO PARQUET")
    converter = CSVToParquetConverter(config)
    success, failed = converter.convert_all_files(start_day=0, end_day=279)
    print(f"\nConversion complete: {success} successful, {failed} failed")
    return success > 0


def step2_prepare_data(config: Config, train_end: int = 195, val_end: int = 237):
    print("\n")
    print("STEP 2: PREPARE DATA")
    processor = StreamingDataProcessor(config)
    train_days = list(range(0, train_end))
    val_days = list(range(train_end, val_end))
    test_days = list(range(val_end, 279))
    print(f"Training days: {len(train_days)} (0-{train_end-1})")
    print(f"Validation days: {len(val_days)} ({train_end}-{val_end-1})")
    print(f"Test days: {len(test_days)} ({val_end}-278)")
    train_count, val_count, test_count = processor.prepare_training_data(train_days, val_days, test_days)
    print(f"\nData preparation complete")
    return train_count > 0


def step3_hyperparameter_tuning(config: Config, device: torch.device, n_trials: int = 50):
    print("\n")
    print("STEP 3: HYPERPARAMETER TUNING")
    tuner = OptunaHyperparameterTuner(config, device)
    study = tuner.run_optimization(n_trials=n_trials)
    print(f"\nHyperparameter tuning complete")
    return study


def step4_train_model(config: Config, device: torch.device, use_best_params: bool = True):
    print("STEP 4: TRAIN MODEL (STREAMING)")
    
    # 1. Create Model (same as before)
    if use_best_params:
        try:
            tuner = OptunaHyperparameterTuner(config, device)
            model, best_params = tuner.create_model_from_best_params()
            config.LEARNING_RATE = best_params['learning_rate']
            config.BATCH_SIZE = best_params['batch_size']
            print("\nUsing optimized hyperparameters")
        except Exception as e:
            print(f"\nCould not load best params: {e}")
            print("Using default configuration")
            model = TemporalFusionTransformer(
                num_features=config.get_num_features(),
                hidden_size=config.HIDDEN_SIZE,
                lstm_layers=config.LSTM_LAYERS,
                num_attention_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT,
                ffn_hidden_size=config.FFN_HIDDEN_SIZE
            )
    else:
        model = TemporalFusionTransformer(
            num_features=config.get_num_features(),
            hidden_size=config.HIDDEN_SIZE,
            lstm_layers=config.LSTM_LAYERS,
            num_attention_heads=config.ATTENTION_HEADS,
            dropout=config.DROPOUT,
            ffn_hidden_size=config.FFN_HIDDEN_SIZE
        )

    # 2. Setup Trainer and Data Processor
    trainer = EnhancedTFTTrainer(model, config, device)
    processor = StreamingDataProcessor(config)
    
    print("\nLoading feature statistics...")
    if not processor.load_feature_statistics():
        print("Could not load stats. Exiting.")
        return None
    
    # Define day ranges from command-line or defaults
    # This assumes --train-end and --val-end are available via args
    # For simplicity, I'll hardcode the defaults from main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-end', type=int, default=195)
    parser.add_argument('--val-end', type=int, default=237)
    args, _ = parser.parse_known_args()
    
    train_days = list(range(0, args.train_end))
    val_days = list(range(args.train_end, args.val_end))
    
    print(f"Streaming {len(train_days)} train days and {len(val_days)} val days.")
    
    # 3. New Online Training Loop
    print(f"Starting online training for {config.MAX_EPOCHS} epochs.")
    
    for epoch in range(config.MAX_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.MAX_EPOCHS} ---")
        
        # --- TRAINING PHASE ---
        model.train()
        random.shuffle(train_days) # Shuffle days each epoch
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        train_batches = 0
        
        with tqdm(total=len(train_days), desc=f"Epoch {epoch+1} Training") as pbar:
            for day_num in train_days:
                X_day, y_day = processor.process_day_for_training(day_num)
                
                if X_day is None or len(X_day) == 0:
                    pbar.update(1)
                    continue
                
                day_dataset = TradingDataset(X_day, y_day) # Use new __init__
                day_loader = DataLoader(
                    day_dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=True, 
                    num_workers=0,
                    pin_memory=config.USE_GPU
                )
                
                nan_batches = 0
                for batch_idx, (sequences, targets) in enumerate(day_loader):
                    sequences = sequences.to(device)
                    targets = targets.to(device)
                    trainer.optimizer.zero_grad()
                    
                    if trainer.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = trainer.model(sequences)
                            loss = trainer.criterion(outputs, targets)
                    else:
                        outputs = trainer.model(sequences)
                        loss = trainer.criterion(outputs, targets)
                        
                    if torch.isnan(loss):
                        nan_batches += 1
                        continue

                    if trainer.use_amp:
                        trainer.scaler.scale(loss).backward()
                        trainer.scaler.unscale_(trainer.optimizer)
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.GRADIENT_CLIP_VAL)
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.GRADIENT_CLIP_VAL)
                        trainer.optimizer.step()
                        
                    total_train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_train_samples += targets.size(0)
                    total_train_correct += (predicted == targets).sum().item()
                
                train_batches += (len(day_loader) - nan_batches)
                pbar.set_postfix_str(f"Day {day_num} loss: {total_train_loss/max(train_batches,1):.4f}")
                
                del X_day, y_day, day_dataset, day_loader
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                pbar.update(1)
        
        avg_train_loss = total_train_loss / max(train_batches, 1)
        avg_train_acc = 100 * total_train_correct / max(total_train_samples, 1)

        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        val_batches = 0
        
        with torch.no_grad():
            with tqdm(total=len(val_days), desc=f"Epoch {epoch+1} Validation") as pbar:
                for day_num in val_days:
                    X_day, y_day = processor.process_day_for_training(day_num)
                    
                    if X_day is None or len(X_day) == 0:
                        pbar.update(1)
                        continue
                        
                    day_dataset = TradingDataset(X_day, y_day) # Use new __init__
                    day_loader = DataLoader(
                        day_dataset,
                        batch_size=config.BATCH_SIZE,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=config.USE_GPU
                    )
                    
                    for sequences, targets in day_loader:
                        sequences = sequences.to(device)
                        targets = targets.to(device)
                        
                        if trainer.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = trainer.model(sequences)
                                loss = trainer.criterion(outputs, targets)
                        else:
                            outputs = trainer.model(sequences)
                            loss = trainer.criterion(outputs, targets)
                            
                        if torch.isnan(loss):
                            continue
                            
                        total_val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total_val_samples += targets.size(0)
                        total_val_correct += (predicted == targets).sum().item()
                        val_batches += 1
                    
                    del X_day, y_day, day_dataset, day_loader
                    gc.collect()
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    pbar.update(1)

        avg_val_loss = total_val_loss / max(val_batches, 1)
        avg_val_acc = 100 * total_val_correct / max(total_val_samples, 1)
        
        # --- EPOCH SUMMARY ---
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        trainer.training_history['train_loss'].append(avg_train_loss)
        trainer.training_history['train_acc'].append(avg_train_acc)
        trainer.training_history['val_loss'].append(avg_val_loss)
        trainer.training_history['val_acc'].append(avg_val_acc)
        trainer.training_history['learning_rate'].append(current_lr)
        
        if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
            print("\nERROR: Training producing NaN! Stopping...")
            break
            
        if avg_val_loss < trainer.best_val_loss:
            trainer.best_val_loss = avg_val_loss
            trainer.best_val_acc = avg_val_acc
            trainer.save_checkpoint(epoch, is_best=True)
            print("  ✓ Best model saved!")
            trainer.patience_counter = 0
        else:
            trainer.patience_counter += 1
            
        if (epoch + 1) % trainer.config.SAVE_CHECKPOINT_EVERY == 0:
            trainer.save_checkpoint(epoch, is_best=False)
            
        if trainer.patience_counter >= trainer.config.EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
            
        trainer.scheduler.step(avg_val_loss)
        if trainer.optimizer.param_groups[0]['lr'] <= trainer.config.LR_MIN:
            print("\nLearning rate reached minimum. Stopping training.")
            break
            
    # --- End of Epoch Loop ---
    if not trainer.model_saved and 'epoch' in locals():
        trainer.save_checkpoint(epoch, is_best=False)
    
    trainer.save_training_history()
    print("TRAINING COMPLETE")
    print(f"Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"Best Val Acc: {trainer.best_val_acc:.2f}%")
    
    return model


def step5_backtest_strategy(config: Config, model, device: torch.device):
    print("STEP 5: BACKTEST STRATEGY")
    strategy = EnhancedTFTTradingStrategy(model, config, device)
    evaluator = PerformanceEvaluator(config)
    processor = StreamingDataProcessor(config)
    print("Loading feature statistics for backtest...")
    if not processor.load_feature_statistics():
        print("Could not load stats. Aborting backtest.")
        return {}
    # processor = StreamingDataProcessor(config)
    test_days = list(range(237, 279))
    all_trades = []
    print(f"\nBacktesting on {len(test_days)} days...")
    for idx, day_num in enumerate(test_days):
        print(f"\rProcessing day {day_num} ({idx+1}/{len(test_days)})...", end='', flush=True)
        ddf = processor.load_parquet_lazy(day_num)
        if ddf is None:
            continue
        ddf = processor.filter_stable_period(ddf)
        ddf = ddf.map_partitions(processor.normalize_partition)
        df = ddf.compute()
        day_trades = strategy.run_day(df, day_num)
        all_trades.extend(day_trades)
        del df, ddf
    print(f"\n\nBacktesting complete: {len(all_trades)} trades executed")
    results = evaluator.calculate_metrics(all_trades, len(test_days))
    evaluator.print_results(results)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"{config.RESULTS_DIR}/backtest_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json_results = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in results.items()}
        json.dump(json_results, f, indent=4)
    print(f"\nResults saved to: {results_file}")
    import pandas as pd
    trades_df = pd.DataFrame(all_trades)
    trades_file = f"{config.RESULTS_DIR}/trades_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    print(f"Trades saved to: {trades_file}")
    stats = strategy.get_statistics()
    print(f"\nStrategy Statistics:")
    print(f"  Total predictions: {stats['total_predictions']:,}")
    print(f"  Confident predictions: {stats['confident_predictions']:,}")
    print(f"  Confidence rate: {stats['confidence_rate']*100:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser(description='TFT Trading Strategy')
    parser.add_argument('--skip-conversion', action='store_true', help='Skip CSV to Parquet conversion')
    parser.add_argument('--skip-data-prep', action='store_true', help='Skip data preparation')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--tuning-trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--train-end', type=int, default=195, help='Last day for training (exclusive)')
    parser.add_argument('--val-end', type=int, default=237, help='Last day for validation (exclusive)')
    args = parser.parse_args()
    config = Config()
    config.print_config()
    setup_environment(config)
    device = torch.device('cuda' if config.USE_GPU and torch.cuda.is_available() else 'cpu')
    try:
        if not args.skip_conversion:
            if not step1_convert_to_parquet(config):
                print("\nConversion failed!")
                return
        if not args.skip_data_prep:
            if not step2_prepare_data(config, args.train_end, args.val_end):
                print("\nData preparation failed!")
                return
        if not args.skip_tuning:
            step3_hyperparameter_tuning(config, device, args.tuning_trials)
        if not args.skip_training:
            model = step4_train_model(config, device, use_best_params=not args.skip_tuning)
        else:
            print("\nLoading existing model...")
            checkpoint = torch.load(f"{config.MODEL_DIR}/best_tft_model.pt", map_location=device)
            model = TemporalFusionTransformer(
                num_features=config.get_num_features(),
                hidden_size=config.HIDDEN_SIZE,
                lstm_layers=config.LSTM_LAYERS,
                num_attention_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT,
                ffn_hidden_size=config.FFN_HIDDEN_SIZE
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
        results = step5_backtest_strategy(config, model, device)
        print("EXECUTION COMPLETE")
        print(f"Annual Return: {results['annual_return']*100:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Calmar Ratio: {results['calmar_ratio']:.3f}")
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
