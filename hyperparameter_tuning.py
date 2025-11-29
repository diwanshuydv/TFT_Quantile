import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
import random
import gc
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config import Config
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset
from trainer_enhanced import EnhancedTFTTrainer
from data_processor_streaming import StreamingDataProcessor

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class OptunaHyperparameterTuner:
    """
    Optimized Optuna tuner with dask_cudf streaming for memory-efficient hyperparameter search.
    
    Key optimizations:
    - Streams data from parquet using dask_cudf (no huge numpy arrays)
    - Samples subset of days per trial for faster iteration
    - Efficient GPU memory management with immediate cleanup
    - Aggressive pruning for faster hyperparameter exploration
    """
    
    # Tuning-specific configuration
    TUNING_TRAIN_DAYS = 25      # Sample 10 train days per trial (very fast)
    TUNING_VAL_DAYS = 5         # Sample 3 val days per trial (very fast)
    TUNING_MAX_EPOCHS = 5       # Max epochs per trial (very fast)
    
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.study_path = f"{config.OPTUNA_DIR}/study.db"
        
        # Initialize streaming data processor
        self.processor = StreamingDataProcessor(config)
        
        # Load feature statistics (needed for normalization)
        print("Loading feature statistics for hyperparameter tuning...")
        if not self.processor.load_feature_statistics():
            raise RuntimeError("Feature statistics not found. Please run data preparation (step 2) first.")
        
        # Define day ranges for sampling
        self.all_train_days = list(range(0, 195))  # All available train days
        self.all_val_days = list(range(195, 237))  # All available val days
        
        print("OPTUNA HYPERPARAMETER TUNER INITIALIZED (DASK_CUDF STREAMING)")
        print(f"Device: {device}")
        print(f"Study path: {self.study_path}")
        print(f"Tuning config: {self.TUNING_TRAIN_DAYS} train days, {self.TUNING_VAL_DAYS} val days, {self.TUNING_MAX_EPOCHS} max epochs")
    
    def sample_days_for_trial(self, trial_number: int):
        """
        Sample subset of days for this trial.
        Uses trial number as seed for reproducibility while ensuring diversity across trials.
        """
        rng = random.Random(self.config.RANDOM_SEED + trial_number)
        
        train_days = rng.sample(self.all_train_days, self.TUNING_TRAIN_DAYS)
        val_days = rng.sample(self.all_val_days, self.TUNING_VAL_DAYS)
        
        train_days.sort()
        val_days.sort()
        
        return train_days, val_days
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for Optuna optimization.
        Streams data from parquet for memory efficiency.
        """
        # Sample hyperparameters
        hidden_size = trial.suggest_categorical('hidden_size', self.config.HP_HIDDEN_SIZE)
        lstm_layers = trial.suggest_categorical('lstm_layers', self.config.HP_LSTM_LAYERS)
        attention_heads = trial.suggest_categorical('attention_heads', self.config.HP_ATTENTION_HEADS)
        dropout = trial.suggest_categorical('dropout', self.config.HP_DROPOUT)
        learning_rate = trial.suggest_categorical('learning_rate', self.config.HP_LEARNING_RATE)
        batch_size = trial.suggest_categorical('batch_size', self.config.HP_BATCH_SIZE)
        ffn_hidden_size = hidden_size * 2
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}:")
        print(f"  hidden_size={hidden_size}, lstm_layers={lstm_layers}")
        print(f"  attention_heads={attention_heads}, dropout={dropout}")
        print(f"  learning_rate={learning_rate}, batch_size={batch_size}")
        print(f"{'='*60}")
        
        # Sample days for this trial
        train_days, val_days = self.sample_days_for_trial(trial.number)
        print(f"Sampled days - Train: {train_days[:5]}... Val: {val_days}")
        
        try:
            # Create model
            print("Creating model...")
            model = TemporalFusionTransformer(
                num_features=self.config.get_num_features(),
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                num_attention_heads=attention_heads,
                dropout=dropout,
                ffn_hidden_size=ffn_hidden_size,
                num_classes=3
            )
            
            # Create trial-specific config
            trial_config = self.create_trial_config(
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            
            # Initialize trainer
            print("Initializing trainer...")
            trainer = EnhancedTFTTrainer(model, trial_config, self.device)
            
            # Training loop with streaming data
            best_val_loss = float('inf')
            
            for epoch in range(self.TUNING_MAX_EPOCHS):
                print(f"\n--- Epoch {epoch+1}/{self.TUNING_MAX_EPOCHS} ---")
                
                # --- TRAINING PHASE ---
                model.train()
                train_day_sample = train_days.copy()
                random.shuffle(train_day_sample)
                
                total_train_loss = 0.0
                total_train_correct = 0
                total_train_samples = 0
                train_batches = 0
                
                print(f"Training on {len(train_day_sample)} days...")
                for day_idx, day_num in enumerate(train_day_sample):
                    print(f"  Day {day_num} ({day_idx+1}/{len(train_day_sample)})...", end=' ', flush=True)
                    
                    try:
                        X_day, y_day = self.processor.process_day_for_training(day_num)
                    except Exception as e:
                        print(f"ERROR: {e}")
                        continue
                    
                    if X_day is None or len(X_day) == 0:
                        print("No data")
                        continue
                    
                    print(f"{len(X_day)} seqs", end=' ', flush=True)
                    
                    day_dataset = TradingDataset(X_day, y_day)
                    day_loader = DataLoader(
                        day_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=self.config.USE_GPU
                    )
                    
                    day_batches = 0
                    for sequences, targets in day_loader:
                        sequences = sequences.to(self.device)
                        targets = targets.to(self.device)
                        
                        trainer.optimizer.zero_grad()
                        
                        if trainer.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = trainer.model(sequences)
                                loss = trainer.criterion(outputs, targets)
                        else:
                            outputs = trainer.model(sequences)
                            loss = trainer.criterion(outputs, targets)
                        
                        if torch.isnan(loss):
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
                        train_batches += 1
                        day_batches += 1
                    
                    print(f"-> {day_batches} batches")
                    
                    # Cleanup
                    del X_day, y_day, day_dataset, day_loader
                    if day_idx % 3 == 0:  # Periodic cleanup
                        gc.collect()
                        if self.device.type == 'cuda':
                            torch.cuda.empty_cache()
                
                avg_train_loss = total_train_loss / max(train_batches, 1)
                avg_train_acc = 100 * total_train_correct / max(total_train_samples, 1)
                
                # --- VALIDATION PHASE ---
                print(f"Validating on {len(val_days)} days...")
                model.eval()
                total_val_loss = 0.0
                total_val_correct = 0
                total_val_samples = 0
                val_batches = 0
                
                with torch.no_grad():
                    for day_idx, day_num in enumerate(val_days):
                        print(f"  Day {day_num} ({day_idx+1}/{len(val_days)})...", end=' ', flush=True)
                        
                        try:
                            X_day, y_day = self.processor.process_day_for_training(day_num)
                        except Exception as e:
                            print(f"ERROR: {e}")
                            continue
                        
                        if X_day is None or len(X_day) == 0:
                            print("No data")
                            continue
                        
                        print(f"{len(X_day)} seqs", end=' ', flush=True)
                        
                        day_dataset = TradingDataset(X_day, y_day)
                        day_loader = DataLoader(
                            day_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=self.config.USE_GPU
                        )
                        
                        day_batches = 0
                        for sequences, targets in day_loader:
                            sequences = sequences.to(self.device)
                            targets = targets.to(self.device)
                            
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
                            day_batches += 1
                        
                        print(f"-> {day_batches} batches")
                        
                        # Cleanup
                        del X_day, y_day, day_dataset, day_loader
                
                # Final cleanup for epoch
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                avg_val_loss = total_val_loss / max(val_batches, 1)
                avg_val_acc = 100 * total_val_correct / max(total_val_samples, 1)
                
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train: Loss={avg_train_loss:.4f}, Acc={avg_train_acc:.2f}%")
                print(f"  Val:   Loss={avg_val_loss:.4f}, Acc={avg_val_acc:.2f}%")
                
                # Report to Optuna for pruning
                trial.report(avg_val_loss, epoch)
                
                if trial.should_prune():
                    print(f"  >>> Trial PRUNED at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                
                # Track best validation loss
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    print(f"  >>> New best val loss!")
                
                # Early stopping if performance degrades significantly
                if epoch > 1 and avg_val_loss > best_val_loss * 1.5:
                    print(f"  >>> Early stopping (val loss diverging)")
                    break
            
            print(f"\nTrial {trial.number} complete - Best val loss: {best_val_loss:.4f}")
            
            # Cleanup
            del model, trainer
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cupy.get_default_memory_pool().free_all_blocks()
            
            return best_val_loss
            
        except optuna.TrialPruned:
            # Cleanup on pruning
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cupy.get_default_memory_pool().free_all_blocks()
            raise
            
        except Exception as e:
            print(f"\n!!! Trial FAILED with error: {str(e)}")
            import traceback
            traceback.print_exc()
            # Cleanup on error
            if 'model' in locals():
                del model
            if 'trainer' in locals():
                del trainer
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            if CUPY_AVAILABLE:
                cupy.get_default_memory_pool().free_all_blocks()
            return float('inf')
    
    def create_trial_config(self, learning_rate: float, batch_size: int):
        """Create trial-specific configuration."""
        class TrialConfig(Config):
            LEARNING_RATE = learning_rate
            BATCH_SIZE = batch_size
            MAX_EPOCHS = OptunaHyperparameterTuner.TUNING_MAX_EPOCHS
            LOG_INTERVAL = 5000  # Very infrequent logging during tuning
            VERBOSE = False
        return TrialConfig()
    
    def run_optimization(self, n_trials: int = None, timeout: int = None):
        """Run Optuna hyperparameter optimization."""
        if n_trials is None:
            n_trials = self.config.OPTUNA_N_TRIALS
        if timeout is None:
            timeout = self.config.OPTUNA_TIMEOUT
        
        print("\n" + "="*60)
        print("STARTING HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout}")
        print(f"Days per trial: {self.TUNING_TRAIN_DAYS} train, {self.TUNING_VAL_DAYS} val")
        print(f"Max epochs per trial: {self.TUNING_MAX_EPOCHS}")
        print("="*60 + "\n")
        
        study = optuna.create_study(
            study_name=f"tft_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction='minimize',
            sampler=TPESampler(seed=self.config.RANDOM_SEED),
            pruner=MedianPruner(
                n_startup_trials=2,      # Very aggressive: start pruning after 2 trials
                n_warmup_steps=1,        # Very aggressive: prune from epoch 1 onwards
                interval_steps=1         # Check every epoch
            ),
            storage=f'sqlite:///{self.study_path}',
            load_if_exists=True
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.OPTUNA_N_JOBS,
            show_progress_bar=True
        )
        
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (val_loss): {trial.value:.4f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("="*60 + "\n")
        
        self.save_results(study)
        return study
    
    def save_results(self, study: optuna.Study):
        """Save optimization results to JSON and CSV."""
        best_params = study.best_params
        best_value = study.best_value
        
        results = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': len(study.trials),
            'tuning_train_days': self.TUNING_TRAIN_DAYS,
            'tuning_val_days': self.TUNING_VAL_DAYS,
            'tuning_max_epochs': self.TUNING_MAX_EPOCHS,
            'timestamp': datetime.now().isoformat()
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = f"{self.config.OPTUNA_DIR}/best_params_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nBest parameters saved to: {json_path}")
        
        df = study.trials_dataframe()
        csv_path = f"{self.config.OPTUNA_DIR}/all_trials_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"All trials saved to: {csv_path}")
    
    def load_best_params(self, json_path: str = None) -> dict:
        """Load best parameters from JSON file."""
        if json_path is None:
            json_files = sorted(Path(self.config.OPTUNA_DIR).glob("best_params_*.json"))
            if not json_files:
                raise FileNotFoundError("No best_params files found")
            json_path = str(json_files[-1])
        
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        return results['best_params']
    
    def create_model_from_best_params(self, json_path: str = None):
        """Create model using best hyperparameters."""
        best_params = self.load_best_params(json_path)
        
        print("Creating model with best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        
        model = TemporalFusionTransformer(
            num_features=self.config.get_num_features(),
            hidden_size=best_params['hidden_size'],
            lstm_layers=best_params['lstm_layers'],
            num_attention_heads=best_params['attention_heads'],
            dropout=best_params['dropout'],
            ffn_hidden_size=best_params['hidden_size'] * 2,
            num_classes=3
        )
        
        return model, best_params


def main():
    """Main entry point for hyperparameter tuning."""
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tuner = OptunaHyperparameterTuner(config, device)
    study = tuner.run_optimization(n_trials=50)
    
    best_model, best_params = tuner.create_model_from_best_params()
    print("\nBest model created successfully!")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
