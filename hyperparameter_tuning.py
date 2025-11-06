import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from datetime import datetime
from pathlib import Path

from config import Config
from tft_model_enhanced import TemporalFusionTransformer, TradingDataset
from trainer_enhanced import EnhancedTFTTrainer


class OptunaHyperparameterTuner:
    def __init__(self, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.train_X_path = f"{config.MODEL_DIR}/train_X.npy"
        self.train_y_path = f"{config.MODEL_DIR}/train_y.npy"
        self.val_X_path = f"{config.MODEL_DIR}/val_X.npy"
        self.val_y_path = f"{config.MODEL_DIR}/val_y.npy"
        self.study_path = f"{config.OPTUNA_DIR}/study.db"
        print("OPTUNA HYPERPARAMETER TUNER INITIALIZED")
        print(f"Device: {device}")
        print(f"Study path: {self.study_path}")
    
    def objective(self, trial: Trial) -> float:
        hidden_size = trial.suggest_categorical('hidden_size', self.config.HP_HIDDEN_SIZE)
        lstm_layers = trial.suggest_categorical('lstm_layers', self.config.HP_LSTM_LAYERS)
        attention_heads = trial.suggest_categorical('attention_heads', self.config.HP_ATTENTION_HEADS)
        dropout = trial.suggest_categorical('dropout', self.config.HP_DROPOUT)
        learning_rate = trial.suggest_categorical('learning_rate', self.config.HP_LEARNING_RATE)
        batch_size = trial.suggest_categorical('batch_size', self.config.HP_BATCH_SIZE)
        ffn_hidden_size = hidden_size * 2
        print(f"\nTrial {trial.number}:")
        print(f"  hidden_size={hidden_size}, lstm_layers={lstm_layers}")
        print(f"  attention_heads={attention_heads}, dropout={dropout}")
        print(f"  learning_rate={learning_rate}, batch_size={batch_size}")
        try:
            model = TemporalFusionTransformer(
                num_features=self.config.get_num_features(),
                hidden_size=hidden_size,
                lstm_layers=lstm_layers,
                num_attention_heads=attention_heads,
                dropout=dropout,
                ffn_hidden_size=ffn_hidden_size,
                num_classes=3
            )
            train_dataset = TradingDataset(self.train_X_path, self.train_y_path, mmap_mode=True)
            val_dataset = TradingDataset(self.val_X_path, self.val_y_path, mmap_mode=True)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.config.USE_GPU else False
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.config.USE_GPU else False
            )
            trial_config = self.create_trial_config(
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            trainer = EnhancedTFTTrainer(model, trial_config, self.device)
            max_epochs = 15
            best_val_loss = float('inf')
            for epoch in range(max_epochs):
                train_loss, train_acc = trainer.train_epoch(train_loader)
                val_loss, val_acc, _ = trainer.validate(val_loader)
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    print(f"  Trial pruned at epoch {epoch+1}")
                    raise optuna.TrialPruned()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if epoch > 5 and val_loss > best_val_loss * 1.2:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
            print(f"  Best val loss: {best_val_loss:.4f}")
            del model, trainer, train_loader, val_loader, train_dataset, val_dataset
            torch.cuda.empty_cache()
            return best_val_loss
        except Exception as e:
            print(f"  Trial failed with error: {str(e)}")
            return float('inf')
    
    def create_trial_config(self, learning_rate: float, batch_size: int):
        class TrialConfig(Config):
            LEARNING_RATE = learning_rate
            BATCH_SIZE = batch_size
            MAX_EPOCHS = 15
            LOG_INTERVAL = 200
            VERBOSE = False
        return TrialConfig()
    
    def run_optimization(self, n_trials: int = None, timeout: int = None):
        if n_trials is None:
            n_trials = self.config.OPTUNA_N_TRIALS
        if timeout is None:
            timeout = self.config.OPTUNA_TIMEOUT
        print("STARTING HYPERPARAMETER OPTIMIZATION")
        print(f"Number of trials: {n_trials}")
        print(f"Timeout: {timeout}")
        study = optuna.create_study(
            study_name=f"tft_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            direction='minimize',
            sampler=TPESampler(seed=self.config.RANDOM_SEED),
            pruner=MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
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
        print("OPTIMIZATION COMPLETE")
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  Value (val_loss): {trial.value:.4f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        self.save_results(study)
        return study
    
    def save_results(self, study: optuna.Study):
        best_params = study.best_params
        best_value = study.best_value
        results = {
            'best_params': best_params,
            'best_value': float(best_value),
            'n_trials': len(study.trials),
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
        if json_path is None:
            json_files = sorted(Path(self.config.OPTUNA_DIR).glob("best_params_*.json"))
            if not json_files:
                raise FileNotFoundError("No best_params files found")
            json_path = str(json_files[-1])
        with open(json_path, 'r') as f:
            results = json.load(f)
        return results['best_params']
    
    def create_model_from_best_params(self, json_path: str = None):
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
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tuner = OptunaHyperparameterTuner(config, device)
    study = tuner.run_optimization(n_trials=50)
    best_model, best_params = tuner.create_model_from_best_params()
    print("\nBest model created successfully!")
    print(f"Best parameters: {best_params}")


if __name__ == "__main__":
    main()
