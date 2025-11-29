import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import math

class TemporalFusionTransformer(nn.Module):
    def __init__(self, 
                 num_features: int,
                 hidden_size: int = 128,
                 lstm_layers: int = 3,
                 num_attention_heads: int = 8,
                 dropout: float = 0.3,
                 ffn_hidden_size: int = 256,
                 num_classes: int = 3):
        super().__init__()
        
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        
        self.input_embedding = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.vsn = EnhancedVariableSelectionNetwork(
            num_features, 
            hidden_size, 
            dropout
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False
        )
        
        self.temporal_fusion = TemporalFusionDecoder(
            hidden_size,
            num_attention_heads,
            dropout
        )
        
        self.enrichment = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, dropout)
            for _ in range(2)
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(2)
        ])
        
        self.attention_norm = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(2)
        ])
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        self.output_gate = GatedLinearUnit(hidden_size, dropout)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        self._print_parameter_count()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param, gain=0.1)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def _print_parameter_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model parameters: {total:,} (trainable: {trainable:,})")
        
    def forward(self, x):
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        
        batch_size, seq_len, _ = x.shape
        x_selected = self.vsn(x)
        lstm_out, (hidden, cell) = self.lstm(x_selected)
        fused = self.temporal_fusion(lstm_out, x_selected)
        
        enriched = fused
        for grn in self.enrichment:
            enriched = grn(enriched)
        
        attn_out = enriched
        for attn, norm in zip(self.attention_layers, self.attention_norm):
            residual = attn_out
            attn_output, _ = attn(attn_out, attn_out, attn_out)
            attn_out = norm(residual + self.dropout(attn_output))
        
        residual = attn_out
        ffn_out = self.ffn(attn_out)
        attn_out = self.ffn_norm(residual + ffn_out)
        
        last_output = attn_out[:, -1, :]
        gated_output = self.output_gate(last_output)
        output = self.output_layer(gated_output)
        
        if torch.isnan(output).any():
            output = torch.nan_to_num(output, nan=0.0)
        
        return output


class EnhancedVariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_size, dropout):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.feature_transform = nn.Linear(num_features, hidden_size)
        self.context_grn = GatedResidualNetwork(num_features, hidden_size, dropout)
        self.feature_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_features)
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.num_features)
        context = self.context_grn(x_flat)
        attention_scores = self.feature_attention(context)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        weighted_features = x_flat * attention_weights
        transformed = self.feature_transform(weighted_features)
        transformed = self.layer_norm(transformed)
        transformed = self.dropout(transformed)
        output = transformed.reshape(batch_size, seq_len, self.hidden_size)
        return output


class TemporalFusionDecoder(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size, 
            num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.gate = GatedLinearUnit(hidden_size, dropout)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, lstm_output, vsn_output):
        attn_out, _ = self.attention(lstm_output, vsn_output, vsn_output)
        gated = self.gate(attn_out + lstm_output)
        return self.norm(gated)


class GatedLinearUnit(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.fc = nn.Linear(hidden_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc(x)
        value, gate = x.chunk(2, dim=-1)
        return self.dropout(value * torch.sigmoid(gate))


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.skip_connection = nn.Linear(input_size, hidden_size) if input_size != hidden_size else None
        
    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        gate = torch.sigmoid(self.gate(F.gelu(self.fc1(x))))
        residual = self.skip_connection(x) if self.skip_connection is not None else x
        output = gate * out + (1 - gate) * residual
        return self.layer_norm(output)


class TradingDataset(Dataset):
    def __init__(self, X_data, y_data):
        # X_data and y_data are already numpy arrays
        self.X = X_data
        self.y = y_data
        # We assume y_data is already 0, 1, 2 (from processor)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        y = torch.LongTensor([self.y[idx]])[0]
        return X, y

    @classmethod
    def from_files(cls, X_path: str, y_path: str, mmap_mode: bool = True):
        # This method is now only used for hyperparameter tuning
        print(f"Loading dataset from files: {X_path}, {y_path}")
        if mmap_mode:
            X = np.load(X_path, mmap_mode='r')
            y = np.load(y_path, mmap_mode='r')
        else:
            X = np.load(X_path)
            y = np.load(y_path)
        y = y + 1 # Add +1 when loading from file
        print(f"Dataset loaded: {len(X)} samples")
        print(f"Memory mapping: {mmap_mode}")
        return cls(X, y)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def test_model():
    from config import Config
    config = Config()
    num_features = config.get_num_features()
    model = TemporalFusionTransformer(
        num_features=num_features,
        hidden_size=config.HIDDEN_SIZE,
        lstm_layers=config.LSTM_LAYERS,
        num_attention_heads=config.ATTENTION_HEADS,
        dropout=config.DROPOUT,
        ffn_hidden_size=config.FFN_HIDDEN_SIZE
    )
    batch = torch.randn(16, config.LOOKBACK_WINDOW, num_features)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")


if __name__ == "__main__":
    test_model()
