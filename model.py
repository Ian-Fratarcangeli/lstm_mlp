import torch
import torch.nn as nn
import torch.nn.functional as F

class OptionPricingModel(nn.Module):
    def __init__(self, 
                 input_size_lstm=1, 
                 hidden_size_lstm=64, 
                 num_layers_lstm=3, 
                 input_size_mlp=5,  # number of static features
                 hidden_size_mlp=64,
                 dropout_rate=0.1):
        super(OptionPricingModel, self).__init__()

        #lstm for log returns, variable number of layers
        self.lstm = nn.LSTM(
            input_size=input_size_lstm,
            hidden_size=hidden_size_lstm,
            num_layers=num_layers_lstm,
            batch_first=True,
            dropout=dropout_rate
        )
        #3 layers, each with batch norm, leakyrelu activation, and dropout
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size_lstm + input_size_mlp, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, log_returns, option_features):
        """
        log_returns: shape (batch, seq_len, 1)
        option_features: shape (batch, input_size_mlp)
        """
        lstm_out, _ = self.lstm(log_returns)  # lstm_out: (batch, seq_len, hidden_size)
        lstm_last = lstm_out[:, -1, :]  # (batch, hidden_size)

        combined = torch.cat((lstm_last, option_features), dim=1)  # (batch, hidden_size + features)


        return self.mlp(combined)
    
