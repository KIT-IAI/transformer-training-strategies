import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, n_layers: int, n_units: int, input_features: int, output_features: int, lookback_size: int, horizon: int):
        super().__init__()
        self.output_features = output_features
        self.horizon = horizon
        self.lstm_cell = nn.LSTM(input_size=input_features, hidden_size=n_units, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(in_features=lookback_size * n_units, out_features=output_features * horizon)

    def forward(self, x_enc, x_dec):
        lstm_output, hidden_states = self.lstm_cell(x_enc)
        flat_output = torch.flatten(lstm_output, start_dim=1)
        y_hat = self.linear(flat_output)
        if self.output_features > 1:
            y_hat = torch.reshape(y_hat, (y_hat.shape[0], self.horizon, self.output_features))
        return y_hat
