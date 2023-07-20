import numpy as np
import torch
from torch.utils.data import Dataset


DTYPE = torch.float


class MultivariateTransformerDataset(Dataset):
    def __init__(self, data: np.array, features: np.array, input_length: int, horizon: int):
        self.data = data
        self.features = features
        self.input_length = input_length
        self.horizon = horizon
        self.n_time_series = self.data.shape[1]
        self.n_time_steps = len(self.data) - self.input_length - self.horizon + 1

    def __len__(self):
        return self.n_time_steps

    def __getitem__(self, item):
        encoder_time_series = torch.tensor(self.data[item:item + self.input_length, :], dtype=DTYPE)
        encoder_features = torch.tensor(self.features[item:item + self.input_length], dtype=DTYPE)
        x_encoder = torch.concat((encoder_time_series, encoder_features), dim=1)
        decoder_start = item + self.input_length
        dummy_time_series = torch.zeros(size=(self.horizon, self.n_time_series), dtype=DTYPE)
        decoder_features = torch.tensor(self.features[decoder_start:decoder_start + self.horizon], dtype=DTYPE)
        x_decoder = torch.concat((dummy_time_series, decoder_features), dim=1)
        labels = torch.tensor(self.data[decoder_start:decoder_start + self.horizon, :], dtype=DTYPE)
        return x_encoder, x_decoder, labels
