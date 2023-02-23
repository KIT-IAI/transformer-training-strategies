import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .electricity_dataset import ElectricityDataset


DTYPE = torch.float


class TransformerDataset(Dataset):
    def __init__(self, data: np.array, features: np.array, input_length: int, horizon: int):
        self.data = data
        self.features = features
        self.input_length = input_length
        self.horizon = horizon
        self.n_time_series = self.data.shape[1]
        self.n_time_steps = len(self.data) - self.input_length - self.horizon + 1
        self.n_samples = self.n_time_steps * self.n_time_series

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        column = item // self.n_time_steps
        row = item % self.n_time_steps
        time_series_window = torch.tensor(self.data[row:row + self.input_length, column], dtype=DTYPE).unsqueeze(dim=1)
        encoder_features = torch.tensor(self.features[row:row + self.input_length], dtype=DTYPE)
        x_encoder = torch.concat((time_series_window, encoder_features), dim=1)
        decoder_start = row + self.input_length
        dummy_time_series = torch.zeros(size=(self.horizon, 1), dtype=DTYPE)
        decoder_features = torch.tensor(self.features[decoder_start:decoder_start + self.horizon], dtype=DTYPE)
        x_decoder = torch.concat((dummy_time_series, decoder_features), dim=1)
        labels = torch.tensor(self.data[decoder_start:decoder_start + self.horizon, column], dtype=DTYPE)
        return x_encoder, x_decoder, labels


if __name__ == "__main__":
    dataset = ElectricityDataset()
    training_data, training_features = dataset.get_training_data()
    transformer_dataset = TransformerDataset(training_data, training_features, input_length=168, horizon=24)
    print(len(transformer_dataset))
    x_enc, x_dec, y = transformer_dataset[0]
    print(x_enc)
    print(x_enc.shape)
    print(x_dec)
    print(x_dec.shape)
    print(y)
    print(y.shape)
    #for x_enc, x_dec, y in tqdm(transformer_dataset):
    #    pass
    data_loader = DataLoader(transformer_dataset, batch_size=256)
    for _ in tqdm(data_loader):
        pass