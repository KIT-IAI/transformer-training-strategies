import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler

from .features import get_calendar_features


ELECTRICITY_DATASET_FILE = "data-external/autoformer/all_six_datasets/all_six_datasets/electricity/electricity.csv"
TRAINING_AMOUNT = 0.7
VALIDATION_AMOUNT = 0.1
TEST_AMOUNT = 0.2


def load_dataframe():
    data = pd.read_csv(ELECTRICITY_DATASET_FILE, parse_dates=[0], index_col=0)
    data.index = pd.date_range(start=datetime.datetime(2012, 1, 1), periods=len(data), freq="H")
    return data


class ElectricityDataset:
    def __init__(self, scale: bool = True, use_calendar_features: bool = True):
        self.df = load_dataframe()
        self.scale = scale
        self.use_calendar_features = use_calendar_features
        self.scaler = None
        if scale:
            self._scale_time_series()
        self.calendar_features = self._compute_calendar_features() if use_calendar_features else None
        self.training_end = int(TRAINING_AMOUNT * len(self.df))
        self.validation_end = int((TRAINING_AMOUNT + VALIDATION_AMOUNT) * len(self.df))

    def _compute_calendar_features(self):
        return np.array([get_calendar_features(self.df.index[i]) for i in range(len(self.df))])

    def _scale_time_series(self):
        training_data = self.df.iloc[:int(TRAINING_AMOUNT * len(self.df))]
        self.scaler = StandardScaler()
        self.scaler.fit(training_data)
        self.df[self.df.columns] = self.scaler.transform(self.df)

    def _get_data(self, start_index: int, end_index: int):
        data = np.array(self.df[start_index:end_index])
        calendar_features = self.calendar_features[start_index:end_index] if self.use_calendar_features else None
        return data, calendar_features

    def get_training_data(self):
        return self._get_data(0, self.training_end)

    def get_validation_data(self):
        return self._get_data(self.training_end, self.validation_end)

    def get_test_data(self):
        return self._get_data(self.validation_end, len(self.df))


if __name__ == "__main__":
    dataset = ElectricityDataset(scale=True, use_calendar_features=True)
    for data, features in (dataset.get_training_data(), dataset.get_validation_data(), dataset.get_test_data()):
        print(data.shape)
        print(features.shape)
