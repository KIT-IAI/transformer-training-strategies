from typing import List

import numpy as np
import pandas
import pandas as pd
import datetime

from dataset.features import get_calendar_features


ORIGINAL_DATASET_FILE = "data/LD2011_2014.txt"
PREPROCESSED_DATASET_FILE = "data/LD2011_2014_preprocessed.txt"
AUTOFORMER_DATASET_FILE = "data-external/autoformer/all_six_datasets/all_six_datasets/electricity/electricity.csv"

DROP_BUILDINGS = [1, 3, 14, 15, 17, 23, 57, 64, 66, 80, 91, 93, 97, 123, 130, 131, 132, 133, 134, 136, 141, 156, 187,
                  192, 205, 223, 236, 246, 275, 278, 279, 282, 288, 302, 313, 315, 321, 329, 331, 332, 338, 346, 347,
                  348, 359, 367]

MINIMUM_SEQUENCE_LENGTH = 2 * 365 * 24

STANDARD_BEGIN_DATETIME = datetime.datetime(2012, 1, 1)


def get_column_name(building_number: int) -> str:
    return f"MT_{building_number:03d}"


def _drop_buildings(dataset: pd.DataFrame, building_numbers: List[int]) -> pd.DataFrame:
    drop_column_names = [get_column_name(building_number) for building_number in building_numbers]
    dataset = dataset.drop(drop_column_names, axis=1)
    return dataset


def _drop_short_sequences(dataset: pd.DataFrame) -> pd.DataFrame:
    drop_columns = []
    for column in dataset:
        time_series = np.trim_zeros(dataset[column], trim="f")
        if len(time_series) < MINIMUM_SEQUENCE_LENGTH:
            drop_columns.append(column)
    dataset = dataset.drop(drop_columns, axis=1)
    return dataset


def _read_dataset_file(path) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter=";", decimal=",", parse_dates=[0], index_col=0)
    data.index = data.index - datetime.timedelta(minutes=15)
    data = data.groupby([data.index.date, data.index.hour]).mean()
    data.index = [datetime.datetime(year=date.year, month=date.month, day=date.day, hour=hour)
                  for date, hour in data.index]
    return data


def load_dataset(uci_repo: bool, preprocessed: bool) -> pd.DataFrame:
    if uci_repo:
        data_path = PREPROCESSED_DATASET_FILE if preprocessed else ORIGINAL_DATASET_FILE
        data = _read_dataset_file(data_path)
        if preprocessed:
            data = _drop_buildings(data, DROP_BUILDINGS)
            data = _drop_short_sequences(data)
    else:
        data = pandas.read_csv(AUTOFORMER_DATASET_FILE, parse_dates=[0], index_col=0)
        data.index = pandas.date_range(start=datetime.datetime(2014, 1, 1), periods=len(data), freq="H")
    return data


def get_first_nonzero_index(time_series: pd.Series):
    for i in range(len(time_series)):
        if time_series.iloc[i] > 0:
            return i
    return None


def trim_leading_zeros(time_series: pd.Series):
    index = get_first_nonzero_index(time_series)
    return time_series[index:]


def time_delta_hours(time_point: datetime.datetime, minus_time_point: datetime.datetime):
    time_delta = time_point - minus_time_point
    return time_delta.days * 24 + time_delta.seconds // 3600


DATASET_START = datetime.datetime(2011, 1, 1)
DATASET_END = datetime.datetime(2014, 12, 31, 23)


class UCIDataset:
    def __init__(self, input_time_steps: int = 168, uci_repo: bool = False, preprocessed: bool = False):
        self.df = load_dataset(uci_repo, preprocessed)
        self.df: pd.DataFrame
        self.input_time_steps = input_time_steps
        self._prepare_calendar_features()

    def get_data(self, building: str):
        return self.df[building]

    def buildings(self):
        return self.df.columns

    def _prepare_calendar_features(self):
        self.calendar_features = np.array(
            [get_calendar_features(time_point) for time_point in pd.date_range(start=DATASET_START,
                                                                               end=DATASET_END,
                                                                               freq="H")]
        )

    def get_calendar_features(self, time_point: datetime.datetime):
        hours_from_start = time_delta_hours(time_point, DATASET_START)
        print(hours_from_start, self.calendar_features[hours_from_start])

    def number_of_buildings(self):
        return len(self.buildings())


if __name__ == "__main__":
    uci_dataset = UCIDataset()
    print(uci_dataset.df)
    print(uci_dataset.calendar_features.shape)
    uci_dataset.get_calendar_features(datetime.datetime(2011, 1, 1))
    uci_dataset.get_calendar_features(datetime.datetime(2012, 1, 1))
    #for building in uci_dataset.buildings():
    #    X, Y, prediction_datetimes = uci_dataset.get_samples(building)
    #    print(building, X.shape, Y.shape, len(prediction_datetimes), prediction_datetimes[0], prediction_datetimes[-1])
    #for time_pt in pd.date_range(start=DATASET_START, end=DATASET_END, freq="H"):
    #    print(time_pt)
