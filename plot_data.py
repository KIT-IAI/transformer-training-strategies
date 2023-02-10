import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd

from dataset import UCIDataset


if __name__ == "__main__":
    dataset = UCIDataset()
    start = datetime.datetime(2013, 12, 1)
    end = datetime.datetime(2013, 12, 31)
    n_buildings = 50
    dataset.df: pd.DataFrame
    for column in dataset.df.columns[:n_buildings]:
        time_series = dataset.df[column]
        mean_val = np.mean(time_series[time_series > 0])
        std = np.std(time_series[time_series > 0])
        plt.plot((time_series[start:end] - mean_val) / std, alpha=0.5)
    plt.show()
