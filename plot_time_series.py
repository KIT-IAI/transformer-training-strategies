from dataset import UCIDataset
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    dataset = UCIDataset()
    print(dataset.df)

    columns_to_plot = dataset.df.columns
    #columns_to_plot = ["MT_002", "MT_019", "MT_021"]
    columns_to_plot = ["MT_005"]

    for column in columns_to_plot:
        time_series = dataset.df[column]
        time_series = time_series[np.cumsum(time_series) > 0]
        plt.plot(time_series)
        plt.title(column)
        plt.show()
