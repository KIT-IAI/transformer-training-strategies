import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt


if __name__ == "__main__":
    directory = "data/"
    start = datetime.datetime(2013, 12, 1)
    end = datetime.datetime(2013, 12, 31)
    files = [file for file in os.listdir(directory) if file.endswith(".csv")]
    for file in files:
        data = pd.read_csv(directory + file, comment="#", parse_dates=[0], index_col=0)
        print(data)
        data = data["PT"]
        data = data[start:end]
        plt.plot(data)
        plt.title(file)
        plt.show()
