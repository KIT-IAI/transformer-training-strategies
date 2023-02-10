import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from dataset import DROP_BUILDINGS


if __name__ == "__main__":
    DATASET_FILE = "data/LD2011_2014.txt"

    data = pd.read_csv(DATASET_FILE, delimiter=";", decimal=",", parse_dates=[0], index_col=0)
    data.index = data.index - datetime.timedelta(minutes=15)
    data = data.groupby([data.index.date, data.index.hour]).mean()
    data.index = [datetime.datetime(year=date.year, month=date.month, day=date.day, hour=hour)
                  for date, hour in data.index]

    print(data)

    drop_buildings = [f"MT_{building:03d}" for building in DROP_BUILDINGS]
    if "--drop" in sys.argv:
        buildings = drop_buildings
    else:
        drop_buildings = set(drop_buildings)
        buildings = [building for building in data.columns if building not in drop_buildings]
    print(buildings)

    for building in buildings:
        plt.plot(data[building])
        plt.title(building)
        plt.ylabel("load [kW]")
        plt.show()
