import os
import numpy as np
import pandas as pd
import datetime


SEPARATOR = ","
TIME_SERIES_INDICATOR_COL = 3
TARGET_TIME_SERIES_INDICATOR = "GC"
CUSTOMER_COL = 0
N_CUSTOMERS = 300
TIME_SERIES_START_COL = 5
TIME_SERIES_LENGTH = 48
DATE_COL = 4

MONTHS = {
    "Jan" : 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12
}

OUTPUT_PATH = os.path.join("data", "ausgrid.csv")

if __name__ == "__main__":
    directory = os.path.join("data-external", "ausgrid")
    files = ["2010-2011.csv",
             "2011-2012.csv",
             "2012-2013.csv"]

    start_date = datetime.date(2010, 7, 1)
    end_date = datetime.date(2013, 7, 1)
    time_range = pd.date_range(start_date, end_date, freq="H", inclusive="left")
    time_index = pd.DatetimeIndex(time_range)
    columns = list(range(1, N_CUSTOMERS + 1))
    df = pd.DataFrame(index=time_index, columns=columns)

    for file in files:
        file_version = 1 if file == "2010-2011.csv" else 2
        print(file)
        seen_customers = set()
        file_path = os.path.join(directory, file)
        with open(file_path) as f:
            lines = f.readlines()
        for line in lines:
            elements = line.split(SEPARATOR)
            if elements[TIME_SERIES_INDICATOR_COL] != TARGET_TIME_SERIES_INDICATOR:
                continue
            customer = int(elements[CUSTOMER_COL])
            if customer not in seen_customers:
                seen_customers.add(customer)
                print(customer)
            row = np.array([float(element) for element in elements[TIME_SERIES_START_COL:TIME_SERIES_START_COL + TIME_SERIES_LENGTH]])
            hourly_data = row.reshape((2, -1)).sum(0)
            if file_version == 1:
                date = elements[DATE_COL].split("-")
                day, month, year = int(date[0]), MONTHS[date[1]], 2000 + int(date[2])
            else:
                date = elements[DATE_COL].split("/")
                day, month, year = int(date[0]), int(date[1]), int(date[2])
            start = datetime.datetime(year, month, day)
            end = start + datetime.timedelta(hours=23)
            df[customer][start:end] = hourly_data

    print(df)
    df.to_csv(OUTPUT_PATH)