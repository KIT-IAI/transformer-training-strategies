import datetime
import math
import numpy as np
from workalendar.europe import Portugal


DAY_IN_HOURS = 24
CALENDAR = Portugal()


def generate_cyclical_time_value(time_value, period_length):
    """
    Converts the given time_value to a cyclical value. Improves features with a cyclical behavior like hour-of-day or
    day of week.

    :param time_value: the value to transform
    :param period_length: the length of the period (e.g. 23 for a hourly value)
    :return: the cyclical feature
    """
    return [
        math.sin(2 * math.pi * time_value / period_length),
        math.cos(2 * math.pi * time_value / period_length)
    ]


def convert_datetime_to_hour_of_the_week(dt: datetime.datetime) -> int:
    """
    Converts the given datetime to the hour of the week.

    :param dt: the datetime to convert
    :return: the hour of the week in the interval [0, 167]
    """
    return datetime.datetime.weekday(dt) * DAY_IN_HOURS + dt.hour


def one_hot_encode(time_value, period_length):
    encoding = np.zeros(period_length, dtype=int)
    encoding[time_value] = 1
    return encoding


def get_calendar_features(time_point: datetime.datetime):
    hour_of_day_sine, hour_of_day_cosine = generate_cyclical_time_value(time_point.hour, DAY_IN_HOURS)
    day_of_week_sine, day_of_week_cosine = generate_cyclical_time_value(time_point.weekday(), 7)
    month_sine, month_cosine = generate_cyclical_time_value(time_point.month, 12)
    is_workday = float(CALENDAR.is_working_day(time_point))
    is_holiday = float(CALENDAR.is_holiday(time_point))
    next_is_workday = float(CALENDAR.is_working_day(time_point + datetime.timedelta(days=1)))
    return [
        hour_of_day_sine,
        hour_of_day_cosine,
        day_of_week_sine,
        day_of_week_cosine,
        month_sine,
        month_cosine,
        is_workday,
        is_holiday,
        next_is_workday
    ]


if __name__ == "__main__":
    time_point = datetime.datetime(year=2011, month=1, day=1, hour=0)
    features = get_calendar_features(time_point)
    print(features)
