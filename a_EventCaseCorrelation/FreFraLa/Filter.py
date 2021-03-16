import inspect
import logging

import pandas as pd
from datetime import datetime


def filter_visitor_days(dict_distance_adjacency_sensor,
                        threshold,
                        raw_sensor_data_sensor_int,
                        logging_level):
    """
    Shorts the full dataset (creates new one) depending on time window and error threshold
    :param threshold: error threshold for each day
    :param dict_distance_adjacency_sensor: contains information about the adjacency of the sensors
    :param raw_sensor_data_sensor_int: Data containing the sensor information raw, unamended
    :param logging_level: from which level logging statements should get executed
    """

    error_count = 0
    currently_active_sensors = []
    dates_to_remove = []
    date_last_row = None
    for data_row in raw_sensor_data_sensor_int.itertuples():

        date_current_row = data_row.DateTime.date()

        # If date has already too many errors, skip until the next one:
        if dates_to_remove and dates_to_remove[-1] == date_current_row:
            continue

        # check if new day has started
        if date_current_row != date_last_row and date_last_row is not None:
            # reset everything and begin counting errors for the new day with a clean slate
            error_count = 0
            currently_active_sensors = []
            date_last_row = date_current_row
            continue
        # if sensor is off, remove from current active sensors
        if data_row.Active == 0:
            if data_row.SensorID not in currently_active_sensors:
                continue
            else:
                currently_active_sensors.remove(data_row.SensorID)
            continue
        # if the current sensor and one of the recent sensors are not neighbours count it as an error
        adjacent = False
        for sensors in currently_active_sensors:
            if dict_distance_adjacency_sensor['adjacency_matrix'][data_row.SensorID][sensors] == 1:
                adjacent = True
                break
        if not adjacent and len(currently_active_sensors) > 0:
            error_count += 1

        # if error threshold is reached, remove the day from data_frame
        if error_count >= threshold:
            dates_to_remove.append(data_row.DateTime.date())
            # reset everything and begin counting errors for the new day with a clean slate skip all dates
            error_count = 0
            currently_active_sensors = []
            continue
        currently_active_sensors.append(data_row.SensorID)

    # count number of entries in dates_to_remove and save to log
    # remove all dates from error list
    raw_sensor_data_sensor_int = raw_sensor_data_sensor_int[~raw_sensor_data_sensor_int.DateTime.isin(dates_to_remove)]

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Filtered %s days of log file", dates_to_remove.count())

    return raw_sensor_data_sensor_int
