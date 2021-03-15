import numpy as np
import pandas as pd
from copy import deepcopy


def choose_method(filtered_dataset,
                  dict_distance_adjacency_sensor,
                  method,
                  vectorization_method):
    if method == 'byRooms':
        case_vectors, raw_sensor_data_sensor_int = \
            get_vectors_by_rooms(dataset=filtered_dataset,
                                 dict_distance_adjacency_sensor=dict_distance_adjacency_sensor,
                                 vectorization_method=vectorization_method)
    else:
        return None
    return case_vectors, raw_sensor_data_sensor_int


def get_vectors_by_rooms(dataset,
                         dict_distance_adjacency_sensor,
                         vectorization_method):
    # create a rooms dictionary to faster check if sensors are in the same room
    # key is the sensor and values are all sensors in the same room
    room_dict = {}
    for key in dict_distance_adjacency_sensor['room_dict']:
        for item in dict_distance_adjacency_sensor['room_dict'][key]:

            # if room already exists, append sensors to already existing key. Reason is sensors in fringe areas that
            # are assigned to multiple rooms
            if item in room_dict:
                room_dict[item].update(dict_distance_adjacency_sensor['room_dict'][key])
            # if room does not exist yet, create new key
            else:
                room_dict[item] = set(dict_distance_adjacency_sensor['room_dict'][key])

    # create future pandas columns first as lists and add them later to the dataframe to safe runtime
    activity_list = []
    sensor_added = []
    lc_activity_list = []
    lc_list = []
    case = []

    # vector to be returned by the method
    time_vector = None
    quantity_vector = None

    currently_active_sensors = []
    current_case = 1
    latest_active_sensor = None
    # calculate the durations
    # subtract time of each row with the preceding row
    duration_list = ((dataset['DateTime'].diff(periods=1)).shift(-1)).dt.total_seconds()

    for data_row in dataset.itertuples():

        # if sensor from current row is not in the room as the previous sensor, start new case
        if latest_active_sensor and \
                data_row.SensorID not in room_dict[latest_active_sensor] and \
                data_row.Active == 1:
            current_case += 1
            currently_active_sensors = []
            latest_active_sensor = None

        if data_row.Active == 1:
            sensor_added.append(data_row.SensorID)
            currently_active_sensors.append(data_row.SensorID)
            lc_list.append('s')
            latest_active_sensor = data_row.SensorID
        elif data_row.Active == 0:
            # remove sensor from active sensors
            if data_row.SensorID in currently_active_sensors:
                currently_active_sensors.remove(data_row.SensorID)
            sensor_added.append(0)
            lc_list.append('c')
        lc_activity_list.append(data_row.SensorID)
        case.append(current_case)
        activity_list.append(currently_active_sensors[:])

    dataset['Activity'] = activity_list
    dataset['Sensor_Added'] = sensor_added
    dataset['Duration'] = duration_list
    dataset['LC_Activity'] = lc_activity_list
    dataset['LC'] = lc_list
    dataset['Case'] = case
    dataset.rename(columns={'DateTime': 'Timestamp'}, inplace=True)
    raw_sensor_data_sensor_int = dataset.drop(columns='Active')

    default_index = ['Activity', 'Sensor_Added', 'Duration', 'Timestamp', 'LC_Activity', 'LC', 'Case']
    raw_sensor_data_sensor_int = raw_sensor_data_sensor_int.reindex(columns=default_index)

    # count how many times a sensor has been activated in a case:
    if vectorization_method == 'quantity' or vectorization_method == 'quantity_time':
        quantity_vector = pd.pivot_table(raw_sensor_data_sensor_int,
                                         index=['Case'], columns=['Sensor_Added'],
                                         values=['LC'], aggfunc='count', fill_value=0)
        # ToDo: rename columns (not multiindexed)
    # count how long a sensor has been activated in a case:
    if vectorization_method == 'time' or vectorization_method == 'quantity_time':
        time_vector = pd.pivot_table(raw_sensor_data_sensor_int,
                                     index=['Case'],
                                     columns=['Sensor_Added'],
                                     values=['Duration'], aggfunc='sum', fill_value=0)

    if vectorization_method == 'time':
        return time_vector, raw_sensor_data_sensor_int
    elif vectorization_method == 'quantity':
        return quantity_vector, raw_sensor_data_sensor_int
    elif vectorization_method == 'quantity_time':
        quantity_time_vector = pd.concat([quantity_vector, time_vector], axis=1)
        return quantity_time_vector, raw_sensor_data_sensor_int
    else:
        # ToDo Kai: Add to logger that wrong method was selected
        return None
