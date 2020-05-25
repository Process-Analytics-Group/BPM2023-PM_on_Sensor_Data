from datetime import datetime
import logging
import numpy as np

# ############################################################
# ADJUSTABLE VARIABLES

# path of sources
path_data_sources = 'Data-Sources/'
# filename of room separation
filename_room_separation = 'Room-separation.csv'
# filename of Adjacency-Matrix
filename_adjacency_matrix = 'Adjacency-Matrix.csv'
# filename of parameters file
filename_parameters_file = '0-Parameters.csv'
# filename of log file coming from logging module
filename_log_file = '1-LogFile'
# filename of Adjacency-plot
filename_adjacency_plot = 'adjacency_plot.pdf'

# csv configuration for sensor data file
# filename of the file that contains the sensor data
filename_sensor_data = 'sensor_raw.csv'
# folder name containing sensor data (relative from directory of sources)
rel_dir_name_sensor_data = 'Sensor-Data/'
# delimiter of the columns in csv file of sensor data (input)
csv_delimiter_sensor_data = ';'
# indicator at which line the data starts
csv_header_sensor_data = 0
# columns that should get parsed as a date
csv_parse_dates_sensor_data = ['DateTime']
# data type of columns in the file
csv_dtype_sensor_data = {'Active': np.int8}

# filename of trace file
filename_traces_raw = 'traces_raw.csv'
# csv delimiter of trace files
csv_delimiter_traces = ';'
# indicator at which line the data starts
csv_header_traces = 0

# folder name containing files read and written during runtime
# dir_runtime_files = 'runtime-files' + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# TODO

# choose between: quantity, time, quantity_time
data_types = 'time'
data_types_list = ['quantity', 'time', 'quantity_time']

# prefix of Motion-Sensor IDs
prefix_motion_sensor_id = 'M'

# set a level of logging
logging_level = logging.DEBUG

# set distance for zero to other sensors
# used in creation of the distance_matrix_real_world matrix
zero_distance_value_list = [1]

# upper limit for input_data (set to -1 if there is no limit)
max_number_of_raw_input = 6500

# threshold when sensors are considered too far away
distance_threshold_list = [1.2]

# maximum number of persons which were in the house while the recording of sensor data
max_number_of_people_in_house = 2

# the time in seconds in which a sensor activation is assigned to a existing trace
traces_time_out_threshold_list = [300]
# maximum length of traces (in case length mode is used to separate raw-traces)
max_trace_length_list = [4, 6, 8, 10]

# number of k-means cluster
k_means_number_of_clusters = [6, 8, 10, 12, 14, 16]
