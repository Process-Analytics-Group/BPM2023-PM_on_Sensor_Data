from datetime import datetime
import logging

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
# folder name containing senor data
dir_name_sensor_data = 'Sensor-Data'
# folder name containing files read and written during runtime
#dir_runtime_files = 'runtime-files' + '/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
# choose between: quantity, time, quantity_time
data_types = 'time'
data_types_list = ['quantity', 'time', 'quantity_time']

# csv delimiter of sensor data (input)
csv_delimiter = '\s+'

# prefix of Motion-Sensor IDs
prefix_motion_sensor_id = 'M'

# set a level of logging
logging_level = logging.DEBUG

# set distance for zero to other sensors
# used in creation of the distance_matrix_real_world matrix
zero_distance_value_list = [1]

# upper limit for input_data
max_number_of_raw_input = 6500

# threshold when sensors are considered too far away
distance_threshold_list = [1.2]

max_number_of_people_in_house = 2

traces_time_out_threshold_list = [300]
# maximum length of traces (in case length mode is used to separate raw-traces)
max_trace_length_list = [4, 6, 8, 10]

# number of k-means cluster
k_means_number_of_clusters = [6, 8, 10, 12, 14, 16]