import logging
from hyperopt import tpe

# ############################################################
# ADJUSTABLE VARIABLES

# folder name containing sensor data (relative from directory of sources)
rel_dir_name_sensor_data = '19-Aruba/'

# path of sources and outputs
path_data_sources = 'Data-Sources/' + rel_dir_name_sensor_data
# folder containing files read and written during runtime
dir_runtime_files = 'runtime-files/'
# folder of one iteration containing files read and written during runtime
dir_runtime_files_iteration = '%Y-%m-%d_%H-%M-%S/'
# filename of room separation
filename_room_separation = 'Room-separation.csv'
# filename of Adjacency-Matrix
filename_adjacency_matrix = 'Adjacency-Matrix.csv'
# filename of parameters file
filename_parameters_file = '0-Parameters.csv'
# filename of log file coming from logging module
filename_log_file = '1-LogFile.log'
# filename of Adjacency-plot
filename_adjacency_plot = 'adjacency_plot.pdf'

# csv configuration for sensor data file
# filename of the file that contains the sensor data
filename_sensor_data = '19-Aruba_data'
# delimiter of the columns in csv file of sensor data (input)
csv_delimiter_sensor_data = '\t'
# indicator at which line the data starts
csv_header_sensor_data = 0
# columns that should get parsed as a date
csv_parse_dates_sensor_data = ['DateTime']
# data type of columns in the file
csv_dtype_sensor_data = {'Active': float}

# filename of cases benchmark file
filename_benchmark = 'benchmark.csv'
# csv delimiter of benchmark file
csv_delimiter_benchmark = ';'
# indicator at which line the data starts
csv_header_benchmark = 0


# output files
# hyperopt parameter tuning
# optimization algorithm (representative Tree of Parzen Estimators (TPE))
opt_algorithm = tpe.suggest
# number of optimization attempts
opt_attempts = 2

# filename of trace file
filename_traces_raw = 'traces_raw.csv'
# csv delimiter of trace file
csv_delimiter_traces = ';'

# filename of divided trace file
filename_traces_basic = 'traces_basic.csv'
# csv delimiter of divided trace file
csv_delimiter_traces_basic = ';'

# filename of cluster file
filename_cluster = 'Cluster.csv'
# csv delimiter of cluster file
csv_delimiter_cluster = ';'

# filename of cases cluster file
filename_cases_cluster = 'Cases_Cluster.csv'
# csv delimiter of cases_cluster file
csv_delimiter_cases_cluster = ';'

# choose between: quantity, time, quantity_time
data_types = 'time'
data_types_list = ['quantity', 'time', 'quantity_time']

# number of motion sensors
number_of_motion_sensors = 31
# prefix of motion sensor IDs
prefix_motion_sensor_id = 'M'

# set a level of logging
logging_level = logging.DEBUG

# set distance for zero to other sensors (range for parameter optimization)
# used in creation of the distance_matrix_real_world matrix
zero_distance_value_min = 1
zero_distance_value_max = 1

# upper limit for input_data (set to None if there is no limit)
max_number_of_raw_input = 6500

# threshold when sensors are considered too far away (range for parameter optimization)
distance_threshold_min = 1.2
distance_threshold_max = 1.2
distance_threshold_step_length = 0.1

# maximum number of persons which were in the house while the recording of sensor data
max_number_of_people_in_house = 1

# the time in seconds in which a sensor activation is assigned to a existing trace (range for parameter optimization)
traces_time_out_threshold_min = 300
traces_time_out_threshold_max = 300
# maximum length of traces (in case length mode is used to separate raw-traces)
trace_length_limit_min = 4
trace_length_limit_max = 10

# range of k-means cluster (parameter optimization)
k_means_number_of_clusters_min = 6
k_means_number_of_clusters_max = 16

# folder containing dfg png files
dir_dfg_cluster_files = 'directly_follows_graphs/'
# filename of dfg file (per cluster)
filename_dfg_cluster = 'DFG_Cluster_{cluster}.png'
# threshold for filtering out sensors in dfg relative to max occurrences of a sensor (value in range 0-1)
rel_proportion_dfg_threshold = 0.5

# # FreFraLa Method
# parameter how high the threshold for errors per day are until they get dropped
max_errors_per_day_min = 100
max_errors_per_day_max = 100

# miner used for process model creation - choose between: heuristic, inductive
miner_type = 'heuristic'
miner_type_list = ['heuristic', 'inductive']
