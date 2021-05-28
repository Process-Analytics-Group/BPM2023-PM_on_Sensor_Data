import logging
from hyperopt import tpe

# ############################################################
# ADJUSTABLE VARIABLES

# folder name containing sensor data (relative from directory of sources)
rel_dir_name_sensor_data = '19-Aruba/'

# path of sources and outputs
path_data_sources = 'z_Data-Sources/' + rel_dir_name_sensor_data
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
filename_sensor_data = '19-Aruba_Data'
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

# ToDo comment 'metric_to_be_maximised'
metric_to_be_maximised = 'Precision'
metric_to_be_maximised_list = ['Precision', 'Fitness', 'entropia:Precision', 'entropia:Fitness']

# range for number of clusters for the elbow-method (only used in "k-Means-Elbow" and "k-Medoids-Elbow")
# is not effected by parameter optimization
min_number_of_clusters = 3
max_number_of_clusters = 20

# Program execution type - Choose between the possible types:
# 'fixed_params' (the parameters are set before the program is executed),
# 'param_optimization' (uses a search space in which the parameters are optimized during execution)
execution_type = 'fixed_params'
# number of times the process model discovery gets executed
number_of_runs = 10

# upper limit for input_data (set to None if there is no limit)
max_number_of_raw_input = 10000

# fixed params execution parameters
fixed_params = {'zero_distance_value': 1,
                'traces_time_out_threshold': 300,
                'trace_length_limit': 6,
                'custom_distance_number_of_clusters': 10,
                'distance_threshold': 1.2,
                'max_errors_per_day': 100,
                'vectorization_type': 'time',
                'event_case_correlation_method': 'Classic',
                'clustering_method': 'sklearn-SOM'}

# hyperopt parameter tuning
# optimization algorithm (representative Tree of Parzen Estimators (TPE))
opt_algorithm = tpe.suggest

# range for vectorization type (parameter optimization)
# possible types: 'quantity', 'time', 'quantity_time'
vectorization_type_list = ['quantity', 'time', 'quantity_time']

# range for clustering method (parameter optimization)
# possible methods: 'Classic' (classical approach), 'FreFlaLa' (filter out days with visitors)
event_case_correlation_method_list = ['Classic', 'FreFlaLa']

# range for clustering method (parameter optimization)
# possible methods: 'SOM', 'sklearn-SOM', 'CustomDistance', 'k-Means', 'k-Medoids'
clustering_method_list = ['SOM']

# sklearn SOM settings hyperopt parameter tuning
# optimization algorithm (representative Tree of Parzen Estimators (TPE))
som_opt_algorithm = tpe.suggest
som_opt_attempts = 20
# The initial step size for updating the SOM weights. (default = 1)
min_lr = 1
max_lr = 1
# Parameter for magnitude of change to each weight. Does not update over training (as does learning rate)more aggressive
# updates to weights. (default = 1)
min_sigma = 1
max_sigma = 1

# number of motion sensors
number_of_motion_sensors = 31
# prefix of motion sensor IDs
prefix_motion_sensor_id = 'M'

# set a level of logging
logging_level = logging.INFO

# set distance for zero to other sensors (range for parameter optimization)
# used in creation of the distance_matrix_real_world matrix
zero_distance_value_min = 1
zero_distance_value_max = 1

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

# range for number of CustomDistance cluster (parameter optimization)
custom_distance_clusters_min = 6
custom_distance_clusters_max = 16

# Specifies the linkage method for clustering with custom distance calculation
linkage_method_for_clustering = 'ward'
linkage_method_for_clustering_list = ['single', 'complete', 'average', 'weighted', 'median', 'centroid', 'ward']

# threshold for filtering out sensors in dfg relative to max occurrences of a sensor (value in range 0-1)
rel_proportion_dfg_threshold = 0.5

# # FreFlaLa Method
# parameter how high the threshold for errors per day are until they get dropped
max_errors_per_day_min = 100
max_errors_per_day_max = 100

# miner used for process model creation - choose between: heuristic, inductive
miner_type = 'heuristic'
miner_type_list = ['heuristic', 'inductive']

# event case correlation export files
# folder containing files read and written during ecc classical method
dir_classic_event_case_correlation = 'ecc/' \
                                     'method-{event_case_correlation_method}/' \
                                     'vec_type-{vectorization_type}/' \
                                     'trace_length-{trace_length_limit}/' \
                                     'traces_time_out-{traces_time_out_threshold}/' \
                                     'distance-{distance_threshold}/'
# folder containing files read and written during ecc freflala method
dir_freflala_event_case_correlation = 'ecc/' \
                                      'method-{event_case_correlation_method}/' \
                                      'vec_type-{vectorization_type}/' \
                                      'trace_length-{trace_length_limit}/' \
                                      'traces_time_out-{traces_time_out_threshold}/' \
                                      'max_errors-{max_errors_per_day}/'
# filename of trace data file
filename_trace_data_time = 'trace_data_time.pickle'
# filename of traces cluster file
filename_output_case_traces_cluster = 'o_c_t_cluster.pickle'

# output files
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

# filename of log export file
filename_log_export = 'log_export.xes'

# folder containing petri net file
dir_petri_net_files = 'petri_nets/'
# filename of petri net .pnml file
filename_petri_net = 'petri_net.pnml'
# filename of petri net image file
filename_petri_net_image = 'ProcessModelHM.png'

# folder containing dfg png files
dir_dfg_files = 'directly_follows_graphs/'
# filename of dfg file (per cluster)
filename_dfg_cluster = 'DFG_Cluster_{cluster}.png'
# filename of dfg file
filename_dfg = 'DFG.png'
