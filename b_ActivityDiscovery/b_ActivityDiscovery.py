# SOM
import inspect
import logging

from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
from b_ActivityDiscovery.self_organizing_map.sompy import SOMFactory
import numpy as np
from u_utils import u_helper as helper
import z_setting_parameters as settings
import b_ActivityDiscovery.FreFlaLa.b_FreFlaLa as b_FreFlaLa


def choose_clustering_method(clustering_method,
                             number_of_clusters,
                             trace_data_without_case_number,
                             dir_runtime_files,
                             logging_level,
                             dict_distance_adjacency_sensor,
                             vectorization_type):
    cluster = None
    if clustering_method == 'SOM':
        k_means_cluster_ids, sm, km = cluster_and_classify_activities(
            trace_data_without_case_number=trace_data_without_case_number,
            number_of_clusters=number_of_clusters, K_opt=number_of_clusters, dir_runtime_files=dir_runtime_files)
        cluster = km.labels_[np.transpose(sm._bmu[0, :]).astype(int)]

    elif clustering_method == 'CustomDistance':
        # Clustering custom distance
        cluster = b_FreFlaLa.clustering_with_custom_distance_calculation(
            trace_data_without_case_number,
            dict_distance_adjacency_sensor,
            vectorization_type,
            number_of_clusters
            )

        pass
    elif clustering_method == 'k-Means':

        cluster = custom_kmeans(trace_data_without_case_number, number_of_clusters)
    else:
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(logging_level)
        error_msg = "'" + clustering_method + "' is not a valid clustering method. Please check the settings."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return cluster

def cluster_and_classify_activities(trace_data_without_case_number, number_of_clusters, K_opt,
                                    dir_runtime_files):
    # k-means clustering
    k_means_cluster_ids = custom_kmeans(data=trace_data_without_case_number, number_of_clusters=number_of_clusters)

    sm, km, quantization_error, topographic_error = self_organising_map(
        trace_data_without_case_number=trace_data_without_case_number, K_opt=K_opt,
        dir_runtime_files=dir_runtime_files)

    return k_means_cluster_ids, sm, km


# k-means Vanilla
def custom_kmeans(data, number_of_clusters):
    np_data_array = data.values
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(np_data_array)
    return kmeans.labels_


# SOM with k-means to cluster the best matching units
def self_organising_map(trace_data_without_case_number, K_opt, dir_runtime_files):
    path_data_sources = settings.path_data_sources
    filename_parameters_file = settings.filename_parameters_file
    number_of_motion_sensors = settings.number_of_motion_sensors
    logging_level = settings.logging_level
    # create list with sensor names
    names = []
    for sensor_number in range(0, number_of_motion_sensors - 1):
        names.append(str("Sensor " + str(sensor_number)))
    # create the SOM network and train it.
    # You can experiment with different normalizations and initializations
    sm = SOMFactory().build(trace_data_without_case_number.values, normalization='var',
                            initialization='pca',
                            component_names=names)
    sm.train(n_job=1, verbose=False, train_rough_len=2, train_finetune_len=5)

    # The quantization error: average distance between each data vector and its BMU.
    # The topographic error: the proportion of all data vectors for which first and second BMUs are not adjacent units.
    topographic_error = sm.calculate_topographic_error()
    quantization_error = np.mean(sm._bmu[1])
    helper.append_to_log_file(
        new_entry_to_log_variable='topographic_error',
        new_entry_to_log_value=topographic_error,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The topographic error: the proportion of all data '
                                     'vectors for which first and second BMUs are not adjacent units.')
    helper.append_to_log_file(
        new_entry_to_log_variable='quantization_error',
        new_entry_to_log_value=quantization_error,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The quantization error: '
                                     'average distance between each data vector and its BMU.')

    helper.append_to_log_file(
        new_entry_to_log_variable='k_means_number_of_clusters',
        new_entry_to_log_value=K_opt,
        path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Number of clusters used to do the vanilla k-means '
                                     'and Cluster the SOM-Neurons. ')

    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Topographic error = %s; Quantization error = %s", topographic_error, quantization_error)

    # component planes view
    from visualization.mapview import View2D

    view2D = View2D(10, 10, "rand data", text_size=12)
    view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True,
                path_data_sources=path_data_sources)

    # U-matrix plot
    from visualization.umatrix import UMatrixView

    umat = UMatrixView(width=10, height=10, title='U-matrix')
    umat.show(sm, path_data_sources=path_data_sources, dir_runtime_files=dir_runtime_files)

    # do the K-means clustering on the SOM grid, sweep across k = 2 to 20
    from visualization.hitmap import HitMapView

    K = 20  # stop at this k for SSE sweep
    # K_opt = 18  # optimal K already found
    [labels, km, norm_data] = sm.cluster(n_clusters=K, opt=K_opt)

    hits = HitMapView(20, 20, "Clustering", text_size=12)
    a = hits.show(sm, path_data_sources=path_data_sources, dir_runtime_files=dir_runtime_files)

    return sm, km, quantization_error, topographic_error
