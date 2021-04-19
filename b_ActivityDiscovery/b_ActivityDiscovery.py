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


def choose_and_perform_clustering_method(clustering_method,
                                         number_of_clusters,
                                         trace_data_without_case_number,
                                         dir_runtime_files,
                                         logging_level,
                                         dict_distance_adjacency_sensor,
                                         vectorization_type,
                                         min_number_of_clusters = 3,
                                         max_number_of_clusters = 20):
    """
    This method manages the different clustering methods and starts the selected method. The results of the different
    clustering methods is in the same Format.

    @param clustering_method:               Specifying the clustering method that is used
    @param number_of_clusters:              Specifying the number of clusters. (not used for "k-Means-Elbow" and
                                            "k-Medoids-Elbow")
    @param trace_data_without_case_number:  List of all vectors that should be clustered
    @param dir_runtime_files:
    @param logging_level:
    @param dict_distance_adjacency_sensor:  Dictionary with distance- and adjacency-matrix
    @param vectorization_type:              Specifying if the values in the dataset are the number of occurrences of the
                                            sensor or the time a sensor was activated
    @param min_number_of_clusters:          minimum number of clusters for the elbow-method (only used in
                                            "k-Means-Elbow" and "k-Medoids-Elbow")
    @param max_number_of_clusters:          maximum number of clusters for the elbow-method (only used in
                                            "k-Means-Elbow" and "k-Medoids-Elbow")

    @return:                                result list returns cluster for each vector
    """
    # initialize variables

    clustering_result = None
    cluster_score = None

    # clustering with self organizing  map
    if clustering_method == 'SOM':
        k_means_cluster_ids, sm, km = cluster_and_classify_activities(
            trace_data_without_case_number=trace_data_without_case_number,
            number_of_clusters=number_of_clusters, K_opt=number_of_clusters, dir_runtime_files=dir_runtime_files)
        # use the k-means inertia as clustering score. Average distance to centroids
        cluster_score = km.inertia_
        clustering_result = km.labels_[np.transpose(sm._bmu[0, :]).astype(int)]

    # clustering with a custom distance calculation
    elif clustering_method == 'CustomDistance':
        clustering_result = b_FreFlaLa.clustering_with_custom_distance_calculation(
            trace_data_without_case_number,
            dict_distance_adjacency_sensor,
            vectorization_type,
            number_of_clusters
            )

    # clustering with k-means form sk-learn
    elif clustering_method == 'k-Means':

        clustering_result, cluster_score = b_FreFlaLa.clustering_kmeans(trace_data_without_case_number,
                                                                        number_of_clusters)

    # elbow method to choose optimal number of clusters and clustering with k-means
    elif clustering_method == 'k-Means-Elbow':

        clustering_result, cluster_score = b_FreFlaLa.elbow_method_kmeans(trace_data_without_case_number,
                                                                          min_number_of_clusters,
                                                                          max_number_of_clusters)

    # clustering with k-medoids form sk-learn-extra
    elif clustering_method == 'k-Medoids':

        clustering_result, cluster_score = b_FreFlaLa.clustering_k_medoids(trace_data_without_case_number,
                                                                           number_of_clusters)

    # elbow method to choose optimal number of clusters and clustering with k-medoids
    elif clustering_method == 'k-Medoids-Elbow':

        clustering_result, cluster_score = b_FreFlaLa.elbow_method_kmedoids(trace_data_without_case_number,
                                                                            min_number_of_clusters,
                                                                            max_number_of_clusters)

    else:
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(logging_level)
        error_msg = "'" + clustering_method + "' is not a valid clustering method. Please check the settings."
        logger.error(error_msg)
        raise ValueError(error_msg)

    return clustering_result

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
