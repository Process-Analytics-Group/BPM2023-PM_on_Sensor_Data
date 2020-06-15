# Zuordnung Cluster und Aktivitäten
from y_sompy import SOMFactory
import numpy as np
import pandas as pd
import z_helper


def create_event_log_files(trace_data_time, output_case_traces_cluster, trace_data_without_case_number,
                           k_means_cluster_ids, K_opt, path_data_sources, dir_runtime_files, filename_parameters_file,
                           logger):
    sm, km, quantization_error, topographic_error = merge_related_tracking_entries(
        trace_data_without_case_number=trace_data_without_case_number, K_opt=K_opt, path_data_sources=path_data_sources,
        dir_runtime_files=dir_runtime_files, filename_parameters_file=filename_parameters_file, logger=logger)

    # write best matching units (BMU) to output file
    trace_data_time['BMU'] = np.transpose(sm._bmu[0, :]).astype(int)

    # write k-Means Cluster to output file
    cluster_list = km.labels_[trace_data_time.BMU]
    trace_data_time['Cluster'] = pd.DataFrame(cluster_list)

    # write vanilla kmeans to output file
    trace_data_time['kMeansVanilla'] = np.transpose(k_means_cluster_ids).astype(int)

    # write BMU and Cluster to output_case file
    # so the raw_data from the beginning now also has clusters
    output_case_traces_cluster['Cluster'] = trace_data_time['Cluster'][
        output_case_traces_cluster['Case'] - 1].values
    output_case_traces_cluster['BMU'] = trace_data_time['BMU'][
        output_case_traces_cluster['Case'] - 1].values
    output_case_traces_cluster['kMeansVanilla'] = k_means_cluster_ids[
        output_case_traces_cluster['Case'] - 1]

    trace_data_time = trace_data_time.sort_values(by=['Cluster', 'BMU'])
    trace_data_time = trace_data_time.reset_index(drop=True)

    # write traces to disk
    trace_data_time.to_csv(path_data_sources + dir_runtime_files + '/Cluster.csv',
                           sep=';',
                           index=None)

    output_case_traces_cluster.to_csv(
        path_data_sources + dir_runtime_files + '/Cases_Cluster.csv',
        sep=';',
        index=None)

    return quantization_error, topographic_error


def merge_related_tracking_entries(trace_data_without_case_number, K_opt, path_data_sources, dir_runtime_files,
                                   filename_parameters_file, logger):
    # create list with sensor names
    names = []
    for sensor_number in range(0, 51):
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
    z_helper.append_to_log_file(
        new_entry_to_log_variable='topographic_error',
        new_entry_to_log_value=topographic_error,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The topographic error: the proportion of all data '
                                     'vectors for which first and second BMUs are not adjacent units.')
    z_helper.append_to_log_file(
        new_entry_to_log_variable='quantization_error',
        new_entry_to_log_value=quantization_error,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='The quantization error: '
                                     'average distance between each data vector and its BMU.')

    z_helper.append_to_log_file(
        new_entry_to_log_variable='k_means_number_of_clusters',
        new_entry_to_log_value=K_opt,
        dir_runtime_files=dir_runtime_files,
        filename_parameters_file=filename_parameters_file,
        new_entry_to_log_description='Number of clusters used to do the vanilla k-means '
                                     'and Cluster the SOM-Neurons. ')

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
    a = hits.show(sm, path_data_sources=path_data_sources,
                  dir_runtime_files=dir_runtime_files)

    return sm, km, quantization_error, topographic_error
