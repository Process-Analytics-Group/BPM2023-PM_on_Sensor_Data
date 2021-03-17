# Zuordnung Cluster und Aktivitäten
import numpy as np
import pandas as pd
import z_setting_parameters as settings


def create_event_log_files(cluster,
                           traces_vectorised,
                           output_case_traces_cluster):

    traces_vectorised['Cluster'] = pd.DataFrame(cluster)
    output_case_traces_cluster['Cluster'] = \
        traces_vectorised['Cluster'][output_case_traces_cluster['Case']].values

    # drop column where cluster = nan -> ToDo: investigate, why there is nan values
    output_case_traces_cluster = output_case_traces_cluster.dropna(subset=['Cluster'])

    traces_vectorised = traces_vectorised.sort_values(by=['Cluster'])
    traces_vectorised = traces_vectorised.reset_index(drop=True)

    # write traces to disk
    # time of sensor activations grouped by case
    traces_vectorised.to_csv(settings.path_data_sources + settings.dir_runtime_files + settings.filename_cluster,
                             sep=settings.csv_delimiter_cluster)

    output_case_traces_cluster.to_csv(settings.path_data_sources + settings.dir_runtime_files +
                                      settings.filename_cases_cluster,
                                      sep=settings.csv_delimiter_cases_cluster)

    return output_case_traces_cluster


def create_event_log_files_deprecated(trace_data_time, output_case_traces_cluster, k_means_cluster_ids, path_data_sources,
                           dir_runtime_files, sm, km, filename_cluster, csv_delimiter_cluster, filename_cases_cluster,
                           csv_delimiter_cases_cluster):
    # write best matching units (BMU) to output file
    trace_data_time['BMU'] = np.transpose(sm._bmu[0, :]).astype(int)

    # write k-Means Cluster to output file
    cluster_list = km.labels_[trace_data_time.BMU]
    trace_data_time['Cluster'] = pd.DataFrame(cluster_list)

    # write vanilla kmeans to output file
    trace_data_time['kMeansVanilla'] = np.transpose(k_means_cluster_ids).astype(int)

    # write BMU and Cluster to output_case file
    # so the raw_data from the beginning now also has clusters
    output_case_traces_cluster['Cluster'] = trace_data_time['Cluster'][output_case_traces_cluster['Case']].values
    output_case_traces_cluster['BMU'] = trace_data_time['BMU'][output_case_traces_cluster['Case']].values
    output_case_traces_cluster['kMeansVanilla'] = k_means_cluster_ids[output_case_traces_cluster['Case']-1]
    # drop column where cluster = nan -> ToDo: investigate, why there is nan values
    output_case_traces_cluster = output_case_traces_cluster.dropna(subset=['Cluster'])

    trace_data_time = trace_data_time.sort_values(by=['Cluster', 'BMU'])
    trace_data_time = trace_data_time.reset_index(drop=True)

    # write traces to disk
    # time of sensor activations grouped by case
    trace_data_time.to_csv(path_data_sources + dir_runtime_files + filename_cluster, sep=csv_delimiter_cluster)

    output_case_traces_cluster.to_csv(path_data_sources + dir_runtime_files + filename_cases_cluster,
                                      sep=csv_delimiter_cases_cluster)

    return output_case_traces_cluster
