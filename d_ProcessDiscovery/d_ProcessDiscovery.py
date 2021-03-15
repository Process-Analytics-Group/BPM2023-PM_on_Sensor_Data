import inspect
import logging
import os
import pandas as pd
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.statistics.attributes.log import get as attributes_get
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.statistics.start_activities.log import get as sa_get
from pm4py.statistics.end_activities.log import get as ea_get
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from d_ProcessDiscovery.miner.HeuristicMiner import apply_heuristic_miner
from d_ProcessDiscovery.miner.InductiveMiner import apply_inductive_miner


def create_activtiy_models(output_case_traces_cluster, path_data_sources, dir_runtime_files, dir_dfg_cluster_files,
                           filename_dfg_cluster, rel_proportion_dfg_threshold, logging_level):
    """
    Creates directly follows graphs out of an event log.
    :param output_case_traces_cluster: traces that are visualised
    :param path_data_sources: path of sources and outputs
    :param dir_runtime_files: folder containing files read and written during runtime
    :param dir_dfg_cluster_files: folder containing dfg png files
    :param filename_dfg_cluster: filename of dfg file (per cluster)
    :param rel_proportion_dfg_threshold: threshold for filtering out sensors in dfg relative to max occurrences of a sensor
    :param logging_level: level of logging
    :return:
    """

    # keep only needed columns
    output_case_traces_cluster = output_case_traces_cluster.reindex(
        columns={'Case', 'LC_Activity', 'Timestamp', 'Cluster'})
    output_case_traces_cluster = output_case_traces_cluster.rename(
        columns={'Case': 'case:concept:name',
                 'LC_Activity': 'concept:name',
                 'Timestamp': 'time:timestamp'})

    # create directory for dfg pngs
    os.mkdir(path_data_sources + dir_runtime_files + dir_dfg_cluster_files)
    # create dfg for each cluster
    clusters = output_case_traces_cluster.Cluster.unique()
    for cluster in clusters:
        log = output_case_traces_cluster.loc[output_case_traces_cluster.Cluster == cluster]
        log = log.astype(str)
        log['time:timestamp'] = pd.to_datetime(output_case_traces_cluster['time:timestamp'])
        # convert pandas data frame to pm4py event log for further processing
        log = log_converter.apply(log)

        # keep only activities with more than certain number of occurrences
        activities = attributes_get.get_attribute_values(log, 'concept:name')
        # determine that number relative to the max number of occurrences of a sensor in a cluster. (the result is
        # the threshold at which an activity/activity strand is kept)
        min_number_of_occurrences = round((max(activities.values()) * rel_proportion_dfg_threshold), 0)
        activities = {x: y for x, y in activities.items() if y >= min_number_of_occurrences}
        log = attributes_filter.apply(log, activities)

        # create png dfg file
        exportDFGImageFile(log=log,
                           path_data_sources=path_data_sources,
                           dir_runtime_files=dir_runtime_files,
                           dir_dfg_files=dir_dfg_cluster_files,
                           filename_dfg=filename_dfg_cluster.format(cluster=str(cluster)))

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Saved directly follows graphs for each cluster into '../%s'.",
                path_data_sources + dir_runtime_files + dir_dfg_cluster_files)


def create_process_model(output_case_traces_cluster, path_data_sources, dir_runtime_files, filename_log_export,
                         dir_petri_net_files, filename_petri_net, dir_dfg_files, filename_dfg,
                         rel_proportion_dfg_threshold, miner_type, logging_level):
    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)

    # create a log that can be understood by pm4py
    pm4py_log = convert_log_to_pm4py(log=output_case_traces_cluster)

    # export log as xes file
    xes_exporter.apply(pm4py_log, path_data_sources + dir_runtime_files + filename_log_export)
    logger.info("Exported log export into '../%s'.", path_data_sources + dir_runtime_files + filename_log_export)

    # create png dfg file
    exportDFGImageFile(log=pm4py_log,
                       path_data_sources=path_data_sources,
                       dir_runtime_files=dir_runtime_files,
                       dir_dfg_files=dir_dfg_files,
                       filename_dfg=filename_dfg)

    logger.info("Saved directly follows graph into '../%s'.",
                path_data_sources + dir_runtime_files + dir_dfg_files + filename_dfg)


    if miner_type == 'heuristic':
        metrics = apply_heuristic_miner(log=pm4py_log,
                                        path_data_sources=path_data_sources,
                                        dir_runtime_files=dir_runtime_files,
                                        dir_petri_net_files=dir_petri_net_files,
                                        filename_petri_net=filename_petri_net,
                                        rel_proportion_dfg_threshold=rel_proportion_dfg_threshold,
                                        logging_level=logging_level)
        logger.info("Applied heuristic miner to log.")
    elif miner_type == 'inductive':
        metrics = apply_inductive_miner(log=pm4py_log,
                                        path_data_sources=path_data_sources,
                                        dir_runtime_files=dir_runtime_files,
                                        dir_petri_net_files=dir_petri_net_files,
                                        filename_petri_net=filename_petri_net,
                                        rel_proportion_dfg_threshold=rel_proportion_dfg_threshold,
                                        logging_level=logging_level)
        logger.info("Applied inductive miner to log.")

    return metrics


def convert_log_to_pm4py(log):
    log['Date'] = log['Timestamp'].dt.date

    log = log.reindex(columns={'Case', 'Timestamp', 'Cluster', 'Date'})
    log = log.rename(columns={'Date': 'case:concept:name',
                              'Cluster': 'concept:name',
                              'Timestamp': 'time:timestamp'})
    log = log.astype(str)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])

    # reduce log entries, so that every case only appears once
    log.drop_duplicates(subset=['Case'], inplace=True)

    # reset row index
    log = log.reset_index(drop=True)

    return log


def exportDFGImageFile(log, path_data_sources, dir_runtime_files, dir_dfg_files, filename_dfg):
    # create dfg out of event log
    dfg = dfg_discovery.apply(log)

    # define start and
    start_activities = sa_get.get_start_activities(log)
    end_activities = ea_get.get_end_activities(log)

    # create png of dfg (if the graph does not show a graph, it is possible that the sensors did not trigger often)
    # parameter has to be dfg0, because apply method requires a dfg0 mehtod, maybe depending on the pm4py version?
    gviz = dfg_visualization.apply(dfg0=dfg, log=log, variant=dfg_visualization.Variants.FREQUENCY,
                                   parameters={'start_activities': start_activities,
                                               'end_activities': end_activities})
    dfg_visualization.save(gviz, path_data_sources + dir_runtime_files + dir_dfg_files + (
        filename_dfg))

    return
