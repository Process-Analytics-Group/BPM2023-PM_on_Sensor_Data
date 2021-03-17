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
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.objects.petri.exporter import exporter as pnml_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
import z_setting_parameters as settings


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
                         rel_proportion_dfg_threshold, miner_type, logging_level, metric_to_be_maximised):
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
    metrics = apply_miner(log=pm4py_log,
                          path_data_sources=path_data_sources,
                          dir_runtime_files=dir_runtime_files,
                          dir_petri_net_files=dir_petri_net_files,
                          filename_petri_net=filename_petri_net,
                          miner_type=miner_type,
                          metric_to_be_maximised=metric_to_be_maximised)

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


def apply_miner(log,
                path_data_sources,
                dir_runtime_files,
                dir_petri_net_files,
                filename_petri_net,
                miner_type,
                metric_to_be_maximised):
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    log_converted = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    # Choose miner type:
    if miner_type == 'heuristic':
        logger.info("Applying heuristic miner to log.")
        net, initial_marking, final_marking = heuristics_miner.apply(log_converted, parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8})
        logger.info("Done with heuristic miner.")
    elif miner_type == 'inductive':
        logger.info("Applying inductive miner to log.")
        net, initial_marking, final_marking = inductive_miner.apply(log_converted)
        logger.info("Done with inductive miner.")
    else:
        return None

    # Choose target value to be maximised
    if metric_to_be_maximised == 'Fitness':
        logger.info("Calculating replay fitness.")
        precision = replay_fitness_evaluator.apply(log_converted, net, initial_marking, final_marking,
                                                   variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        logger.info("Replay fitness calculated.")
        metrics = precision['log_fitness']
    elif metric_to_be_maximised == 'Precision':
        logger.info("Calculating precision.")
        metrics = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                            variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        logger.info("Precision calculated.")
    else:
        metrics = None

    # alternative metrics
    # generalization = generalization_evaluator.apply(log, net, im, fm)
    # simplicity = simplicity_evaluator.apply(net)

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(settings.logging_level)

    # create directory for petri net files
    os.mkdir(path_data_sources + dir_runtime_files + dir_petri_net_files)

    # export petri net png
    gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    pn_visualizer.save(gviz, path_data_sources + dir_runtime_files + dir_petri_net_files + (str('ProcessModelHM.png')))

    # export petri net pnml
    pnml_exporter.apply(net, initial_marking,
                        path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net,
                        final_marking=final_marking)
    logger.info("Exported petri net pnml file into '../%s'.",
                path_data_sources + dir_runtime_files + dir_petri_net_files + filename_petri_net)

    return metrics
