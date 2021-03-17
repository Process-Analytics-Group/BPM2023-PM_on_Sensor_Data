import inspect
import logging
import os
import pandas as pd
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


def apply_heuristic_miner(log,
                          path_data_sources,
                          dir_runtime_files,
                          dir_petri_net_files,
                          filename_petri_net,
                          rel_proportion_dfg_threshold,
                          logging_level):
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
    log_converted = log_converter.apply(log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

    net, initial_marking, final_marking = heuristics_miner.apply(log_converted, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.8})

    fitness = replay_fitness_evaluator.apply(log_converted, net, initial_marking, final_marking,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    precision = precision_evaluator.apply(log, net, initial_marking, final_marking,
                                          variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    # generalization = generalization_evaluator.apply(log, net, im, fm)
    # simplicity = simplicity_evaluator.apply(net)

    # ToDo: DJ uncomment metrics & metrics calculation above
    metrics = {'fitness': fitness,
                          'precision': precision,
               #           'generalization': generalization,
               #           'simplicity': simplicity
               }

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)

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
