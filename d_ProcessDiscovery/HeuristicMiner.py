from tkinter import filedialog


import os
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.evaluation.generalization import evaluator as generalization_evaluator
from pm4py.evaluation.simplicity import evaluator as simplicity_evaluator


def applyHeuristicMiner(log,
                        path_data_sources,
                        dir_runtime_files,
                        dir_dfg_cluster_files,
                        filename_dfg_cluster,
                        rel_proportion_dfg_threshold,
                        logging_level):
    print("Prepare Heuristic Miner")

    log['Date'] = log['Timestamp'].dt.date

    log = log.reindex(columns={'Case', 'Timestamp', 'Cluster', 'Date'})
    log = log.rename(columns={'Date': 'case:concept:name',
                              'Cluster': 'concept:name',
                              'Timestamp': 'time:timestamp'})
    log = log.astype(str)
    log['time:timestamp'] = pd.to_datetime(log['time:timestamp'])
    net, im, fm = heuristics_miner.apply(log, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5})

    fitness = replay_fitness_evaluator.apply(log, net, im, fm,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    # precision = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    # generalization = generalization_evaluator.apply(log, net, im, fm)
    # simplicity = simplicity_evaluator.apply(net)

    metrics = {'Fitness': fitness,
    #           'Precision': precision,
    #           'Generalization': generalization,
    #           'Simplicity': simplicity
               }

    gviz = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz, path_data_sources + dir_runtime_files + dir_dfg_cluster_files + (str('ProcessModel.png')))

    pn_visualizer.view(gviz)

    print("Heuristic Miner completed.")

    return metrics