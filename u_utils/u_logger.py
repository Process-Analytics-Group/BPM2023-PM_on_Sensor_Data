import os
import logging
import pandas as pd

import z_setting_parameters as settings


def configure_logger(handlers):
    """
    Creates the logger configuration and disable logging in external libraries which is not needed.
    :return:
    """
    logging.basicConfig(
        level=settings.logging_level,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] [%(name)s] %(message)s",
        handlers=handlers)

    # "disable" specific logger
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('hyperopt').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    logging.getLogger('graphviz').setLevel(logging.WARNING)
    # disable the warning on copying columns
    pd.options.mode.chained_assignment = None

    return


def initialise_logger():
    # define logging handlers (e.g. StreamHandler - writes log into console)
    handlers = [
        logging.FileHandler(settings.path_data_sources + settings.dir_runtime_files + settings.filename_log_file),
        logging.StreamHandler()]

    # Logger configuration
    configure_logger(handlers)
