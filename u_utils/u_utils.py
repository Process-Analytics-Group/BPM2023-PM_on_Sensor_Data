import pandas as pd
import logging
import pathlib
import inspect
import numpy as np
import timeit
import z_setting_parameters as settings


def read_csv_file(filedir, filename, separator, header, parse_dates=None, dtype=None):
    """Reads a csv file and returns the content of the file as pandas data frame.

    :param filedir: The folder the file lies in.
    :param filename: Name of the File.
    :param separator: The character the which saparates each column in a csv file.
    :param header: Indicator at which line the data starts (length of the header)
    :param logging_level: level of logging
    :param parse_dates: Collection of the columns that should get parsed as a date.
    :param dtype: A mapping of data types to the columns in the file.
    :return: the content of the file as pandas data frame
    """
    logging_level = settings.logging_level
    try:
        # start timer
        t0_read_csv_files = timeit.default_timer()

        # creates path out of file dir and file name
        file_path = pathlib.Path(filedir + filename)

        # logger
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(logging_level)
        logger.info("Reading csv file '../%s'.", file_path)

        # reads csv file
        data_frame = pd.read_csv(file_path,
                                 sep=separator,
                                 header=header,
                                 parse_dates=parse_dates,
                                 dtype=dtype,
                                 error_bad_lines=False)

        # drop all lines without motion sensor
        # ToDo: in future versions allow for more sensor types if implemented
        index_names = data_frame[data_frame['SensorID'].str.startswith(settings.prefix_motion_sensor_id)].index
        data_frame = data_frame.loc[index_names]
        data_frame.reset_index(inplace=True, drop=True)

        # calculate how many data points there are
        number_of_data_points = data_frame.shape[0]

        # stop timer
        t1_read_csv_files = timeit.default_timer()
        # calculate runtime
        runtime_read_csv_files = np.round(t1_read_csv_files - t0_read_csv_files, 1)

        # logger
        logger = logging.getLogger(inspect.stack()[0][3])
        logger.setLevel(settings.logging_level)
        logger.info("Extracted %s data points from csv-File on disc in %s seconds",
                    number_of_data_points, runtime_read_csv_files)

    # if there is no file the program ends
    except FileNotFoundError as err:
        err_msg = str('There is no file named "../' + str(file_path) + '".')
        logger.error(err, err_msg)
        raise err


    return data_frame


def write_csv_file(data, filedir, filename, separator, logging_level):
    # creates path out of file dir and file name
    file_path = pathlib.Path(filedir + filename)

    # logger
    logger = logging.getLogger(inspect.stack()[0][3])
    logger.setLevel(logging_level)
    logger.info("Writing csv file to disk '../%s'.", file_path)

    data.to_csv(file_path, sep=separator)
