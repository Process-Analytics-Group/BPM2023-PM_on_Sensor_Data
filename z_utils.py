import pandas as pd
import logging
import pathlib
import inspect


def read_csv_file(filedir, filename, separator, header, logging_level, parse_dates=None, dtype=None):
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

    try:
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
                                 dtype=dtype)

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
