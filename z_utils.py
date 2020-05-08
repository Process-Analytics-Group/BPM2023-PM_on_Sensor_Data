import pandas as pd
import logging
import pathlib


def read_csv_file(filedir, filename, separator, header, parse_dates=None, dtype=None):
    # TODO Kommentierung vervollständigen
    """Reads a csv file and returns the content of the file as pandas data frame.

    :param filedir:
    :param filename:
    :param separator:
    :param header:
    :param parse_dates:
    :param dtype:
    :return: the content of the file as pandas data frame
    """

    try:
        file_path = pathlib.Path(filedir + filename)
        logging.info("Reading csv file '../%s", file_path)

        # reads csv file
        data_frame = pd.read_csv(file_path,
                                 sep=separator,
                                 header=header,
                                 parse_dates=parse_dates,
                                 dtype=dtype)

    # if there is no file the program ends
    except FileNotFoundError as err:
        print(err.args)
        logging.info("There is no file named '../%s", file_path)
        exit(1)

    return data_frame