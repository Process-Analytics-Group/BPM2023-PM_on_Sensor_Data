import z_main as main
from u_utils import u_helper as helper
import logging
import z_setting_parameters as settings

if __name__ == "__main__":

    helper.import_user_config()

    # define looging handlers (e.g. StreamHandler - writes log into console)
    handlers = [
        logging.FileHandler(settings.path_data_sources + settings.dir_runtime_files + settings.filename_log_file),
        logging.StreamHandler()]

    # Logger configuration
    helper.configure_logger(handlers)
    main.execute_process_model_discovery()