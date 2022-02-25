import os
import z_main as main
from u_utils import u_helper as helper
from u_utils import u_logger as logger
import z_setting_parameters as settings


if __name__ == "__main__":
    helper.import_user_config()

    # make directory if it does not exist yet
    os.makedirs(os.path.dirname(settings.path_data_sources + settings.dir_runtime_files), exist_ok=True)

    logger.initialise_logger()

    main.execute_process_model_discovery()
