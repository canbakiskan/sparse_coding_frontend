import os
import logging
import sys


def logger_setup(log_name):

    if not os.path.exists(os.path.dirname(log_name)):
        os.makedirs(os.path.dirname(log_name))

    logging.basicConfig(
        format="[%(asctime)s] - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_name),
            logging.StreamHandler(sys.stdout),
        ],
    )
