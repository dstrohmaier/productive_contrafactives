import logging

from pathlib import Path


def start_logging(
    logging_directory: Path,
    logger_name: str = "contrafactives",
    file_name: str = "run.log",
):
    logging_directory.mkdir(parents=True, exist_ok=True)

    logging_path = logging_directory / file_name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(logging_path)
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s:" "%(levelname)s:" "%(filename)s: " "%(message)s"
    )
    fh.setFormatter(formatter)

    for old_fh in logger.handlers:  # remove all old handlers
        logger.removeHandler(old_fh)
    logger.addHandler(fh)  # set the new handler

    logger.info("Started running")
    return logger


def log_model_device(model):
    logger = logging.getLogger("contrafactives")

    for name, parameter in model.named_parameters():
        logger.debug(f'Model parameter "{name}" on device: {parameter.device}')
