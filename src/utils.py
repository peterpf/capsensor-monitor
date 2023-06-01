import logging

def configure_logger(name: str, level=logging.DEBUG) -> logging.Logger:
    """Configures logging.

    :param name: class name that instantiates the logger
    :type name: str
    :param level: Logging leve, defaults to DEBUG
    :type level: str
    :return: logging.Logger
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    return logger
