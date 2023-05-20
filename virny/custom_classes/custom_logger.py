import logging


class CustomHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)
        fmt = '%(asctime)s %(filename)-18s %(levelname)-8s: %(message)s'
        fmt_date = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(fmt, fmt_date)
        self.setFormatter(formatter)


def get_logger(verbose: int = 0):
    logger = logging.getLogger('root')
    logger.setLevel(logging.INFO if verbose >= 1 else logging.WARNING)
    logging.disable(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(CustomHandler())

    return logger
