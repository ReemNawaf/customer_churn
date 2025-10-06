import logging


def get_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s: %(message)s")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


# how to use the logger
# from src.logger import get_logger

# log = get_logger(__name__)
# log.info("starting data ingestion…")
