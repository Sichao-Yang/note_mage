import logging


IMGPATTERNS = ["\!\[.*?\]\((.*?)\)", "\<img src=[\"'](.*?)[\"']"]


def get_logger(filename, verb_level="info", name=None, method=None):
    """filename: 保存log的文件名
    name：log对象的名字，可以不填
    """
    level_dict = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verb_level])

    if method == "w2file":
        fh = logging.FileHandler(filename, mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
