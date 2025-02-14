import shutil
import os
import logging
from os import path as osp
import re

ROOT = osp.abspath(osp.join(osp.dirname(__file__), "../"))
pats = ["\!\[.*?\]\((.*?)\)", "\<img src=[\"'](.*?)[\"']"]
IMGPATTERNS = [re.compile(y) for y in pats]
IMGFORMAT = [".jpg", ".png"]


def get_logger(filename, verb_level="info", name=None, method=None):
    """filename: 保存log的文件名
    name：log对象的名字，可以不填
    """
    level_dict = {"debug": logging.DEBUG, "info": logging.INFO, "warn": logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s")
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


def build_workdir(path=""):
    if path == "":
        logging.info("build workdir from $ROOT")
        path = osp.join(ROOT, "workdir")
    if osp.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def remove_workdir(path):
    shutil.rmtree(path)


def backup_dir(src_dir):
    bak_dir = src_dir + "__bak"
    if osp.exists(bak_dir):
        shutil.rmtree(bak_dir)
    os.makedirs(bak_dir)
    shutil.copytree(src=src_dir, dst=bak_dir, dirs_exist_ok=True)
    return bak_dir


def backup_file(src_path):
    bak_path = src_path + "__bak"
    shutil.copy(src_path, bak_path)


def is_string_url(string):
    identifiers = ["https://", "http://"]
    for iden in identifiers:
        if iden in string:
            return True
    return False


def check_path(filepath):
    """check if the img path is
    1. hyper link, 2. absolute path
    then print them out and remove from path if remove==True
    """
    hps = [r"https://", r"http://"]
    # r"http://img.freepik.com/free-photo/abstract-"
    # r"https://img.freepik.com/free-photo/abstract-grunge-decorative-relief-navy-blue-stucco-wall-texture-wide-angle-rough-colored-background_1258-28311.jpg?w=2000"
    # r"C:/fdsjl/fd.png"
    if is_string_url(filepath):
        logging.warn(f"HyperLink detected in filepath: {filepath}")
        return True
    if osp.isabs(filepath):
        logging.warn(f"AbsolutePath detected in filepath: {filepath}")
        return True
    return False


def ext_path(string, pats):
    res = []
    for p in pats:
        res.extend(re.findall(p, string))
    return res
