import sys
from os import path as osp
import os
import argparse
import shutil

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *
from md_manager.imgpath_corrector import main as imgpath_corrector


def renamer(file_path, new_name, leave_old_file):
    filerfolder = osp.dirname(file_path)
    if leave_old_file:
        shutil.copy(file_path, osp.join(filerfolder, new_name))
    else:
        os.rename(file_path, osp.join(filerfolder, new_name))
    imgpath_corrector(input_folder=filerfolder, output_folder=filerfolder + "_bak")


if __name__ == "__main__":
    logger = get_logger(filename="md_renamer.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="abc.md",
    )
    parser.add_argument(
        "--new_name",
        type=str,
        default="cba.md",
    )
    parser.add_argument(
        "--leave_old_file",
        action="store_false",
        default="cba.md",
    )
    args = parser.parse_args()
    renamer(args.file_path, args.new_name, args.leave_old_file)
