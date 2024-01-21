import sys
from os import path as osp
import os
import argparse
import shutil

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *
from mmage.imgpath_corrector import correct_imgpath


def md_rename(src_path, dst_path, leave_old_file, auto_imgpath_change):
    idir = osp.dirname(src_path)
    odir = osp.dirname(dst_path)
    if not osp.exists(odir):
        logging.info(f"output folder doesn't exist, make one: {odir}")
        os.makedirs(odir)
    # imgpath should be renamed -> corrected -> renamed!
    tmp_path = osp.join(idir, osp.basename(dst_path))
    if leave_old_file:
        if osp.basename(dst_path) == osp.basename(src_path):
            raise ValueError(
                "leave-old-file set to True, new_name should not be same as old name"
            )
        shutil.copy(src_path, tmp_path)
    else:
        os.rename(src_path, tmp_path)
    logging.info(f"file: {osp.basename(dst_path)} renamed to: {osp.basename(src_path)}")
    if auto_imgpath_change:
        imgfolder = correct_imgpath(src_path=tmp_path, backup=False)
    shutil.move(tmp_path, dst_path)
    if auto_imgpath_change and imgfolder != "":
        shutil.copytree(
            imgfolder,
            osp.join(osp.dirname(dst_path), f"media/{osp.basename(imgfolder)}"),
        )


if __name__ == "__main__":
    logger = get_logger(filename="md_renamer.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default="abc.md",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default="cba.md",
    )
    parser.add_argument(
        "--leave_old_file",
        action="store_false",
    )
    parser.add_argument(
        "--auto_imgpath_change",
        action="store_true",
    )
    args = parser.parse_args()
    md_rename(
        args.src_path,
        args.dst_path,
        args.leave_old_file,
        args.auto_imgpath_change,
    )
