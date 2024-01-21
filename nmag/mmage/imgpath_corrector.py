import os
from os import path as osp
import shutil
import re
import copy
import logging
import argparse
from pprint import pformat
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def writeback(f_path, new_lines):
    with open(f_path, "w", encoding="utf8") as fp:
        fp.writelines(new_lines)


def path_correction(fn, pic_path_abs, md_dir):
    new_pic_path = ""
    if fn not in pic_path_abs:
        # picture path is not inside file_name folder, get it corrected
        pic_name = osp.basename(pic_path_abs)
        new_pic_path = osp.join(md_dir, f"media/{fn}", pic_name)
    return new_pic_path


def correct_imgpath_batch(src_dir, backup=True):
    md_pathlist = [osp.join(src_dir, x) for x in os.listdir(src_dir) if ".md" in x]
    logging.info(
        f"Found {len(md_pathlist)} markdown files from {src_dir}:\n{pformat(md_pathlist)}"
    )
    art_dirs = []
    for src_path in md_pathlist:
        art_dirs.append(correct_imgpath(src_path, backup))
    return md_pathlist, art_dirs


def correct_imgpath(src_path, backup=True):
    """
    standardieze imgpath in markdown file, and copy all img sources into standard folder: media/<file name>/
    """
    # backup only the markdown file
    if backup:
        backup_file(src_path)

    logging.info(f">>> working on {src_path}")
    with open(src_path, "r", encoding="utf8") as fp:
        lines = fp.readlines()
    md_dir = osp.dirname(src_path)
    title = osp.basename(src_path)[:-3]  # remove '.md' part of file name to get title
    new_lines = copy.deepcopy(lines)
    copied_list = []
    # read the md file line by line to extract image path
    for lineno, line in enumerate(lines):
        for p in IMGPATTERNS:
            res = re.findall(pattern=p, string=line)
            if len(res) == 1:
                pic_path = res[0]
                # only make img folder when md file contains img
                os.makedirs(osp.join(md_dir, f"media/{title}"), exist_ok=True)
                # if img_path exists and its not url, we continue to check if img_path is correct
                if check_path(pic_path):
                    logging.warn(
                        f"Found error path!\n>>>FileName: {title}, LineNo: {lineno}"
                    )
                    continue
                if not osp.isabs(pic_path):
                    pic_path_abs = osp.join(md_dir, osp.relpath(pic_path))
                new_pic_path = path_correction(title, pic_path_abs, md_dir)
                if new_pic_path != "":
                    # move picture to the corrected path and add it to moved_list to avoid any future move
                    if pic_path_abs not in copied_list:
                        if not osp.exists(pic_path_abs):
                            logging.error(
                                f"Referred imgpath doesn't exsits!\n>>>{src_path}\n>>>{lineno}\n>>>{pic_path_abs}"
                            )
                            continue
                        shutil.copy(src=pic_path_abs, dst=new_pic_path)
                        copied_list.append(pic_path_abs)
                    # change content in md file lines
                    relative_new_pic_path = osp.relpath(new_pic_path, md_dir)
                    new_lines[lineno] = line.replace(pic_path, relative_new_pic_path)
                break
    writeback(src_path, new_lines)
    # if md file contains imgs return artifacts' folder
    if osp.exists(osp.join(md_dir, f"media/{title}")):
        return osp.join(md_dir, f"media/{title}")
    else:
        return ""


if __name__ == "__main__":
    logger = get_logger(
        filename="imgpath_correct.log", verb_level="info", method="w2file"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="",
        help="the folder that contains markdown files",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="data",
        help="the goto folder for corrected markdown files",
    )
    parser.add_argument(
        "--backup",
        action="store_false",
        help="backup original md file",
    )
    args = parser.parse_args()

    if args.src_dir != "" and args.src_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

    if args.src_dir != "":
        correct_imgpath_batch(args.src_dir, args.backup)
    else:
        correct_imgpath(args.src_path, args.backup)
