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


def is_string_url(s):
    identifiers = ["https://", "http://"]
    for iden in identifiers:
        if iden in s:
            return True
    return False


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


def main(input_folder, output_folder):
    """
    move all img files with non-standard relative path into media/<file name>/ folder
    """
    if input_folder == output_folder:
        raise ValueError(
            "output folder should not be the same as input folder, otherwise data in input folder will be deleted!"
        )
    if osp.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    shutil.copytree(src=input_folder, dst=output_folder, dirs_exist_ok=True)
    # working only on output folder
    md_files = [
        osp.join(output_folder, x) for x in os.listdir(output_folder) if ".md" in x
    ]
    logging.info(
        f"Found {len(md_files)} markdown files from {input_folder}:\n{pformat(md_files)}"
    )
    global IMGPATTERNS
    compiled_p = [re.compile(y) for y in IMGPATTERNS]
    for f_path in md_files:
        logging.info(f">>> working on {f_path}")
        with open(f_path, "r", encoding="utf8") as fp:
            lines = fp.readlines()
        md_dir = osp.dirname(f_path)
        fn = osp.basename(f_path)[:-3]  # remove '.md' part of file name
        new_lines = copy.deepcopy(lines)
        moved_list = []
        # read the md file line by line to extract image path
        for lineno, line in enumerate(lines):
            for p in compiled_p:
                res = re.findall(pattern=p, string=line)
                if len(res) != 0:
                    pic_path = res[0]
                    # only make img folder when md file contains img
                    os.makedirs(osp.join(md_dir, f"media/{fn}"), exist_ok=True)
                    # if img_path exists and its not url, we continue to check if img_path is correct
                    if is_string_url(pic_path):
                        logging.warning(
                            f"Found url path!\n>>>FileName: {fn}, LineNo: {lineno}, ImagePath: {pic_path}"
                        )
                        continue
                    if not osp.isabs(pic_path):
                        pic_path_abs = osp.join(md_dir, osp.relpath(pic_path))
                    new_pic_path = path_correction(fn, pic_path_abs, md_dir)
                    if new_pic_path != "":
                        # move picture to the corrected path and add it to moved_list to avoid any future move
                        if pic_path_abs not in moved_list:
                            shutil.move(src=pic_path_abs, dst=new_pic_path)
                            moved_list.append(pic_path_abs)
                        # change content in md file lines
                        relative_new_pic_path = osp.relpath(new_pic_path, md_dir)
                        new_lines[lineno] = line.replace(
                            pic_path, relative_new_pic_path
                        )
                    break
        writeback(f_path, new_lines)


if __name__ == "__main__":
    logger = get_logger(
        filename="imgpath_correct.log", verb_level="info", method="w2file"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data",
        help="the folder that contains markdown files",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data",
        help="the goto folder for corrected markdown files",
    )
    args = parser.parse_args()
    main(args.input_folder, args.output_folder)
