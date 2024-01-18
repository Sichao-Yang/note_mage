import os
from os import path as osp
import shutil
import re
from warnings import warn
import copy
import logging
import argparse
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), '../../')))
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


def checkpath(fn, pic_path_abs, md_dir):
    new_pic_path = ""
    if fn not in pic_path_abs:
        # picture path is not inside file_name folder, get it corrected
        pic_name = osp.basename(pic_path_abs)
        new_pic_path = osp.join(md_dir, f"media/{fn}", pic_name)
    return new_pic_path


def main(data_dir):
    """
    move all img files with non-standard relative path into media/<file name>/ folder
    """
    bak_dir = osp.join(osp.dirname(data_dir), osp.basename(data_dir)+"_bak")
    if osp.exists(bak_dir):
        shutil.rmtree(bak_dir)
    os.makedirs(bak_dir, exist_ok=True)
    shutil.copytree(src=data_dir, dst=bak_dir, dirs_exist_ok=True)
    md_files = [osp.join(data_dir, x) for x in os.listdir(data_dir) if ".md" in x]
    global IMGPATTERNS
    patterns = IMGPATTERNS
    compiled_p = [re.compile(y) for y in patterns]
    for f_path in md_files:
        with open(f_path, "r", encoding="utf8") as fp:
            lines = fp.readlines()
        md_dir = osp.dirname(f_path)
        fn = osp.basename(f_path)[:-3]  # remove '.md' part of file name
        new_lines = copy.deepcopy(lines)
        moved_list = []
        # read the md file line by line to extract image path
        for lineno, line in enumerate(lines):
            for p in compiled_p:
                tmp = re.findall(pattern=p, string=line)
                if len(tmp) != 0:
                    pic_path = tmp[0]
                    # if img_path exists and its not url, we continue to check if img_path is correct
                    if is_string_url(pic_path):
                        logging.warning(f"Found url path!\n>>>FileName: {fn}, LineNo: {lineno}, ImagePath: {pic_path}")
                        continue
                    else:
                        if not osp.isabs(pic_path):
                            pic_path_abs = osp.join(md_dir, osp.relpath(pic_path))
                        new_pic_path = checkpath(fn, pic_path_abs, md_dir)
                        # move picture to the correct new path
                        if (not pic_path_abs in moved_list) and new_pic_path != "":
                            os.makedirs(osp.join(md_dir, f"media/{fn}"), exist_ok=True)
                            shutil.move(src=pic_path_abs, dst=new_pic_path)
                            moved_list.append(pic_path_abs)
                        if new_pic_path != "":
                            # change content in md file lines
                            relative_new_pic_path = osp.relpath(new_pic_path, md_dir)
                            new_lines[lineno] = line.replace(
                                pic_path, relative_new_pic_path
                            )
                            writeback(f_path, new_lines)
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="data",
        help="the folder that contains markdown files",
    )
    args = parser.parse_args()
    main(args.input_folder)
