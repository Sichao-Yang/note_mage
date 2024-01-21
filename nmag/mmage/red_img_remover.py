import os, sys
from os import path as osp
import re
import shutil
import argparse
import sys
from pprint import pformat

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *

DEBUG = False


def get_all_files(dir, filelist):
    ps = os.listdir(dir)
    ps = [osp.join(dir, p) for p in ps]
    fs = [p for p in ps if osp.isfile(p)]
    filelist.extend(fs)
    for f in fs:
        ps.remove(f)
    for d in ps:
        get_all_files(d, filelist)


def get_mdfiles(filelist):
    mdpaths = []
    for f in filelist:
        if ".md" in f:
            mdpaths.append(f)
    return mdpaths


def check_files(filelist, warn_fmt=[".md", ".pdf"]):
    for f in filelist:
        for fmt in warn_fmt:
            if fmt in f:
                logging.warn(
                    f"Warnning format({fmt}) detected in img filelist after md_file removed: {f}"
                )


def ext_path(string, pats):
    res = []
    for p in pats:
        res.extend(re.findall(p, string))
    return res


def get_red_paths(all_imgpaths, used_imgpaths):
    red_paths = all_imgpaths.copy()
    for p in used_imgpaths:
        # osp.abspath is necessary to convert '\' to '//' in windows system
        abspath = osp.abspath(p)
        red_paths.remove(abspath)
    if len(red_paths) != 0:
        logging.info(f"All redundant paths are:\n{pformat(red_paths)}")
    else:
        logging.info("There is no redundant path!")
    return red_paths


def referredpath_in_dir(referred_path, all_imgpaths):
    abspath = osp.abspath(referred_path)
    return abspath in all_imgpaths


class MdRedImgRemover:
    """this parser will take a typora markdown file as input and operate on it:
    1. folder path extracting:
        1. recursively get all paths in target folder
        2. extract md files from all paths, all paths include only img paths
        3. check img_paths and warn user if there is any pdf files
    2. extract using imgpaths from md files:
        1. recursively extract all "![]()" & "<img src=''...>" patterns in md filelist
        2. filter out all hyper-links and absolute paths e.g.: ![xxx](https://github.com/typora/typora-issues)
            notice: there maybe some hyper-links rendered as image or contents, it needs to be
            either manually or automatically checked & converted.
    3. redundant path removing:
        1. remove used_img_paths from all_src_imgs to get red_imgs
        2. move red_imgs to a redundant folder
    4. manual postprocess:
        1. manual verification of those red_imgs in red folder
        2. check if there is any unmatched relative path in md files (potential broken img link)
    """

    def __init__(self, src_dir, backup=True, ignore_items=[]) -> None:
        # work on src_dir, backup files into bak_dir
        if backup:
            backup_dir(src_dir)

        self.work_dir = osp.abspath(src_dir)
        self.ignore_items = ignore_items

    def get_imgpath_fromdir(self):
        logging.info(f"Processing folder: {self.work_dir}")
        all_imgpaths = []
        get_all_files(self.work_dir, all_imgpaths)
        logging.info(f"Total file count: {len(all_imgpaths)}")
        mdpaths = get_mdfiles(all_imgpaths)
        for f in mdpaths:
            all_imgpaths.remove(f)
        logging.info(f"Total md file count: {len(mdpaths)}\n{pformat(mdpaths)}")
        check_files(all_imgpaths)
        return all_imgpaths, mdpaths

    def img_path_frommd(self, mdpaths, all_imgpaths):
        # txt = "test ![fdajk fda ](media/a/bfds.png) testse ![fdajfdsk](meddifds a/b/bfds.png) test"
        logging.info(f"\nImg path extracting... check for hyper-link & abs path")
        used_imgpaths = []
        for lineno, md in enumerate(mdpaths):
            logging.info(f"extracting imgpath from md file: {osp.basename(md)}")
            dir = osp.dirname(md)
            with open(md, "r", encoding="utf8") as fp:
                lines = fp.readlines()
            for line in lines:
                # one line can contain multiple imgpaths
                matched_paths = ext_path(line, IMGPATTERNS)
                if len(matched_paths) != 0:
                    # check if its hyperlink or abs path, then remove it from img_path
                    ilegal_paths = []
                    for p in matched_paths:
                        if check_path(p):
                            ilegal_paths.append(p)
                    for i in ilegal_paths:
                        matched_paths.remove(i)
                    # convert relative path to abs path
                    ps = [osp.join(dir, path) for path in matched_paths]
                    for p in ps:
                        if not osp.exists(p):
                            logging.error(
                                f"Referred imgpath doesn't exsits!\n>>>{md}\n>>>{lineno}\n>>>{p}"
                            )
                            continue
                        if not referredpath_in_dir(p, all_imgpaths):
                            logging.error(
                                f"Referred imgpath ({p}) exists but not in the dir!"
                            )
                            continue
                        used_imgpaths.append(p)
        # one imgpath can be refered infinitely many times, so reduplication is needed
        used_imgpaths = list(set(used_imgpaths))
        return used_imgpaths

    def remove_red_paths(self, red_paths, method="manual_inspect"):
        if method == "manual_inspect":
            drop_dir = osp.join(self.work_dir, "red_files")
            os.makedirs(drop_dir, exist_ok=True)
            for path in red_paths:
                ignore = False
                for ignore_item in self.ignore_items:
                    if ignore_item in path:
                        ignore = True
                if ignore:
                    continue
                opath = path.replace(self.work_dir, drop_dir)
                d = osp.dirname(opath)
                if not osp.exists(d):
                    os.makedirs(d)
                try:
                    shutil.move(path, opath)
                except Exception as e:
                    logging.warn(e)
            logging.info(
                f"All redundant files are moved to {drop_dir} waiting for manual insepection"
            )

    def run(self):
        all_imgpaths, mdpaths = self.get_imgpath_fromdir()
        used_imgpaths = self.img_path_frommd(mdpaths, all_imgpaths)
        red_paths = get_red_paths(all_imgpaths, used_imgpaths)
        if len(red_paths) != 0:
            self.remove_red_paths(red_paths, method="manual_inspect")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="data",
        help="the folder that contains markdown files",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="data",
        help="the folder that contains new markdown files",
    )
    parser.add_argument(
        "--ignore_items",
        nargs="+",
        help="the folder that contains markdown files",
        default=[".pdf", ".txt"],
    )
    args = parser.parse_args()

    logger = get_logger(filename="remover.log", verb_level="info", method="w2file")

    tp = MdRedImgRemover(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        ignore_items=args.ignore_items,
    )
    tp.run()
