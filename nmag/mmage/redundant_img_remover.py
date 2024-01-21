import os, sys
from os import path as osp
import re
import shutil
import argparse
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *

DEBUG = False


class typora_img_red_remover:
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

    def __init__(self, input_folder, output_folder, ignore_formats) -> None:
        # working only on output_folder
        if osp.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)
        shutil.copytree(src=input_folder, dst=output_folder, dirs_exist_ok=True)

        self.work_dir = osp.abspath(output_folder)
        self.ignore_items = ignore_formats

    def img_src_extract(self):
        logger.info(f"Processing folder: {self.work_dir}")
        self.all_src_imgs = []
        self.ext_all_files(self.work_dir)
        logger.info(f"Total file count: {len(self.all_src_imgs)}")
        self.ext_mds()
        logger.info(f"Total md file count: {len(self.mdpaths)}")
        fns = ""
        for x in self.mdpaths:
            fns += x + "\n"
        logger.info(f"All md in this folder:\n {fns}")
        self.check_files()

    def ext_mds(self):
        self.mdpaths = []
        for f in self.all_src_imgs:
            if ".md" in f:
                self.mdpaths.append(f)
        for f in self.mdpaths:
            self.all_src_imgs.remove(f)

    def check_files(self, warn_fmt=[".md", ".pdf"]):
        for f in self.all_src_imgs:
            for fmt in warn_fmt:
                if fmt in f:
                    logger.warn(
                        f"Warnning format({fmt}) detected in img filelist after md remover: {f}"
                    )

    def ext_all_files(self, dir):
        ps = os.listdir(dir)
        ps = [osp.join(dir, p) for p in ps]
        try:
            ps.remove(self.mdpath)
        except:
            pass
        fs = [p for p in ps if osp.isfile(p)]
        self.all_src_imgs.extend(fs)
        for f in fs:
            ps.remove(f)
        for d in ps:
            self.ext_all_files(d)

    def ext_path(self, txt):
        res = []
        for y in self.x:
            res.extend(re.findall(y, txt))
        return res

    def check_path(self, fp, md):
        """check if the img path is
        1. hyper link, 2. absolute path
        then print them out and remove from path if remove==True
        """
        hps = [r"https://", r"http://"]
        # self.all_used_imgs.append(r"http://img.freepik.com/free-photo/abstract-")
        # self.all_used_imgs.append(r"https://img.freepik.com/free-photo/abstract-grunge-decorative-relief-navy-blue-stucco-wall-texture-wide-angle-rough-colored-background_1258-28311.jpg?w=2000")
        # self.all_used_imgs.append(r"C:/fdsjl/fd.png")
        for hp in hps:
            if hp in fp:
                logger.warn(f"HyperLink detected in md imgpath: \n{fp}\n{md}")
                return True
        if osp.isabs(fp):
            logger.warn(f"AbsolutePath detected in md imgpath: \n{fp}\n{md}")
            return True
        return False

    def img_used_extract(self, remove=True):
        """所有插入图片的格式是：![]()，这里loop md文档，然后把所有()里的路径抓出来,
        注意：
        因为可能有hyperlink图片和absolute path的图片，默认都要把这两种去掉
        因为可能有多个图片引用同一地址的情况，所以要做一下set处理
        """
        # txt = "test ![fdajk fda ](media/a/bfds.png) testse ![fdajfdsk](meddifds a/b/bfds.png) test"
        logger.info(
            f"\nImg path extracting... check for hyper-link & abs path, remove method=={remove}"
        )
        global IMGPATTERNS
        patterns = IMGPATTERNS
        if remove:
            logger.info(
                "[Imgpath Extraction]Auto remove for AbsolutePath and HyperLink is opened"
            )
        self.x = [re.compile(y) for y in patterns]

        self.all_used_imgs = []
        for md in self.mdpaths:
            dir = osp.dirname(md)
            with open(md, "r", encoding="utf8") as fp:
                line = fp.readline()
                while line:
                    ps = self.ext_path(line)
                    if len(ps) != 0:
                        rs = []
                        # check if its hyperlink or abs path, then remove it from img_path
                        for p in ps:
                            if remove and self.check_path(p, md):
                                rs.append(p)
                        for r in rs:
                            ps.remove(r)
                        # convert relative path to abs path
                        ps = [osp.join(dir, p) for p in ps]
                        self.all_used_imgs.extend(ps)
                    line = fp.readline()
        self.all_used_imgs = list(set(self.all_used_imgs))

    def get_red_paths(self):
        self.red_paths = self.all_src_imgs.copy()
        # self.all_used_imgs = [osp.abspath(osp.join(self.work_dir,i)) for i in self.all_used_imgs]
        for i in self.all_used_imgs:
            try:
                self.red_paths.remove(
                    osp.abspath(i)
                )  # osp.abspath is necessary to convert '\' to '//' in windows system
                logger.debug(f"{i} in dir")
            except Exception:
                logger.error(f"Img path ({i}) is not in the dir!")
        if len(self.red_paths) != 0:
            logger.info("\nAll redundant paths are:")
            for i in self.red_paths:
                logger.info(f"{i}")
        else:
            logger.info("There is no redundant path!")

    def remove_red_paths(self, method="manual_veri"):
        if method == "manual_veri":
            tar = osp.join(self.work_dir, "red_files")
            os.makedirs(tar, exist_ok=True)
            for i in self.red_paths:
                ignore = False
                o = i.replace(self.work_dir, tar)
                d = osp.dirname(o)
                if not osp.exists(d):
                    os.makedirs(d)

                for ignore_item in self.ignore_items:
                    if ignore_item in i:
                        ignore = True
                if ignore:
                    continue
                try:
                    shutil.move(i, o)
                except Exception as e:
                    logger.warn(e)
            logger.info(
                f"All redundant files are moved to {tar} waiting for manual verification"
            )

    def run(self):
        self.img_src_extract()
        self.img_used_extract()

        self.get_red_paths()
        if len(self.red_paths) != 0:
            self.remove_red_paths(method="manual_veri")


if __name__ == "__main__":
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
        help="the folder that contains new markdown files",
    )
    parser.add_argument(
        "--ignore_formats",
        nargs="+",
        help="the folder that contains markdown files",
        default=[],
    )
    args = parser.parse_args()

    logger = get_logger(filename="remover.log", verb_level="info", method="w2file")

    tp = typora_img_red_remover(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        ignore_formats=args.ignore_formats,
    )
    tp.run()
