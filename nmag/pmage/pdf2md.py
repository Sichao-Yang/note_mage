import os
from os import path as osp
from pdf2image import convert_from_path
import argparse
from pathlib import Path
from pprint import pformat
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def pdf_to_md_batch(src_dir, dst_dir):
    path_list = [osp.join(src_dir, x) for x in os.listdir(src_dir)]
    path_list = [osp.join(src_dir, x) for x in os.listdir(src_dir)]
    pdfpath_list = [x for x in path_list if osp.isfile(x) and "pdf" in Path(x).suffix]
    logging.info(f"Found {len(pdfpath_list)} pdfs,\n{pformat(pdfpath_list)}")
    for src_path in pdfpath_list:
        pdf_to_md(src_path, dst_dir)


def pdf_to_md(src_path, dst_dir):
    # make media/title folder
    title = osp.basename(src_path).split(".")[0]
    odir = osp.join(dst_dir, f"media/{title}")
    os.makedirs(odir, exist_ok=True)
    # store images to media/title folder
    logging.info(f"converting {title} to images")
    images = convert_from_path(src_path)
    for i in range(len(images)):
        filepath = osp.join(odir, "{:02d}.jpg".format(i))
        images[i].save(filepath, "JPEG")
    # generate markdown files with references to sorted images
    imgs = sorted(os.listdir(odir))
    outpath = osp.join(dst_dir, title + ".md")
    logging.info(f"generating markdown from images and write to {outpath}")
    with open(outpath, "w") as fp:
        fp.write(f"# {title}\n\n")
        for fn in imgs:
            fp.write(f"![img](media/{title}/{fn})\n\n")
    return outpath


if __name__ == "__main__":
    logger = get_logger(filename="pdf2md.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="examples/pdf2md",
    )
    args = parser.parse_args()

    if args.src_dir != "" and args.src_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

    if args.src_dir != "":
        pdf_to_md_batch(args.src_dir, args.dst_dir)
    else:
        pdf_to_md(args.src_path, args.dst_dir)
