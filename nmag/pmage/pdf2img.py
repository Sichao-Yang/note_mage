from pdf2image import convert_from_path
from os import path as osp
import argparse
import os
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def pdf_to_img(src_path, dst_dir):
    images = convert_from_path(src_path)

    if not osp.exists(dst_dir):
        os.makedirs(dst_dir)

    for i in range(len(images)):
        path = osp.join(dst_dir, "p" + str(i) + ".jpg")
        images[i].save(path, "JPEG")


if __name__ == "__main__":
    logger = get_logger(filename="slide2md.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default="examples/pdf2img/OR2_03_BnBandHeuristic.pdf",
        help="the folder that contains images",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="examples/pdf2img/OR2_03_BnBandHeuristic",
    )
    args = parser.parse_args()
    pdf_to_img(args.src_path, args.dst_dir)
