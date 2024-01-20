from pdf2image import convert_from_path
from os import path as osp
import argparse
import os
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def main(input_path, output_folder):
    images = convert_from_path(input_path)

    if not osp.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(len(images)):
        path = osp.join(output_folder, "p" + str(i) + ".jpg")
        images[i].save(path, "JPEG")


if __name__ == "__main__":
    logger = get_logger(filename="slide2md.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="examples/pdf2img/OR2_03_BnBandHeuristic.pdf",
        help="the folder that contains images",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="examples/pdf2img/OR2_03_BnBandHeuristic",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_folder)
