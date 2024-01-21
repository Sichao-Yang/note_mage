import os
from os import path as osp
from pdf2image import convert_from_path
import argparse
from pathlib import Path
from pprint import pformat
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def pdf_to_md_batch(input_folder, output_folder):
    path_list = [osp.join(input_folder, x) for x in os.listdir(input_folder)]
    path_list = [osp.join(input_folder, x) for x in os.listdir(input_folder)]
    pdfpath_list = [x for x in path_list if osp.isfile(x) and "pdf" in Path(x).suffix]
    logging.info(f"Found {len(pdfpath_list)} pdfs,\n{pformat(pdfpath_list)}")
    for input_path in pdfpath_list:
        pdf_to_md(input_path, output_folder)


def pdf_to_md(input_path, output_folder):
    # make media/title folder
    title = osp.basename(input_path).split(".")[0]
    odir = osp.join(output_folder, f"media/{title}")
    os.makedirs(odir, exist_ok=True)
    # store images to media/title folder
    logging.info(f"converting {title} to images")
    images = convert_from_path(input_path)
    for i in range(len(images)):
        filepath = osp.join(odir, "{:02d}.jpg".format(i))
        images[i].save(filepath, "JPEG")
    # generate markdown files with references to sorted images
    imgs = sorted(os.listdir(odir))
    outpath = osp.join(output_folder, title + ".md")
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
        "--input_folder",
        type=str,
        default="",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="examples/pdf2md",
    )
    args = parser.parse_args()

    if args.input_folder != "" and args.input_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

    if args.input_folder != "":
        pdf_to_md_batch(args.input_folder, args.output_folder)
    else:
        pdf_to_md(args.input_path, args.output_folder)
