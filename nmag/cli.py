import os
import argparse
from pathlib import Path
from os import path as osp
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from nmag.image.img_concat import concat_img
from nmag.mmage.md_renamer import md_rename
from nmag.mmage.imgpath_corrector import correct_imgpath, correct_imgpath_batch
from nmag.mmage.red_img_remover import MdRedImgRemover
from nmag.pmage.pdf2img import pdf_to_img
from nmag.pmage.pdf2md import pdf_to_md, pdf_to_md_batch
from nmag.pmage.pdfext import pdf_extract
from nmag.utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sd",
        "--src_dir",
        type=str,
        default="",
        help="the source folder",
    )
    parser.add_argument("-sp", "--src_path", type=str, default="", help="source file")
    parser.add_argument(
        "-dd",
        "--dst_dir",
        type=str,
        default="",
        help="the output folder",
    )
    parser.add_argument(
        "-dp",
        "--dst_path",
        type=str,
        default="",
        help="the output path",
    )
    parser.add_argument(
        "-bak",
        "--backup",
        action="store_false",
        help="backup original data to <file_path>__bak",
    )
    parser.add_argument(
        "-direction",
        "--concat_direction",
        type=str,
        default="v",
        help="the direction for concatenation, v or h",
    )
    parser.add_argument(
        "-auto_cip",
        "--auto_imgpath_change",
        action="store_true",
        help="change imgpath automatically after renamed file",
    )
    parser.add_argument(
        "-ignore",
        "--ignore_items",
        nargs="+",
        help="items to be ignored when collecting redundant files",
        default=[".pdf", ".txt"],
    )
    parser.add_argument(
        "--range",
        type=str,
        default="[5,20]",
        help="extract pdf range from page a to b: '[a,b]'",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    parser.add_argument("cmd", default="")
    args = parser.parse_args()
    return args


def run():
    # logger = get_logger(filename="imgcat.log", verb_level="info", method="w2file")
    args = get_args()

    if args.src_dir != "" and args.src_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

    LEGAL_CMDs = ["icat", "cip", "rir", "rn", "p2i", "p2m", "pe"]
    if args.cmd not in LEGAL_CMDs:
        raise ValueError(f"Legal command doesnot include {args.cmd}")

    if args.cmd == "icat":
        concat_img(args.src_dir, args.concat_direction, args.dst_path)
    elif args.cmd == "cip":
        if args.src_dir != "":
            correct_imgpath_batch(args.src_dir, backup=args.backup)
        else:
            correct_imgpath(args.src_path, backup=args.backup)
    elif args.cmd == "rir":
        tp = MdRedImgRemover(
            src_dir=args.src_dir,
            backup=args.backup,
            ignore_items=args.ignore_items,
        )
        tp.run()
    elif args.cmd == "rn":
        md_rename(
            args.src_path,
            args.dst_path,
            backup=args.backup,
            auto_imgpath_change=args.auto_imgpath_change,
        )
    elif args.cmd == "p2i":
        pdf_to_img(args.src_path, args.dst_dir)
    elif args.cmd == "p2m":
        if args.src_dir != "":
            pdf_to_md_batch(args.src_dir, args.dst_dir)
        else:
            pdf_to_md(args.src_path, args.dst_dir)
    elif args.cmd == "pe":
        rangelist = eval(args.range)
        pdf_extract(args.src_path, args.dst_path, rangelist[0], rangelist[1])


run()
