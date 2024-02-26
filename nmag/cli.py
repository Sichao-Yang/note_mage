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
from nmag.utils import get_logger
from nmag.fmage import makedir


def get_args():
    parser = argparse.ArgumentParser(
        description="details",
        usage='use "%(prog)s --help" for more information',
        formatter_class=argparse.RawTextHelpFormatter,
    )
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
        help="the output filepath",
    )
    parser.add_argument(
        "-bak",
        "--backup",
        action="store_false",
        help="backup original data to <output filepath>__bak",
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
        "-range",
        type=str,
        default="[5,20]",
        help="extract pdf range from page a to b: '[a,b]'",
    )
    parser.add_argument(
        "--mirror_rule",
        type=str,
        default="['0_MOOC_Videos', 'Drive/sync/0_notes_all/0_MOOC_notes']",
        help="replace path of src path with part of dst path",
    )
    parser.add_argument(
        "--linking",
        action="store_false",
        help="make shortcut link between two dirs",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    cmd_string = """The available cmds are described in format of `cmd | desc | args`:
    "icat":     | concatenate images                    | -sd, -direction, -dp
    "cip":      | correct imgpath in md(s)              | -sd|-sp, -bak
    "rir":      | remove redundant images from md       | -sd, -bak, -ignore
    "rn":       | rename md                             | -sp, -dp, -auto_cip -bak
    "p2i":      | pdf to img                            | -sp, -dd
    "p2m":      | pdf(s) to md                          | -sp|-sd, -dp
    "pe":       | extract subpages from pdf to pdf      | -sp, -dp, -range
    "md":       | makedir on primal and mirror paths    | -sd, --mirror_rule, --linking"""
    parser.add_argument("cmd", help=cmd_string)
    args = parser.parse_args()
    return args


def run():
    get_logger(filename="cli.log", verb_level="info")
    args = get_args()

    if args.src_dir != "" and args.src_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

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
    elif args.cmd == "md":
        makedir(args.src_dir, eval(args.mirror_rule), args.linking)
    else:
        raise ValueError(f"Legal command doesnot include {args.cmd}")


if __name__ == "__main__":
    run()
