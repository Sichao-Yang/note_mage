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
from nmag.vid_duration import calc_vid_duration

nmag_desc = """
--------------------------NMag--------------------------
    Welcome to nmag, a toolset for file manipulations
"""


def get_args():
    parser = argparse.ArgumentParser(
        description=nmag_desc,
        usage='Use "%(prog)s --help|-h" for more information on the options',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-sd",
        "--src_dir",
        type=str,
        default="",
        help="str: the source folder, (default: '%(default)s')",
    )
    parser.add_argument(
        "-sp", "--src_path", type=str, default="", help="str: the source filepath, (default: '%(default)s')"
    )
    parser.add_argument(
        "-dd",
        "--dst_dir",
        type=str,
        default="",
        help="str: the output folder, (default: '%(default)s')",
    )
    parser.add_argument(
        "-dp",
        "--dst_path",
        type=str,
        default="",
        help="str: the output filepath, (default: '%(default)s')",
    )
    parser.add_argument(
        "-bak",
        "--backup",
        action="store_false",
        help="bool: backup original data to <output_filepath>__bak, (default: '%(default)s')",
    )
    parser.add_argument(
        "-direction",
        "--concat_direction",
        type=str,
        default="v",
        help="str: the direction for concatenation, v or h, (default: '%(default)s')",
    )
    parser.add_argument(
        "-auto_cip",
        "--auto_imgpath_change",
        action="store_true",
        help="bool: change imgpath automatically after renamed file, (default: '%(default)s')",
    )
    parser.add_argument(
        "-ignore",
        "--ignore_items",
        nargs="+",
        help="list: items to be ignored when collecting redundant files, (default: '%(default)s')",
        default=[".pdf", ".txt"],
    )
    parser.add_argument(
        "-range",
        type=str,
        default="[5,20]",
        help="str: extract pdf range from page a to b, (default: '%(default)s')",
    )
    parser.add_argument(
        "--mirror_rule",
        type=str,
        default="['0_MOOC_Videos', 'Drive/sync/0_notes_all/0_MOOC_notes']",
        help="str: replace path of src path with part of dst path, (default: '%(default)s')",
    )
    parser.add_argument(
        "--linking",
        action="store_false",
        help="bool: make shortcut link between two dirs, (default: '%(default)s')",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="bool: increase output verbosity, (default: '%(default)s')"
    )
    cmd_string = """
The available cmds are described in format of `cmd | desc | args`:
"icat":     | concatenate images                    | -sd, -direction, -dp
"cip":      | correct imgpath in md(s)              | -sd|-sp, -bak
"rir":      | remove redundant images from md       | -sd, -bak, -ignore
"rn":       | rename md                             | -sp, -dp, -auto_cip -bak
"p2i":      | pdf to img                            | -sp, -dd
"p2m":      | pdf(s) to md                          | -sp|-sd, -dp
"pe":       | extract subpages from pdf to pdf      | -sp, -dp, -range
"md":       | makedir on primal and mirror paths    | -sd, --mirror_rule, --linking
"vd":       | video durations in a given folder     | -sd
"""
    parser.add_argument("task", help=cmd_string)
    args = parser.parse_args()
    return args


def run():
    get_logger(filename="cli.log", verb_level="info")
    args = get_args()

    if args.src_dir != "" and args.src_path != "":
        raise ValueError("You can't set both input filepath and input folder!")

    if args.task == "icat":
        concat_img(args.src_dir, args.concat_direction, args.dst_path)
    elif args.task == "cip":
        if args.src_dir != "":
            correct_imgpath_batch(args.src_dir, backup=args.backup)
        else:
            correct_imgpath(args.src_path, backup=args.backup)
    elif args.task == "rir":
        tp = MdRedImgRemover(
            src_dir=args.src_dir,
            backup=args.backup,
            ignore_items=args.ignore_items,
        )
        tp.run()
    elif args.task == "rn":
        md_rename(
            args.src_path,
            args.dst_path,
            backup=args.backup,
            auto_imgpath_change=args.auto_imgpath_change,
        )
    elif args.task == "p2i":
        pdf_to_img(args.src_path, args.dst_dir)
    elif args.task == "p2m":
        if args.src_dir != "":
            pdf_to_md_batch(args.src_dir, args.dst_dir)
        else:
            pdf_to_md(args.src_path, args.dst_dir)
    elif args.task == "pe":
        rangelist = eval(args.range)
        pdf_extract(args.src_path, args.dst_path, rangelist[0], rangelist[1])
    elif args.task == "md":
        makedir(args.src_dir, eval(args.mirror_rule), args.linking)
    elif args.task == "vd":
        calc_vid_duration(args.src_dir)
    else:
        raise ValueError(f"Legal command doesnot include {args.task}")


if __name__ == "__main__":
    run()
