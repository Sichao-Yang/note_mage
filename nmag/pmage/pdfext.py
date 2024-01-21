from pypdf import PdfWriter, PdfReader
import argparse
from os import path as osp
import sys

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from utils import *


def pdf_extract(pdf_in, pdf_out, start, end):
    writer = PdfWriter()
    # 读取pdf
    reader = PdfReader(pdf_in)
    logging.info(f"successfully read pdf: {pdf_in}, total {len(reader.pages)} pages")
    for i in range(start, end):
        writer.add_page(reader.pages[i])
    # 写出pdf
    logging.info(f"write page {start}-{end} to {pdf_out}")
    with open(pdf_out, "wb") as fp:
        writer.write(fp)


if __name__ == "__main__":
    logger = get_logger(filename="pdfext.log", verb_level="info", method="w2file")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        default=r"examples\pdfext\NumericalOptimization.pdf",
    )
    parser.add_argument(
        "--dst_path",
        type=str,
        default=r"examples\pdfext\out.pdf",
    )
    parser.add_argument(
        "--range",
        type=str,
        default="[5,20]",
    )
    args = parser.parse_args()
    rangelist = eval(args.range)
    pdf_extract(args.src_path, args.dst_path, rangelist[0], rangelist[1])
