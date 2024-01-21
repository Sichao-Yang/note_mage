import os
import unittest
import sys
from os import path as osp
from pypdf import PdfReader

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from nmag.utils import build_workdir, remove_workdir, ROOT
from nmag.pmage.pdf2img import pdf_to_img
from nmag.pmage.pdf2md import pdf_to_md, pdf_to_md_batch
from nmag.pmage.pdfext import pdf_extract
import shutil
from random import randint


class TestImgMage(unittest.TestCase):
    def setUp(self):
        self.workdir = build_workdir()
        self.srcdir = osp.join(self.workdir, "src")
        shutil.copytree(osp.join(ROOT, r"examples/pdfs"), dst=self.srcdir)
        filelist = os.listdir(self.srcdir)
        self.pdf_src = osp.join(self.srcdir, filelist[randint(0, len(filelist) - 1)])
        self.title = osp.basename(self.pdf_src).split(".")[0]
        reader = PdfReader(self.pdf_src)
        self.expect_pages = len(reader.pages)

    def tearDown(self):
        remove_workdir(self.workdir)

    def test_to_img(self):
        output_folder = osp.join(self.workdir, "out")
        pdf_to_img(self.pdf_src, output_folder)
        self.assertTrue(osp.exists(output_folder))
        self.assertEqual(len(os.listdir(output_folder)), self.expect_pages)

    def test_to_md(self):
        output_folder = osp.join(self.workdir)
        outpath = pdf_to_md(self.pdf_src, output_folder)
        self.assertTrue(osp.exists(outpath))
        art_dir = osp.join(self.workdir, f"media/{self.title}")
        self.assertEqual(len(os.listdir(art_dir)), self.expect_pages)

    def test_to_md_batch(self):
        output_folder = osp.join(self.workdir, "out")
        pdf_to_md_batch(self.srcdir, output_folder)
        self.assertTrue(osp.exists(output_folder))
        for src in os.listdir(self.srcdir):
            title = osp.basename(src).split(".")[0]
            art_dir = osp.join(output_folder, f"media/{title}")
            self.assertTrue(len(os.listdir(art_dir)) != 0)

    def test_pdf_extract(self):
        rangelist = [3, 5]
        out_path = osp.join(self.workdir, self.title + ".pdf")
        pdf_extract(self.pdf_src, out_path, rangelist[0], rangelist[1])
        self.assertTrue(osp.exists(out_path))
        reader = PdfReader(out_path)
        self.assertEqual(len(reader.pages), 2)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestImgMage)
    runner = unittest.TextTestRunner()
    runner.run(suite)
