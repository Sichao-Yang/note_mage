import os
import unittest
import sys
from os import path as osp
from pypdf import PdfReader

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from nmag.utils import build_workdir, remove_workdir, ROOT
from nmag.pmage.pdf2img import pdf_to_img
import shutil


class TestImgMage(unittest.TestCase):
    def setUp(self):
        self.workdir = build_workdir()
        self.srcdir = osp.join(self.workdir, "src")
        shutil.copytree(osp.join(ROOT, r"examples/pdfs"), dst=self.srcdir)

    def tearDown(self):
        remove_workdir(self.workdir)

    def test_imgpath_correct(self):
        ...

    def test_mdrename(self):
        ...

    def test_red_img_remove(self):
        ...


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestImgMage)
    runner = unittest.TextTestRunner()
    runner.run(suite)
