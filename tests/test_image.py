import os
import unittest
import sys
from os import path as osp
import shutil

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from nmag.utils import build_workdir, remove_workdir, ROOT
from nmag.image.img_concat import concate_to_pdf, concate_imgs, resize_all
from pathlib import Path
from PIL import Image


class TestIMage(unittest.TestCase):
    def setUp(self):
        self.workdir = build_workdir()
        src = osp.join(ROOT, r"examples\img_concat")
        img_src = osp.join(self.workdir, "src")
        shutil.copytree(src, dst=img_src)

        self.supported_img_format = [".jpg", ".png"]

        filelist = [
            osp.join(img_src, x)
            for x in sorted(os.listdir(img_src))
            if Path(x).suffix in self.supported_img_format
        ]
        self.images = [Image.open(x) for x in filelist]

    def tearDown(self):
        remove_workdir(self.workdir)

    def test_imgconcat_img(self):
        dst_path = osp.join(self.workdir, "o.png")
        concat_direction = "h"
        concate_imgs(
            resize_all(self.images), direction=concat_direction, file_path=dst_path
        )
        self.assertTrue(osp.exists(dst_path))

    def test_imgconcat_pdf(self):
        dst_path = osp.join(self.workdir, "o.pdf")
        concate_to_pdf(resize_all(self.images), file_path=dst_path)
        self.assertTrue(osp.exists(dst_path))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIMage)
    runner = unittest.TextTestRunner()
    runner.run(suite)
