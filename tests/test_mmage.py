import os
import unittest
import sys
from os import path as osp
from pypdf import PdfReader

sys.path.append(osp.abspath(osp.join(osp.dirname(__file__), "../")))
from nmag.utils import build_workdir, remove_workdir, ROOT
from nmag.mmage.md_renamer import md_rename
from nmag.mmage.imgpath_corrector import correct_imgpath, correct_imgpath_batch
from nmag.mmage.red_img_remover import MdRedImgRemover
import shutil


class TestMMage(unittest.TestCase):
    def setUp(self):
        self.workdir = build_workdir()
        self.src_dir = osp.join(self.workdir, "src")
        shutil.copytree(osp.join(ROOT, r"examples/mds"), dst=self.src_dir)

    def tearDown(self):
        remove_workdir(self.workdir)

    def test_correct_imgpath(self):
        src_path = osp.join(self.src_dir, "DimensionReduction.md")
        correct_imgpath(src_path, backup=True)
        self.assertTrue(osp.exists(src_path + "__bak"))
        art_dir = osp.join(self.src_dir, "media/DimensionReduction")
        self.assertTrue(len(os.listdir(art_dir)) == 14)

    def test_correct_imgpath2(self):
        src_path = osp.join(self.src_dir, "DimensionReduction.md")
        correct_imgpath(src_path, backup=False)
        self.assertTrue(~osp.exists(src_path + "__bak"))
        art_dir = osp.join(self.src_dir, "media/DimensionReduction")
        self.assertTrue(len(os.listdir(art_dir)) == 14)

    def test_correct_imgpath_batch(self):
        md_pathlist, art_dirs = correct_imgpath_batch(self.src_dir, backup=True)
        self.assertTrue(len(md_pathlist) == 3)
        for src_path, art_dir in zip(md_pathlist, art_dirs):
            self.assertTrue(osp.exists(src_path + "__bak"))
            if art_dir != "":
                self.assertTrue(osp.exists(art_dir), "art_dir")
        art_dir = osp.join(self.src_dir, "media/DimensionReduction")
        self.assertTrue(len(os.listdir(art_dir)) == 14)

    def test_mdrename(self):
        src_path = osp.join(self.src_dir, "DimensionReduction.md")
        dst_path = osp.join(self.workdir, "out/out.md")
        md_rename(src_path, dst_path, backup=True, auto_imgpath_change=False)
        self.assertTrue(osp.exists(dst_path))
        self.assertTrue(osp.exists(src_path))

    def test_mdrename2(self):
        src_path = osp.join(self.src_dir, "DimensionReduction.md")
        dst_path = osp.join(self.workdir, "out/out.md")
        md_rename(src_path, dst_path, backup=False, auto_imgpath_change=False)
        self.assertTrue(osp.exists(dst_path))
        self.assertTrue(~osp.exists(src_path))

    def test_mdrename3(self):
        src_path = osp.join(self.src_dir, "DimensionReduction.md")
        dst_path = osp.join(self.workdir, "out/out.md")
        md_rename(src_path, dst_path, backup=True, auto_imgpath_change=True)
        self.assertTrue(osp.exists(dst_path))
        self.assertTrue(osp.exists(src_path))
        art_dir = osp.join(self.workdir, "out/media/out")
        self.assertTrue(osp.exists(art_dir))
        os.listdir(art_dir)
        self.assertTrue(len(os.listdir(art_dir)) == 14)

    def test_red_img_remove(self):
        tp = MdRedImgRemover(
            src_dir=self.src_dir,
            backup=True,
            ignore_items=[".pdf", ".txt"],
        )
        tp.run()
        self.assertTrue(osp.exists(self.src_dir + "__bak"))
        self.assertTrue(osp.exists(osp.join(self.src_dir, "red_files/sample1.png")))
        self.assertTrue(osp.exists(osp.join(self.src_dir, "red_files/sample2.jpg")))
        self.assertTrue(osp.exists(osp.join(self.src_dir, "media/test.pdf")))
        self.assertTrue(osp.exists(osp.join(self.src_dir, "media/test.txt")))
        self.assertTrue(osp.exists(osp.join(self.src_dir, "media/test.md")))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMMage)
    runner = unittest.TextTestRunner()
    runner.run(suite)
