import os
from os import path as osp
from win32com.client import Dispatch


def create_link(link_path, target_path):
    # link_path: Path to be saved (shortcut)
    # target: The shortcut target file or folder
    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(link_path)
    shortcut.Targetpath = target_path
    shortcut.save()


def makedir(
    path,
    mirror_rule=("0_MOOC_Videos", "Drive/sync/0_notes_all/0_MOOC_notes"),
    linking=True,
):
    full_path = osp.abspath(path)
    print(f"make primary dir: {full_path}")
    os.makedirs(full_path)
    if mirror_rule[0] in full_path:
        print(f"based on rule {mirror_rule}, this path is mirrorable")
        mirror_path = full_path.replace(mirror_rule[0], mirror_rule[1])
        print(f"make its mirror dir: {mirror_path}")
        if not osp.exists(osp.dirname(mirror_path)):
            raise ValueError(
                "the parent directory on mirror path doesnot exist, please check!",
                " this is normaly due to parent path inconsistency btw. primal and mirror",
            )
        os.makedirs(mirror_path)
    if linking:
        print("creating cross linking between two dirs")
        create_link(
            link_path=osp.join(mirror_path, "primary.lnk"), target_path=full_path
        )
        create_link(
            link_path=osp.join(full_path, "mirror.lnk"),
            target_path=osp.abspath(mirror_path),
        )
