import os
from os import path as osp
from win32com.client import Dispatch


def get_relative_path(src, dst):
    if src == "":
        src = os.getcwd()
    assert dst != "", "You need to set the target path"
    relative_path = os.path.relpath(dst, src)
    print(f"Relative path from\n{src}\nto\n{dst}\nis:\n{relative_path}")


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
        print(f"found {mirror_rule[0]} part in the src path")
        mirror_path = full_path.replace(mirror_rule[0], mirror_rule[1])
        print(f"try to make its mirror dir: {mirror_path}")
        # Check if the parent directory of `mirror_path` exists
        if not osp.exists(osp.dirname(mirror_path)):
            # Prompt the user to decide whether to create the missing directories
            user_input = (
                input("The parent directory on the mirror path does not exist. Do you want to create it? (y/n): ")
                .strip()
                .lower()
            )
            if user_input == "y":
                # Create the non-existent parent directories
                os.makedirs(osp.dirname(mirror_path))
                print(f"Parent directory '{osp.dirname(mirror_path)}' has been created.")
            elif user_input == "n":
                print("Operation aborted. Please check the parent directory of the mirror path.")
                return
            else:
                print("Invalid input. Operation aborted.")
                return
        os.makedirs(mirror_path)
    if linking:
        print("creating cross linking between two dirs")
        create_link(link_path=osp.join(mirror_path, "primary.lnk"), target_path=full_path)
        create_link(
            link_path=osp.join(full_path, "mirror.lnk"),
            target_path=osp.abspath(mirror_path),
        )
