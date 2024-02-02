import os
from os import path as osp


def makedir(path, mirror_rule=("0_MOOC_Videos", "Drive/sync/0_notes_all/0_MOOC_notes")):
    full_path = osp.abspath(path)
    print(f"make video dir: {full_path}")
    os.makedirs(full_path)
    if mirror_rule[0] in full_path:
        print(f"based on rule {mirror_rule}, this path is mirrorable")
        mirror_path = full_path.replace(mirror_rule[0], mirror_rule[1])
        print(f"make its mirror dir: {mirror_path}")
        assert osp.exists(osp.dirname(mirror_path))
        os.makedirs(mirror_path)
