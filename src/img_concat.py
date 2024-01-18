from PIL import Image
import os
import argparse
from pathlib import Path
from os import path as osp


def resize_all(images):
    widths, heights = zip(*(im.size for im in images))
    mw, mh = max(widths), max(heights)
    return [im.resize((mw, mh)) for im in images]


def concate_imgs(images, file_path="./out.png", direction="h"):
    def _horizon():
        total_width = sum(widths)
        max_height = max(heights)
        new_im = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]
        return new_im

    def _vertical():
        width = max(widths)
        height = sum(heights)
        new_im = Image.new("RGB", (width, height))
        y_offset = 0
        for im in images:
            new_im.paste(im, (0, y_offset))
            y_offset += im.size[1]
        return new_im

    widths, heights = zip(*(im.size for im in images))
    if direction == "h":
        new_im = _horizon()
    elif direction == "v":
        new_im = _vertical()
    else:
        raise ValueError(f"direction {direction} is not within ['h', 'v']")
    new_im.save(file_path)


def concate_to_pdf(images, file_path="out.pdf"):
    # images = [im.convert('RGB') for im in images]
    # images[0].save(filepath, save_all=True, append_images=images[1:])
    first_img = True
    imgs = []
    for img in images:
        img = img.convert("RGB")
        if not first_img:
            imgs.append(img)
        else:
            begin = img
            first_img = False
    begin.save(file_path, save_all=True, append_images=imgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_folder",
        type=str,
        default="examples/img_concat",
        help="the folder that contains images",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="examples/img_concat.png",
        help="the output path for concatenated images",
    )
    parser.add_argument(
        "--concat_direction",
        type=str,
        default="v",
        help="the direction for concatenation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="increase output verbosity"
    )
    args = parser.parse_args()

    supported_img_format = [".jpg", ".png"]

    filelist = [
        osp.join(args.input_folder, x)
        for x in sorted(os.listdir(args.input_folder))
        if Path(x).suffix in supported_img_format
    ]

    images = [Image.open(x) for x in filelist]

    assert args.concat_direction in ["h", "v"], "unsupported concat direction!"
    if Path(args.out_path).suffix in supported_img_format:
        concate_imgs(
            resize_all(images), direction=args.concat_direction, file_path=args.out_path
        )
    elif Path(args.out_path).suffix == ".pdf":
        concate_to_pdf(resize_all(images), file_path=args.out_path)
    else:
        raise ValueError(f"unsupported output format! {args.out_path}")
    print(f"Done.\nSaved concated image to {args.out_path}")
