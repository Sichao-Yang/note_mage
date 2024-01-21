#!/usr/bin/sh
cd ..
python nmag/image/img_concat.py --src_dir examples/img_concat --dst_path examples/img_concat.png
python nmag/image/img_concat.py --src_dir examples/img_concat --dst_path examples/img_concat.pdf
