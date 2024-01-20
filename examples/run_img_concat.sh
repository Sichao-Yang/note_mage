#!/bin/sh
cd ..
python nmag/img_manager/img_concat.py --input_folder examples/img_concat --out_path examples/img_concat.png
python nmag/img_manager/img_concat.py --input_folder examples/img_concat --out_path examples/img_concat.pdf
