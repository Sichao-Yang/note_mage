#!/bin/bash
indir=examples/redundant_img_remove
outdir=examples/redundant_img_remove_new
cd ..
python nmag/mmage/red_img_remover.py --src_dir $indir --ignore_items pdf txt --dst_dir $outdir
