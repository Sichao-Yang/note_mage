indir=examples/redundant_img_remove
outdir=examples/redundant_img_remove_new
cd ..
python nmag/md_manager/redundant_img_remover.py --input_folder $indir --ignore_formats pdf txt --output_folder $outdir