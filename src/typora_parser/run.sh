datafolder=data

cd redundant_img_remover
python redundant_img_remover.py --input_folder $datafolder --ignore_formats pdf txt

cd imgpath_corrector
python imgpath_corrector.py --input_folder $datafolder 