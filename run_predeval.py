import os
import sys
import glob
from pathlib import Path

base = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_class 0 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/pothole_data/testset/240404_seoul/tmp1 \
            --pred_save True \
            --pred_save_with_conf True \
            --use_cropimg True \
            --conf 0.0 \
            --several_models True \
            --sample_images False \
            --use_type False \
            --use_type_json results/240131/dataset_txt/class_type_case03.json """

# 'results/231005/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_2_GAN+detection_except_augmentation+seoul_pos+manhole_02_neg/checkpoint-best.pth'

# 'results/240516/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_2_case01/checkpoint-best.pth'
models = [  
    'results/240516/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_2_case02/checkpoint-best.pth'
]

class_names = [
    'results/240516/dataset_txt/class_name.txt',
]
# ../res/testset/240131/testcase_03
# /240115-240121_korea_site/set01
# /240117-240125_GC/set01
# output_dir_path = "../res/pothole_data/ray_serve_test_3"
output_dir_path = "../res/pothole_data/ray_serve_test_tmp1"
# graph_save_dir = "../res/testset/240219/testcase_01/"
graph_save_dir = output_dir_path 

for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()


ckpt = ' '.join(models)
cln = ' '.join(class_names)

os.system(f"""{base} \
            --resume {ckpt} \
            --pred_eval_name {graph_save_dir} \
            --pred_save_path {output_dir_path} \
            --cls_name_txt {cln}""")
