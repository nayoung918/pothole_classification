import os

org_output_dir_name = "240516"
log_dir = "log_240516"

base1 = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
            --path_type false --txt_type true \
            --train_txt_path results/240516/dataset_txt/20240516_case02_train.txt \
            --valid_txt_path results/240516/dataset_txt/20240516_case02_train.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0 \
            --use_cropimg True"""

name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_2_case02'
cls_name_txt1 = 'results/240516/dataset_txt/class_name.txt'
os.system(f"""{base1} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name1} \
            --nb_classes 2 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --cls_name_txt {cls_name_txt1}""")



# # ##########################################################################################################
# base2 = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --batch_size 128 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
#             --path_type false --txt_type true \
#             --train_txt_path results/240516/dataset_txt/padding_70_train.txt \
#             --valid_txt_path results/240516/dataset_txt/padding_70_valid.txt \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0 \
#             --use_cropimg True"""

# name2 = f'pad_PIXEL_padsize_70.0_box_False_shift_True_nbclss_16'
# cls_name_txt2 = 'results/240516/dataset_txt/class_name.txt'
# os.system(f"""{base2} \
#             --padding PIXEL \
#             --padding_size 70.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/{name2} \
#             --nb_classes 16 \
#             --log_dir {log_dir} \
#             --log_name {name2} \
#             --cls_name_txt {cls_name_txt2}""")


# ########################################################################################################
# base3 = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --batch_size 128 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
#             --path_type false --txt_type true \
#             --train_txt_path results/240516/dataset_txt/padding_80_train.txt \
#             --valid_txt_path results/240516/dataset_txt/padding_80_valid.txt \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0 \
#             --use_cropimg True"""

# name3 = f'pad_PIXEL_padsize_80.0_box_False_shift_True_nbclss_16'
# cls_name_txt3 = 'results/240516/dataset_txt/class_name.txt'
# os.system(f"""{base3} \
#             --padding PIXEL \
#             --padding_size 80.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/{name3} \
#             --nb_classes 16 \
#             --log_dir {log_dir} \
#             --log_name {name3} \
#             --cls_name_txt {cls_name_txt3}""")


##########################################################################################################
# base4 = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --batch_size 128 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
#             --path_type false --txt_type true \
#             --train_txt_path results/240516/dataset_txt/padding_90_train.txt \
#             --valid_txt_path results/240516/dataset_txt/padding_90_valid.txt \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0 \
#             --use_cropimg True"""

# name4 = f'pad_PIXEL_padsize_90.0_box_False_shift_True_nbclss_16'
# cls_name_txt4 = 'results/240516/dataset_txt/class_name.txt'
# os.system(f"""{base4} \
#             --padding PIXEL \
#             --padding_size 90.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/{name4} \
#             --nb_classes 16 \
#             --log_dir {log_dir} \
#             --log_name {name4} \
#             --cls_name_txt {cls_name_txt4}""")

# ##########################################################################################################
# base5 = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --batch_size 128 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
#             --path_type false --txt_type true \
#             --train_txt_path results/240516/dataset_txt/padding_100_train.txt \
#             --valid_txt_path results/240516/dataset_txt/padding_100_valid.txt \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0 \
#             --use_cropimg True"""

# name5 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_16'
# cls_name_txt5 = 'results/240516/dataset_txt/class_name.txt'
# os.system(f"""{base5} \
#             --padding PIXEL \
#             --padding_size 100.0 \
#             --use_bbox False \
#             --use_shift True \
#             --output_dir results/{org_output_dir_name}/{name5} \
#             --nb_classes 16 \
#             --log_dir {log_dir} \
#             --log_name {name5} \
#             --cls_name_txt {cls_name_txt5}""")
