checkpoint=$1
python main_aitod.py \
  --output_dir data/work_dir/aitodv2/DQ_eval \
	-c config/DQ_5scale.py --coco_path /mnt/e/data/aitod \
	--eval --resume $checkpoint \
	--options dn_scalar=100 embed_init_tgt=False \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0


# checkpoint=$1
# python main_aitod.py \
#   --output_dir logs/DQ_eval \
# 	-c config/DQ_5scale.py --coco_path /mnt/data0/Garmin/datasets \
# 	--eval --resume $checkpoint \
# 	--options dn_scalar=100 embed_init_tgt=False \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0