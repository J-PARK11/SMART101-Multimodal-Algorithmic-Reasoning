python extract_caption.py \
    --model_name IBLIP \
    --num_workers 8 \
    --batch_size 24 \
    --word_embed none \
    --data_root /data/SMART101-release-v1/SMART101-Data/ \
    --save_root ./checkpoints/caption/ \
    --puzzles all \
    --split_type puzzle \
    --use_option_prompting \
    --caption_type Qwen \
    --extract_partition 1200 \
    --extract_phase 2 \
    --gpu_num 1