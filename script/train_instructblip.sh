python main.py \
    --model_name IBLIP \
    --num_workers 8 \
    --num_epochs 5 \
    --loss_type Both \
    --batch_size 8 \
    --log_freq 4 \
    --train_backbone \
    --word_embed none \
    --data_root /data/SMART101-release-v1/SMART101-Data/ \
    --save_root ./checkpoints/IBLIP-Flan-T5/full_trainset/ \
    --puzzles all \
    --split_type puzzle \
    --use_option_prompting \
    --use_LORA \
    --tensorboard_freq 200 \
    --lr 0.0003 \
    --split_ratio 100:5:15 \ 
    --LLM_type t5_xl \