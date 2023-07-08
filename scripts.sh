To train the model use:
python main.py --model_name resnet50 --num_workers 8 --num_epochs 10 --puzzles all --loss_type classifier --batch_size 64 --log_freq 10  --data_tot 2000 --data_root /homes/cherian/train_data/NAR/vilps/vilps_cpl/VLPS_v2/SMART101-release-v1/SMART101-Data/ --word_embed bert

For test on the challenge val_set, use:
python main.py --model_name resnet50 --num_workers 0 --loss_type classifier --word_embed bert --split_type puzzle --challenge --phase val --pretrained_model_path ./checkpoints/ckpt_resnet50_bert_212.pth


For test on the challenge test_set, use:
python main.py --model_name resnet50 --num_workers 0 --loss_type classifier --word_embed bert --split_type puzzle --challenge --phase test --pretrained_model_path ./checkpoints/ckpt_resnet50_bert_212.pth





