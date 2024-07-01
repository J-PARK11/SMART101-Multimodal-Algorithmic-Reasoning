"""
KT Qwen Caption 추출 파일 가이드라인
추가 설치해야 하는 패키지: pip install tiktoken, pip install transformers_stream_generator
명령어: python extract_caption_for_kt.py --num_workers 8 --batch_size 5 --data_root /dataset/test-images/ --save_root ./checkpoints/dump --gpu_num 0 --Whole_Segment 1 --Phase 1
Arguement 설명
    --data_root는 추출하고자 하는 이미지가 들어있는 폴더이름.
    --save_root는 caption을 json파일로 묶어서 반환할 위치.
    --gpu_num은 사용할 GPU 넘버.
    --Whole_Segment는 만약 이미지가 너무 많고, 캡션뽑는게 너무 오래걸리면 GPU숫자만큼 Whole Segment를 설정해서 나눠서 추론함.
    --Phase는 본 커널에서 수행할 Phase.
'
# 만약 GPU 4개가 비어서 데이터를 4등분하여 캡션을 나누어 뽑고싶다면, Whole Segment=4, Phase는 각각 [1,2,3,4]로 설정하여
# 서로 다른 4 개의 터미널에서 Phase와 GPU를 바꿔가며, 돌려야함.
# GPU 하나로 다할거면 --Whole_Segment 1 --Phase 1로 설정.
Ex.)
1번 터미널: --gpu_num 0 --Whole_Segment 4 --Phase 1
2번 터미널: --gpu_num 1 --Whole_Segment 4 --Phase 2
3번 터미널: --gpu_num 2 --Whole_Segment 4 --Phase 3
4번 터미널: --gpu_num 3 --Whole_Segment 4 --Phase 4
"""    
# python extract_caption_for_kt.py --num_workers 8 --batch_size 4 --data_root ./dataset/test-images/ --save_root ./checkpoints/dump --gpu_num 0 --Whole_Segment 1 --Phase 1


import os
import json
import copy
import time
import torch
import warnings
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

warnings.filterwarnings("ignore")

import torch.nn.functional as F
from torch.utils.data import Dataset

def extract(args, dataloader):
    
    global caption_log
    caption_log = dict()
        
    if args.caption_type == "Qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        global Qwen_tokenizer
        global Qwen
        global max_cap_len
        max_cap_len = 600
        Qwen_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='Qwen/Qwen-VL-Chat', trust_remote_code=True)
        Qwen = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='Qwen/Qwen-VL-Chat', device_map=device, trust_remote_code=True).eval()
        Qwen.generation_config = GenerationConfig.from_pretrained('Qwen/Qwen-VL-Chat', trust_remote_code=True)
        
        Qwen_tokenizer.model_max_length=1000 #600 # 400
        Qwen.generation_config.max_new_tokens = 400 # 200 # 100
        Qwen.generation_config.do_sample = True # False
        Qwen.generation_config.max_window_size = 2000 # 1500

        def decode_image(im_list):
            im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
            im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))] 
            return im_list

    def val_loop(val_loader, whole_segment, phase):
        unit = (len(val_loader)//whole_segment + 1)
        start_idx, end_idx = unit*(phase-1), unit*(phase)
        print(f'Total Length of Dataloader: {len(val_loader)}')   
        print(f'Mode: Valid & Test, Whole_Segment: {whole_segment}, Phase: {phase}')
        print(f'This Valid & Test Loop begin at {start_idx} to {end_idx}')
        with torch.no_grad():
            for i, (im_path) in enumerate(val_loader):
            
                if i < start_idx: continue
                if i >= end_idx: break          

                # start_time = time.time()
                if args.caption_type == "Qwen":    
                    for cap_iter, cap_img_path in enumerate(im_path):
                        
                        # question = q_stn[cap_iter]
                        
                        vqa_prompt = f'Looking at this image, propose 3 question-answer pairs. The questions and answers should be based on the visual, locational information. Use only english.' 
                        vqa_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': vqa_prompt},
                            ])
                        caption_prompt = f'Please create a description that includes both a detailed explanation of the image and a visual element. You should only use English.'
                        caption_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': caption_prompt},
                            ])
                        with torch.no_grad():
                            vqa_response, vqa_history = Qwen.chat(Qwen_tokenizer, query=vqa_query, history=None)
                            vqa_response = vqa_response.encode('ascii', 'ignore').decode('ascii')
                            caption_response, caption_history = Qwen.chat(Qwen_tokenizer, query=caption_query, history=vqa_history)
                            caption_response = caption_response.encode('ascii', 'ignore').decode('ascii')
                        im_name = im_path[cap_iter].split('/')[-1]
                        caption_log[im_name] = {'vqa':vqa_response, 'caption':caption_response}
                        
                # end_time = time.time()
                # dur_time = end_time - start_time
                # print(f'Batch: {i}/{len(val_loader)}  Dur_time: {dur_time:.4f} for {len(im_path)} images')
                                
    val_loader = dataloader["valid"]
    
    print("Starting training set extraction")
    json_name = f'caption_output_W{args.Whole_Segment}_P{args.Phase}.json'
    print(json_name)
    
    for epoch in range(1):
        print(f'Batch_size: {args.batch_size}')
        val_loop(val_loader, args.Whole_Segment, args.Phase)
        
        with open(os.path.join(args.save_root, json_name),'w') as f:
                    json.dump(caption_log, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(args.save_root, json_name),'r') as f:
                    saved_caption = json.load(f)
                    print(f'Saved caption: {len(saved_caption)}')
        
    print('================= Complete =================')

class KT_SMART_Challenge_Data(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root        
        self.img_list = os.listdir(self.data_root)

    def __getitem__(self, idx):
        im_name = self.img_list[idx]
        im_path = os.path.join(self.data_root, im_name)
        
        return im_path

    def __len__(self):
        return len(self.img_list)

def get_data_loader(args, split, batch_size=100, shuffle=False, num_workers=6, pin_memory=True):
        
    dataset = KT_SMART_Challenge_Data(args.data_root)
    collate_fn = None
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        )
        
    return data_loader

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SMART dataset")
    parser.add_argument("--batch_size", default=64, type=int, help="batch size (16)")
    parser.add_argument("--data_root", type=str, default="./dataset/test-images/")
    parser.add_argument("--save_root", type=str, default="./checkpoints/dump/", help="location to save intermediate files.")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--caption_type", type=str, default='Qwen', help="Which of VLM use for VQA & Captioning")
    
    parser.add_argument("--gpu_num", type=int, default=0, help="Define GPU used")
    parser.add_argument("--Whole_Segment", type=int, default=0, help="Whole Segment of GPU")
    parser.add_argument("--Phase", type=int, default=0, help="Phase of GPU")
    

    args = parser.parse_args()

    val_loader = get_data_loader(args, "val", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
    dataloader = {"valid": val_loader,}
    
    print(args)
        
    global device
    device = f'cuda:{args.gpu_num}'
    extract(args, dataloader)
