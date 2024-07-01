#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
from PIL import Image
import numpy as np
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "1"

import warnings

warnings.filterwarnings("ignore")
import argparse
import copy
import time

import torch.nn.functional as F
from tqdm import tqdm

import nltk
nltk.download('punkt')

import build_vocab as vocab_utils
import caption_data_loader as dl
import globvars as gv
import losses
import net
import utils
import json

import solve_VLAR as VLAR

from utils import init_DDP
from torch.utils.data.distributed import DistributedSampler

def reset_state(args):
    #    global seed
    gv.seed = np.random.randint(10000) if args.seed == -1 else args.seed
    args.seed = gv.seed
    manualSeed = gv.seed  #
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (gv.seed))

def extract(args, dataloader):
    
    if args.mode == 'train':
        exist_file_path = './checkpoints/caption/Whole_Train_Valid_Test_Caption_95percent.json'
        # exist_file_path = './checkpoints/caption/Whole_Train_Caption_95percent.json'
    else:
        exist_file_path = './checkpoints/caption/Whole_Valid_Test_Caption.json'
        
    global existing_caption_dict
    with open (exist_file_path, 'r') as json_file:
            existing_caption_dict = json.load(json_file)
            print(f'Existing Caption Dict: {len(existing_caption_dict)}')
            
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

    def train_loop(epoch, train_loader, whole_segment, phase):
        unit = (len(train_loader)//whole_segment+1)
        start_idx, end_idx = unit*(phase-1), unit*(phase)
        print(f'Total Length of Dataloader: {len(train_loader)}')
        print(f'Mode: Train, Whole_Segment: {whole_segment}, Phase: {phase}')
        print(f'This Train Loop begin at {start_idx} to {end_idx}')
        for i, (im, q, _, a, av, pids, q_stn, im_path) in tqdm(enumerate(train_loader)):

            # if i >= 2 : break            
            
            if i < start_idx: continue
            if i >= end_idx: break
            
            start_time = time.time()   
            if args.caption_type == "Qwen":
                for cap_iter, cap_img_path in enumerate(im_path):
                    
                    img_name = cap_img_path.split('/')[-1]
                    if img_name in existing_caption_dict:
                        continue
                    
                    question = q_stn[cap_iter]
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

            end_time = time.time()
            dur_time = end_time - start_time
            print(f' Batch: {i}/{len(train_loader)}  Dur_time: {dur_time:.4f} for {im.shape[0]} images')
            # batch 24, whole len 4734 기준으로 1 batch = 4min, whole time = 4734*4 = 19000min = 316 hour = 13.2 days
            # If 6 GPU runs, days per gpu = 2.2 days, 800 loop

    def val_loop(val_loader, whole_segment, phase):
        unit = (len(val_loader)//whole_segment+1)
        start_idx, end_idx = unit*(phase-1), unit*(phase)
        print(f'Total Length of Dataloader: {len(val_loader)}')   
        print(f'Mode: Valid & Test, Whole_Segment: {whole_segment}, Phase: {phase}')
        print(f'This Valid & Test Loop begin at {start_idx} to {end_idx}')
        with torch.no_grad():
            for i, (im, q, o, a, av, pids, q_stn, info, im_path) in enumerate(val_loader):
            
                # if i >= 2 : break            
            
                if i < start_idx: continue
                if i >= end_idx: break          

                start_time = time.time()
                if args.caption_type == "Qwen":    
                    for cap_iter, cap_img_path in enumerate(im_path):
                        
                        img_name = cap_img_path.split('/')[-1]
                        if img_name in existing_caption_dict:
                            continue
                        
                        # 오로지 캡션만 했을 떄, ===================================== #
                        # question = q_stn[cap_iter]
                        # # caption_prompt = f'{question}. Based on the problem, please give me a caption of image. It should describe the image in detail. Use only english and do not mention the options'
                        # caption_prompt = f'Please create a caption that explains the image well. It should describe the image in detail. You should only use English.'
                        # caption_query = Qwen_tokenizer.from_list_format([
                        #         {'image': cap_img_path},
                        #         {'text': caption_prompt},
                        #     ])
                        # with torch.no_grad():
                        #     caption_response, caption_history = Qwen.chat(Qwen_tokenizer, query=caption_query, history=None)
                        #     caption_response = caption_response.encode('ascii', 'ignore').decode('ascii')
                        # ======================================================== #
                        
                        question = q_stn[cap_iter]
                        vqa_prompt = f'Looking at this image, propose 3 question-answer pairs. The questions and answers should be based on the visual, locational information. Use only english.' 
                        vqa_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': vqa_prompt},
                            ])
                        # caption_prompt = f'The Question corresponding to the image is "{question}".\nBased on the Question and VQA dataset, please give a final version of the image description. It should describe image in detail considering the information in the problem, but should not contain the question and options itself. Use only english'
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
                        # caption_log[im_name] = {'caption':caption_response}
                        
                end_time = time.time()
                dur_time = end_time - start_time
                print(f'Batch: {i}/{len(val_loader)}  Dur_time: {dur_time:.4f} for {im.shape[0]} images')
                                
    train_loader = dataloader["train"]
    val_loader = dataloader["valid"]
    test_loader = dataloader["test"]
    
    print("Starting training set extraction")
    json_name = f'Extra_{args.mode}_output_W{args.Whole_Segment}_P{args.Phase}.json'
    print(json_name)
    
    for epoch in range(1):
        print(f'Batch_size: {args.batch_size}')
        
        if args.mode=='train':
            train_loop(epoch, train_loader, args.Whole_Segment, args.Phase)
        elif args.mode=='valid':
            val_loop(val_loader, args.Whole_Segment, args.Phase)
        elif args.mode=='test':
            val_loop(test_loader, args.Whole_Segment, args.Phase) 
        
        with open(os.path.join(args.save_root, json_name),'w') as f:
                    json.dump(caption_log, f, ensure_ascii=False, indent=4)
        
        with open(os.path.join(args.save_root, json_name),'r') as f:
                    saved_caption = json.load(f)
                    print(f'Saved caption: {len(saved_caption)}')
        
    print('================= Complete =================')


def get_data_loader(args, split, batch_size=100, shuffle=False, num_workers=6, pin_memory=True):
    if split == "train":
        args.preprocess = None
        dataset = dl.SMART_TrainData(args, split)
        collate_fn = None
        
    else:
        dataset = dl.SMART_ValData(args, split)
        collate_fn = dl.SMART_collate_fn
    
    if args.use_DDP:
        sampler = DistributedSampler(dataset = dataset, shuffle=shuffle)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size//num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            sampler=sampler
            )
    else:    
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
    parser.add_argument(
        "--puzzles", default="all", type=str, help="comma separated / all / puzzle groups (counting,math etc.)"
    )
    parser.add_argument("--batch_size", default=64, type=int, help="batch size (16)")
    parser.add_argument("--num_epochs", default=100, type=int, help="epoch")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate (0.001)")
    parser.add_argument("--test_file", type=str, help="csv file for train")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/SMART101-release-v1/SMART101-Data/",
        help="location of the csv files, and location of the images, relative location is provided in the csv file.",
    )
    parser.add_argument("--train_diff", type=str, default="easy", help="easy/medium/hard")
    parser.add_argument("--test_diff", type=str, default="easy", help="easy/medium/hard")
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="80:5:15",
        help="how to split train and val, when both use the same instance list.",
    )
    parser.add_argument("--save_root", type=str, default="./ckpt/dump/", help="location to save intermediate files.")
    parser.add_argument("--vocab_path", type=str, default="none", help="location to save intermediate files.")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--pretrained", type=str, help="should use a pretrained model?")
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer to use")
    parser.add_argument("--loss_type", type=str, default="regression", help="classifier/regression/Both")
    parser.add_argument("--model_name", type=str, help="model to use resnet50/resnet18/...")
    parser.add_argument("--seed", type=int, default=-1, help="seed to use")
    parser.add_argument("--data_tot", type=int, default=2000, help="how many instances to use for train+val+test")
    parser.add_argument("--use_clip_text", action="store_true", help="should use clip text embeddings?")
    parser.add_argument("--no_meta", action="store_true", help="do not use meta learning for optimization?")
    parser.add_argument("--log", action="store_true", help="should print detailed log of accuracy?")
    parser.add_argument("--baselines", action="store_true", help="run the baselines from answer distributions?")
    parser.add_argument(
        "--monolithic", action="store_true", help="use a single head for all puzzles (except the sequential ones)?"
    )
    parser.add_argument(
        "--split_type", type=str, default="standard", help="type of data split: stanard/exclude/puzzle/fewshot"
    )
    parser.add_argument("--word_embed", type=str, default="standard", help="standard/gpt/glove")
    parser.add_argument(
        "--use_single_image_head", action="store_true", help="use a single image head for all the puzzles?"
    )
    parser.add_argument(
        "--fsK", type=int, default=100, help="how many samples should we use to train in a fewshot setting?"
    )
    parser.add_argument("--log_freq", type=int, default=50, help="log frequency?")
    parser.add_argument("--mode", type=str, default=None, help="Extraction Mode")
    parser.add_argument("--train_backbone", action="store_true", help="train the image backbone?")
    parser.add_argument("--no_question", action="store_true", help="do not use questions?")
    parser.add_argument("--no_image", action="store_true", help="do not use images?")
    parser.add_argument("--num_meta_updates", type=int, default=1, help="number of meta updates?")
    parser.add_argument(
        "--feat_size", type=int, default=128, help="intermediate feature size for image and language features?"
    )
    parser.add_argument("--challenge", action="store_true", help="evaluate a model on the challenge val dataset")
    parser.add_argument("--phase", type=str, default='val', help="phase of the challenge")
    parser.add_argument("--pretrained_model_path", type=str, default="./", help="path to pretrained VLAR reasoning model")
    
    # 내가 추가한 Argument List =================================================================== #
    parser.add_argument("--use_puzzle_type_classifier", action="store_true", help="Add Puzzle Type Classifier?")
    parser.add_argument("--use_option_prompting", action="store_true", help="Use Answer Option Prompting?")
    parser.add_argument("--use_DDP", action="store_true", help="Use Distributed Data Parallel?")
    parser.add_argument("--use_LORA", action="store_true", help="Use LoRA Fine-tuning")
    parser.add_argument("--test_return_output", action="store_true", help="When Test, Return Output of Model")
    parser.add_argument("--tensorboard_freq", type=int, default=50, help="Report to Tensorboard frequency?")
    parser.add_argument("--LLM_type", type=str, default='xl', help="Which of LLM use")
    parser.add_argument("--caption_type", type=str, default='None', help="Which of VLM use for VQA & Captioning")
    parser.add_argument("--gpu_num", type=int, default=0, help="Define GPU used")
    parser.add_argument("--Whole_Segment", type=int, default=0, help="Whole Segment of GPU")
    parser.add_argument("--Phase", type=int, default=0, help="Phase of GPU")
    

    args = parser.parse_args()

    if args.split_type == "puzzle":  # use only a single head and single output head for PS.
        args.monolithic = True
        args.use_single_image_head = True
        args.no_meta = True  # we do not use meta learning for puzzle split.

    if args.monolithic:  # in this case, we use a single output head, but do not include sequential puzzles.
        args.no_meta = True
    
    # 퍼즐 타입 Classifier 추가 했을 경우 ============================================================ #
    if args.use_puzzle_type_classifier:
        args.monolithic = False
        args.use_single_image_head = False
        args.no_meta = False

    gv.globals_init(args)

    args.puzzle_ids_str, args.puzzle_ids = utils.get_puzzle_ids(args)   # #puzzle_ids=95, in PS
    args.location = os.path.join(args.save_root, "checkpoints")
    
    reset_state(args)
        
    gv.NUM_CLASSES_PER_PUZZLE = utils.get_puzzle_class_info(    # 95 dict
        args
    )  # initialize the global with the number of outputs for each puzzle.
        
    vocab = vocab_utils.process_text_for_puzzle(args)
    if args.vocab_path == "none":
        args.vocab_path = os.path.join(args.save_root, "vocab_puzzle_" + args.puzzle_ids_str + ".pkl")
        
    train_loader = get_data_loader(
            args, "train", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.split_type == 'standard':
        val_loader = None
        test_loader = None
    else:
        val_loader = get_data_loader(args, "val", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = get_data_loader(args, "test", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
    dataloader = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }
    
    print(args)
    print("num_puzzles=%d" % (len(args.puzzle_ids)))
        
    global device
    device = f'cuda:{args.gpu_num}'
    extract(args, dataloader)