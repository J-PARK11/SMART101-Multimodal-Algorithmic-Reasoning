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
import data_loader as dl
import globvars as gv
import losses
import net
import utils
import json
import pickle

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
    global pkl_log
    pkl_log = dict()
    output_save_root = os.path.join(args.save_root)
            
    def decode_image(im_list):
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))] 
        return im_list

    def train_loop(epoch, train_loader):
        print(f'Total Length of Dataloader: {len(train_loader)}')
        for i, (im, q, o, a, av, pids, q_stn, info, im_path) in enumerate(train_loader):
            # if i >= 2: break
            for j in range(len(info)):
                puzzle_name = info[j]['image']
                pkl_log[puzzle_name] = info[j]
                print(f'batch: {i}/{len(train_loader)}, idx: {j}, puzzle_name:{puzzle_name}')
          
    def val_loop(val_loader):
        print(f'Total Length of Dataloader: {len(val_loader)}')   
        with torch.no_grad():
            for i, (im, q, o, a, av, pids, q_stn, info, im_path) in enumerate(val_loader):
                for j in range(len(info)):
                    puzzle_name = info[j]['image']
                    pkl_log[puzzle_name] = info[j]       
                    print(f'batch: {i}/{len(val_loader)}, idx: {j}, puzzle_name:{puzzle_name}') 
                
    train_loader = dataloader["train"]
    # val_loader = dataloader["valid"]
    # test_loader = dataloader["test"]
    
    for epoch in range(1):
        print(f'Batch_size: {args.batch_size}')
        train_loop(epoch, train_loader)
        # val_loop(val_loader)
        # val_loop(test_loader) 
        
        pkl_name = 'whole_train.pickle'
        with open(os.path.join(output_save_root, pkl_name),'wb') as fw:
            pickle.dump(pkl_log, fw)
        
        with open(os.path.join(output_save_root, pkl_name),'rb') as fr:
            saved_pickle = pickle.load(fr)
            print(f'Saved Pickle: {len(saved_pickle)}')
        
    print('================= Complete =================')

def get_data_loader(args, split, batch_size=100, shuffle=False, num_workers=6, pin_memory=True):
    if split == "train":
        args.preprocess = None
        dataset = dl.SMART_ValData(args, split)
        collate_fn = dl.SMART_collate_fn
        
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
    parser.add_argument("--test", action="store_true", help="evaluate a model?")
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
    parser.add_argument("--extract_partition", type=int, default=0, help="Divide Extraction partition")
    parser.add_argument("--extract_phase", type=int, default=0, help="Define Extraction Phase")
    parser.add_argument("--gpu_num", type=int, default=0, help="Define GPU used")
    parser.add_argument("--pkl_extraction_mode", type=bool, default=True, help="extraction pkl data mode")
    
    

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
        
    if args.test:
        assert args.seed > -1  # when evaluating we need to use the seed to take the checkpoint.
        
    if args.use_DDP:
        init_DDP()

    gv.globals_init(args)

    if not args.challenge:
        args.puzzle_ids_str, args.puzzle_ids = utils.get_puzzle_ids(args)   # #puzzle_ids=95, in PS
        args.location = os.path.join(args.save_root, "checkpoints")
        args.log_path = os.path.join(args.save_root, "log")
    
        reset_state(args)
        
        gv.NUM_CLASSES_PER_PUZZLE = utils.get_puzzle_class_info(    # 95 dict
            args
        )  # initialize the global with the number of outputs for each puzzle.
        
        vocab = vocab_utils.process_text_for_puzzle(args)
        if args.vocab_path == "none":
            args.vocab_path = os.path.join(args.save_root, "vocab_puzzle_" + args.puzzle_ids_str + ".pkl")
        
        train_loader = get_data_loader(
                args, "train", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_loader = get_data_loader(args, "val", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = get_data_loader(args, "test", batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            
        dataloader = {
            "train": train_loader,
            "valid": val_loader,
            "test": test_loader,
        }
    
        utils.backup_code_and_start_logger(args, args.log_path, args.seed)
    
        print(args)
        print("num_puzzles=%d" % (len(args.puzzle_ids)))
        
        global device
        device = f'cuda:{args.gpu_num}'
        extract(args, dataloader)
        
    elif args.challenge: # if we are using the VLAR challenge evaluation
        VLAR.predict_on_challenge_data(args, args.pretrained_model_path, challenge_phase=args.phase)
        
    
