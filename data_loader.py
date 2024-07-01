#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
import pdb
import pickle
import json

import nltk
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import baselines
import globvars as gv
import utils


class SMART_Data(Dataset):
    def __init__(self, args):
        vocab_path = args.vocab_path
        self.max_qlen = 110
        self.max_olen = 4  # max option length
        self.use_word_embed = False
        self.word_embed = None
        self.im_side = 224
        self.preprocess = args.preprocess
        self.no_question = args.no_question
        self.no_image = args.no_image

        if vocab_path != 'none':
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
                print("vocabulary size = %d" % (len(self.vocab)))

        if args.preprocess is None:  # VL models, will do preprocess later.
            self.transform = Compose(
                [
                    Resize(224),  # if the images are of higher resolution. we work with pre-resized 224x224 images.
                    # RandomCrop(224),
                    ToTensor(),
                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
                ]
            )
        elif args.model_name in ["flava", "mae"]:  # this will do feature extractin later.
            self.transform = Compose(
                [
                    Resize(300),
                    ToTensor(),
                ]
            )
        else:
            self.transform = args.preprocess

    def apply_transform(self, im_path):
        if self.no_image:  # create a dummy image.
            im = Image.fromarray((np.random.rand(self.im_side, self.im_side, 3) * 255).astype("uint8"))
        else:
            im = Image.open(im_path).convert("RGB")
        return self.transform(im)

    def quest_encode(self, question):
        tokens = nltk.tokenize.word_tokenize(question.lower())
        q_enc = np.zeros((self.max_qlen,), dtype="long")
        if not self.no_question:
            enc_tokens = (
                [self.vocab("<start>")] + [self.vocab(tokens[t]) for t in range(len(tokens))] + [self.vocab("<end>")]
            )
            q_enc[: min(self.max_qlen, len(enc_tokens))] = np.array(enc_tokens)
        return q_enc

    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def opts_encode(self, opts, key):
        opts = opts.lower()
        tokens = nltk.tokenize.word_tokenize(opts)
        enc_tokens = [self.vocab(tokens[t]) for t in range(len(tokens))]
        opt_enc = np.zeros((self.max_olen,), dtype="long")
        opt_enc[: min(self.max_olen, len(enc_tokens))] = np.array(enc_tokens)
        return opt_enc

    def split_fewshot_puzzles(self, puzzle_ids, split_ratio, split_name, split_type):
        if split_name == "train":
            split_pids = self.split_puzzles(puzzle_ids, split_ratio, "train", split_type)
            other_pids = self.split_puzzles(puzzle_ids, split_ratio, "test", split_type)
            other_pids = other_pids + self.split_puzzles(puzzle_ids, split_ratio, "val", split_type)
            return split_pids, other_pids
        else:
            split_pids = self.split_puzzles(puzzle_ids, split_ratio, split_name, split_type)
            other_pids = None
        return split_pids, other_pids

    def split_puzzles(self, puzzle_ids, split_ratio, split_name, split_type):
        if split_type == "puzzle" or split_type == "fewshot":
            if split_name == "train":
                val_test = gv.PS_VAL_IDX + gv.PS_TEST_IDX
                val_test = set([str(ii) for ii in val_test])
                puzzle_ids = list(set(puzzle_ids).difference(val_test))
                print("number of train puzzles = %d" % (len(puzzle_ids)))
            elif split_name == "val":
                puzzle_ids = [str(ii) for ii in gv.PS_VAL_IDX]
                print("number of val puzzles = %d" % (len(puzzle_ids)))
            else:
                puzzle_ids = [str(ii) for ii in gv.PS_TEST_IDX]
                print("number of test puzzles = %d" % (len(puzzle_ids)))
        else:
            splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
            n = len(puzzle_ids)
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                puzzle_ids = puzzle_ids[st:en]
            elif split_name == "val":
                st = int(np.ceil(n * splits[0] / 100.0))
                en = int(np.floor(n * splits[1] / 100.0))
                puzzle_ids = puzzle_ids[st:en]
            else:
                st = int(np.ceil(n * splits[1] / 100.0))
                puzzle_ids = puzzle_ids[st:]
        print("puzzles for %s =" % (split_name))
        print(puzzle_ids)
        return puzzle_ids

    def split_data(self, info, split_ratio, split_name, split_type="standard"):
        """
        split_type=standard is to use the split_ratio in the instance order
        split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
        split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
        """
        if split_type == "standard" or split_type == "fewshot":
            splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
            n = len(info)
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                info = info[st:en]
            elif split_name == "val":
                st = int(np.ceil(n * splits[0] / 100.0))
                en = int(np.floor(n * splits[1] / 100.0))
                info = info[st:en]
            else:
                st = int(np.ceil(n * splits[1] / 100.0))
                info = info[st:]
        elif split_type == "puzzle":
            splits = np.array([int(spl) for spl in split_ratio.split(":")])
            n = len(info)   # 2000
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                info = info[st:en]
            elif split_name == "val":
                st = 0
                en = int(np.floor(n * splits[1] / 100.0))
                info = info[st:en]
            else:
                st = 85
                en = (int(np.floor(n * splits[2] / 100.0)) + st) # 100
                info = info[st:en]
        elif split_type == "exclude":
            pid = info[0]["puzzle_id"]
            if int(pid) in gv.SEQ_PUZZLES or int(pid) == 58:
                # we don't do exclude splits for seq_puzzles are as they are most likely always unique
                info = self.split_data(info, split_ratio, split_name, split_type="standard")
            else:
                ans_distr = []
                for t in range(len(info)):
                    ans_distr.append(info[t]["AnswerValue"])
                ans_distr = np.array(ans_distr)
                bclassids = np.arange(gv.NUM_CLASSES_PER_PUZZLE[pid])
                x = np.histogram(ans_distr, bclassids)[0]
                x = x / x.sum()

                # select reasonable answers.
                valid_ans_idx = np.where(x > 0.01)
                x_cls = bclassids[valid_ans_idx]
                x = x[valid_ans_idx]
                median_class = x_cls[x <= np.median(x)][-1]
                try:
                    train_inst = np.array(info)[ans_distr != median_class]
                    test_inst = np.array(info)[ans_distr == median_class]
                except:
                    print(pid)
                    pdb.set_trace()

                n = len(train_inst)
                splits = np.array([int(spl) for spl in split_ratio.split(":")])
                splits[0] = splits[0] + splits[2]
                splits = splits.cumsum()[:2]

                if split_name == "train":
                    st = 0
                    en = int(np.floor(n * splits[0] / 100.0))
                    info = train_inst[st:en].tolist()
                elif split_name == "val":
                    st = int(np.ceil(n * splits[0] / 100.0))
                    en = int(np.floor(n * splits[1] / 100.0))
                    info = train_inst[st:en].tolist()
                else:
                    info = test_inst.tolist()
        else:
            raise "Unknown puzzle split type!!"

        return info


class SMART_TrainData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot  # how many instances per puzzles should we use?
        self.diff = args.train_diff
        self.word_embed = args.word_embed
        self.fewshot_K = args.fsK
        self.qa_info = []
        self.args = args
        train_pids = None

        puzzle_ids = (
            self.split_puzzles(args.puzzle_ids, args.split_ratio, split, args.split_type)
            if args.split_type == "puzzle"
            else args.puzzle_ids
        )
        if args.split_type == "fewshot":
            train_pids, fewshot_other_pids = self.split_fewshot_puzzles(
                args.puzzle_ids, args.split_ratio, split, args.split_type
            )
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            if args.split_type == "fewshot" and puzzle_id in fewshot_other_pids:
                qa_info = qa_info[: self.fewshot_K]
            else:
                qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + self.split_data(qa_info, args.split_ratio, split, args.split_type)
            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        if args.baselines:
            self.baseline_perf = baselines.get_baseline_performance(args, self.qa_info, split, self.num_tot, log=True)
        print("num_train=%d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
    
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im = self.apply_transform(im_path)
        qa = self.quest_encode(info["Question"])
        opts = 0
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            gv.MAX_DECODE_STEPS,
        )
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()
        
        if info:
            pass
        else:
            info = []
        
        if ("IBLIP" in self.args.model_name) or (self.args.model_name == 'Qwen'):
            if self.args.use_option_prompting:
                opts = [utils.get_val(info, key, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
                Answer_Option_phrase = ' Options:'
                for op, an in zip(["A", "B", "C", "D", "E"], opts):
                    # Answer_Option_phrase += f' {op}: {an},'
                    Answer_Option_phrase += f' {an},'
                Answer_Option_phrase = Answer_Option_phrase[:-1]
                
                # q_stn = 'Question: ' + info["Question"]
                q_stn = 'Question: ' + info["Question"] + " You must Return in following 5 Option's Numeral Value."
                q_stn = q_stn + Answer_Option_phrase
            else:
                q_stn = info['Question']
            if self.args.pkl_extraction_mode:
                return info
            return im, torch.tensor(qa), torch.tensor(opts), torch.tensor(lbl), torch.tensor(answer), torch.tensor(int(pid)), q_stn, im_path
                
        return im, torch.tensor(qa), torch.tensor(opts), torch.tensor(lbl), torch.tensor(answer), torch.tensor(int(pid)), [], []

    def __len__(self):
        return len(self.qa_info)


class SMART_ValData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot
        self.word_embed = args.word_embed
        self.fewshot_K = args.fsK
        self.args = args
        self.qa_info = []

        self.diff = args.test_diff if split == "test" else args.train_diff
        puzzle_ids = (
            self.split_puzzles(args.puzzle_ids, args.split_ratio, split, args.split_type)
            if args.split_type == "puzzle"
            else args.puzzle_ids
        )
        if args.split_type == "fewshot":
            puzzle_ids, fewshot_other_pids = self.split_fewshot_puzzles(
                args.puzzle_ids, args.split_ratio, split, args.split_type
            )

        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            if args.split_type == "fewshot":
                qa_info = qa_info[
                    self.fewshot_K : self.num_tot
                ]  # we use the fewshot_K for training. so use the rest for evaluation.
            else:
                qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + self.split_data(qa_info, args.split_ratio, split, args.split_type)
            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        print("num_val = %d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        if args.baselines:
            self.baseline_perf = baselines.get_baseline_performance(args, self.qa_info, split, self.num_tot, log=True)
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im = self.apply_transform(im_path)
        qa = self.quest_encode(info["Question"])

        _ = [utils.str_replace_(info, key) for key in ["A", "B", "C", "D", "E"]]
        opts = [utils.get_val(info, key, is_one_of_option=True) for key in ["A", "B", "C", "D", "E"]]
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            gv.MAX_DECODE_STEPS,
        )
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value
            
        if info:
            pass
        else:
            info = []
            
        if ("IBLIP" in self.args.model_name) or (self.args.model_name == 'Qwen'):
            if self.args.use_option_prompting:
                Answer_Option_phrase = ' Options:'
                for op, an in zip(["A", "B", "C", "D", "E"], opts):
                    # Answer_Option_phrase += f' {op}: {an},'
                    Answer_Option_phrase += f' {an},'
                Answer_Option_phrase = Answer_Option_phrase[:-1]
                
                q_stn = 'Question: ' + info["Question"] + " You must Return in following 5 Option's Numeral Value."
                q_stn = q_stn + Answer_Option_phrase
            else:
                q_stn = info['Question']
            return im, torch.tensor(qa), torch.tensor(opts), torch.tensor(lbl), torch.tensor(answer), torch.tensor(int(pid)), q_stn, info, im_path
            
        return im, torch.tensor(qa), opts, torch.tensor(lbl), torch.tensor(answer), torch.tensor(int(info["puzzle_id"])), [], info, []

    def __len__(self):
        return len(self.qa_info)


class SMART_Challenge_Data(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        vocab_path = args.vocab_path
        self.data_root = gv.VLAR_CHALLENGE_data_root
        self.word_embed = args.word_embed
        self.puzzles =  self.get_puzzle_questions(args.puzzles_file)
        
        self.max_qlen = 110
        gv.MAX_VAL = 256 # this is the maximum value of any answer in the SMART 101 training set. 
                         #This need not be the max value in the challenge test set.
                         
        if vocab_path != 'none':
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
                print("vocabulary size = %d" % (len(self.vocab)))
        
    def get_puzzle_questions(self, json_file):
        """ 
        Reads the provided json_file into a dictionary. 
        We expect the json file to be of the format: {'vlar': [{}. {}, ...]} 
        """
        
        with open(os.path.join(self.data_root, json_file)) as test_file:
            puzzle_data = test_file.read()
        puzzle_data = json.loads(puzzle_data)
        assert('VLAR' in puzzle_data.keys())
        return puzzle_data['VLAR'][1:]
    
    def quest_encode(self, question):
        tokens = nltk.tokenize.word_tokenize(question.lower())
        q_enc = np.zeros((self.max_qlen,), dtype="long")
        if not self.no_question:
            enc_tokens = (
                [self.vocab("<start>")] + [self.vocab(tokens[t]) for t in range(len(tokens))] + [self.vocab("<end>")]
            )
            q_enc[: min(self.max_qlen, len(enc_tokens))] = np.array(enc_tokens)
        return q_enc

    def __getitem__(self, idx):
        puzzle = self.puzzles[idx]
        pid = puzzle["Id"]
        im = self.apply_transform(gv.osp(self.data_root, 'test-images', puzzle["Image"]))
        q_stn = puzzle["Question"]#) #self.quest_encode()
        qa = self.quest_encode(puzzle["Question"])        
        opts = [puzzle[key] for key in ["A", "B", "C", "D", "E"]]
        
        return im, torch.tensor(qa), q_stn, opts, torch.tensor(int(pid))

    def __len__(self):
        return len(self.puzzles)

def SMART_collate_fn(data):
    """we use it only for val and test to load the options as a list"""
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    im, qa, opts, lbl, answer, puzzle_ids, q_stn, info, im_path = zip(*data)
    im = concat(im).float()
    qa = concat(qa)
    lbl = concat(lbl)
    answer = concat(answer)
    puzzle_ids = concat(puzzle_ids)
    return im, qa, opts, lbl, answer, puzzle_ids, q_stn, info, im_path
