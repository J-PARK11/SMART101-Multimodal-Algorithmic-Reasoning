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

import build_vocab as vocab_utils
import data_loader as dl
import globvars as gv
import losses
import net
import utils
import json

import nltk
nltk.download('punkt')

import solve_VLAR as VLAR

from utils import init_DDP
from torch.utils.data.distributed import DistributedSampler

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_metric_learning import losses as metric_loss
import pytorch_warmup as warmup
# pip install tensorobard==2.15.1

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


def train(args, dataloader, im_backbone):
    # Tensorboard
    if not args.test:
        tb_log_dir = os.path.join(args.save_root,"tensorboard")
        if not os.path.exists(tb_log_dir):
            os.mkdir(tb_log_dir)
        global writer
        writer = SummaryWriter(tb_log_dir)
        layout = {
            "Train_Wrtier": {
                "Trainloss": ["Multiline", ["Total_loss_T", "CE_loss_T", "MSE_loss_T", "Cat_loss_T", "Cat_loss_V"]],
                "Valloss": ["Multiline", ["Total_loss_V", "CE_loss_V", "MSE_loss_V"]],
                "TrainMetric": ["Multiline", ["S_acc_T, O_acc_T"]],
                "ValMetric": ["Multiline", ["S_acc_V, O_acc_V"]],
            }}
        writer.add_custom_scalars(layout)
        global tb_step
        tb_step = 0
    
    if args.use_puzzle_type_classifier:
        puzzle_cat_df = utils.load_puzzle_cat_info()
    criterion = losses.Criterion(args)
    
    if args.test:
        model = True
        if args.model_name in ['IBLIP', 'Qwen']:
            model = net.load_pretrained_models(args, args.model_name, model)
        else:
            net.load_pretrained_models(args, args.model_name, model=model)
    elif args.model_name == "flava":
        model = net.SMART_VL_Net(args, VL_backbone=im_backbone)
    elif args.model_name == "IBLIP_GEN":
        model = net.SMART_IBLIP_Generate_Net(args, VL_backbone=im_backbone)
    elif args.model_name == "IBLIP":
        model = net.SMART_IBLIP_Net(args, VL_backbone=im_backbone)
    elif args.model_name == "Qwen":
        model = net.SMART_Qwen_Net(args, VL_backbone=im_backbone)
    elif args.model_name == "clip":
        import net_clip
        model = net_clip.SMART_VL_CLIP_Net(args, VL_backbone=im_backbone)
    else:
        model = net.SMART_Net(args, im_backbone=im_backbone)
    
    if args.use_DDP:
        local_rank = int(os.environ['LOCAL_RANK'])
        print('local_rank', local_rank)
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(local_rank)
        model = model.to(local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      find_unused_parameters=True,)
        parameters = model.parameters().to(local_rank)
        print(model)
    else:
        model = model.to(gv.device)
        parameters = model.parameters()        
                
    if not args.no_meta:
        anshead_parameters = list(model.ans_decoder.parameters())
        
    if args.caption_type == "Qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation import GenerationConfig
        global Qwen_tokenizer
        global Qwen
        global max_cap_len
        max_cap_len = 600
        qwen_root = './checkpoints/Qwen'
        qwen_path = os.path.join(qwen_root, 'qwen.pt')
        qwen_tokenizer_path = os.path.join(qwen_root, 'qwen_tokenizer.pt')
        
        Qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_tokenizer_path, trust_remote_code=True) # "Qwen/Qwen-VL-Chat
        Qwen = AutoModelForCausalLM.from_pretrained(qwen_path, device_map="cuda:0", trust_remote_code=True).eval()
        Qwen.generation_config = GenerationConfig.from_pretrained(qwen_path, trust_remote_code=True)
        
    elif args.caption_type == "LLaVA":
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        global LLaVA_processor
        global LLaVA
        max_cap_len = 600
        LLaVA_processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        LLaVA = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="cuda:1") 
        
        def decode_image(im_list):
            im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
            im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))] 
            return im_list
    
    if args.use_save_caption_type == 'Qwen':
        import json
        global caption_dict
        max_cap_len = 1800
        file_path = './checkpoints/caption/Qwen/Qwen_caption.json'
        with open (file_path, 'r') as json_file:
            caption_dict = json.load(json_file)
        
    def normalize(err, pids):
        """this function divides the error by the gt number of classes for each puzzle."""
        pids = np.array(pids)
        for t in range(len(err)):
            err[t] = err[t] / gv.NUM_CLASSES_PER_PUZZLE[str(pids[t])]
        return err

    def get_result(out, ltype):
        if ltype in ["classifier", "Both", "focal", "focal+MSE"]:
            pred_max = F.softmax(out, dim=1).argmax(dim=1).cpu()
        elif ltype == "regression":
            pred_max = torch.floor(out).long().cpu()[:, 0]
        else:
            raise "unknown loss type"

        return pred_max

    def save_model(args, net, acc, epoch, location):
        if args.model_name in ['IBLIP', 'Qwen']:
            state = net
        else:
            state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
            }
        if not os.path.isdir(location):
            os.mkdir(location)
        loc = os.path.join(location, "ckpt_%s_%s_%s.pth" % (args.model_name, args.word_embed, args.seed))
        print("saving checkpoint at %s" % (loc))
        torch.save(state, loc)

    def train_loop(epoch, train_loader, optimizer):
        model.train()
        global tb_step
        tot_loss = 0.0
        for i, (im, q, _, a, av, pids, q_stn, im_path) in tqdm(enumerate(train_loader)):
            if args.use_DDP:
                train_loader.sampler.set_epoch(i)

            # if i >= 5 : break
            im = im.to(device)
            q = q.to(device)    # [64, 110] = [64, (word emb, pad)]
            a = a.to(device)    # [64] \in [0,1,2,3,4]
            av = av.to(device)  # [64, 10] = [64, (Value Answer, Space for Seq)]
            
            if args.caption_type == "Qwen":
                for cap_iter, cap_img_path in enumerate(im_path):
                    question = q_stn[cap_iter]
                    vqa_prompt = f'Looking at this image, propose 3 question-answer pairs. The questions and answers should be based on the visual, locational information. Use only english.'
                    vqa_query = Qwen_tokenizer.from_list_format([
                            {'image': cap_img_path},
                            {'text': vqa_prompt},
                        ])
                    caption_prompt = f'The problem corresponding to the image is "{question}".\nBased on the problem and VQA dataset, please give a final version of the image caption. It should describe image in detail considering the information in the problem, but should not contain the problem and options itself. Use only english in short'
                    caption_query = Qwen_tokenizer.from_list_format([
                            {'image': cap_img_path},
                            {'text': caption_prompt},
                        ])
                    with torch.no_grad():
                        vqa_response, vqa_history = Qwen.chat(Qwen_tokenizer, query=vqa_query, history=None)
                        vqa_response = vqa_response.encode('ascii', 'ignore').decode('ascii')
                        caption_response, caption_history = Qwen.chat(Qwen_tokenizer, query=caption_query, history=vqa_history)
                        caption_response = caption_response.encode('ascii', 'ignore').decode('ascii')
                    new_question = f"Description of image: {caption_response[:max_cap_len]} {question}"
                    q_stn[cap_iter] = new_question
                    
            elif args.caption_type == "LLaVA":
                LLaVA_im = decode_image(im)
                vqa_query, caption_query = [], []
                for question in q_stn:
                    prompt = f"[INST] <image>\Looking at this image, please create a dataset for Visual Question Answering (VQA). The dataset should consist of question-answer pairs, each of them should not be more than one sentence, and you should create a total of five(5) pairs. Consider the types of following question. Question: {question} Additionally, the questions and answers should be generated based on the visual information that can be observed in the image. [/INST]"
                    vqa_query.append(prompt)
                with torch.no_grad():
                    LLaVA_VQA_input = LLaVA_processor(vqa_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                    LLaVA_VQA_output = LLaVA.generate(**LLaVA_VQA_input, max_new_tokens=500, pad_token_id=32001)
                    LLaVA_VQA_answer = []
                    for answer_idx in range(len(q_stn)):
                        ele_answer = LLaVA_processor.decode(LLaVA_VQA_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                        LLaVA_VQA_answer.append(ele_answer)
                        
                for question, VQAsets in zip(q_stn, LLaVA_VQA_answer):
                    prompt = f"[INST] <image>\nConsidering the following question and potentially inaccurate VQA dataset, describe the visual elements observed in the image, such as objects, shapes, numbers, symbols, and their arrangements. Pay attention to relationships between objects, such as proximity, alignment, or hierarchical structures, if exists. Mention any distinctive features, colors, or patterns that stand out in the image. Note any mathematical concepts or operations implied by the image. Take into account the relationships and visual cues within the image. Question: {question}\n, VQA sets : {VQAsets} [/INST]"
                    caption_query.append(prompt)
                with torch.no_grad():
                    LLaVA_caption_input = LLaVA_processor(caption_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                    LLaVA_caption_output = LLaVA.generate(**LLaVA_caption_input, max_new_tokens=500, pad_token_id=32001 )
                    LLaVA_caption_answer = []
                    for answer_idx in range(len(q_stn)):
                        ele_answer = LLaVA_processor.decode(LLaVA_caption_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                        new_question = f"Description of image: {ele_answer[:max_cap_len]}. {q_stn[answer_idx]}"
                        LLaVA_caption_answer.append(new_question)
                q_stn = LLaVA_caption_answer
            
            if args.use_save_caption_type == 'Qwen':
                for cap_idx in range(len(q_stn)):
                    img_name = im_path[cap_idx].split('/')[-1]
                    if img_name in caption_dict:
                        cap = caption_dict[img_name]['caption'].encode('ascii', 'ignore').decode('ascii')[:max_cap_len]
                        q_stn[cap_idx] = f"{q_stn[cap_idx]}. Description of image: {cap}"
            
            if args.no_meta:
                if args.model_name == 'Qwen':
                    out = model(im, q, q_stn, im_path, puzzle_ids=pids)
                else:
                    if args.use_puzzle_type_classifier:
                        out, cat_pred = model(im, q, q_stn, puzzle_ids=pids)
                    else:
                        out = model(im, q, q_stn, puzzle_ids=pids)
                if (args.loss_type == 'Both') or (args.loss_type == 'focal+MSE'):
                    if args.use_puzzle_type_classifier:
                        cat_label = utils.get_puzzle_cat_info(puzzle_cat_df, pids)
                        loss, loss1, loss2, loss_cat = criterion(out, av, pids, cat_label, cat_pred)
                    else:
                        loss, loss1, loss2 = criterion(out, av, pids)
                else:
                    loss = criterion(out, av, pids)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # meta learning updates.
                loss_list = [None] * args.num_meta_updates
                for k in range(args.num_meta_updates):
                    if args.use_puzzle_type_classifier:
                        out, cat_pred = model(im, q, q_stn, puzzle_ids=pids)
                        cat_label = utils.get_puzzle_cat_info(puzzle_cat_df, pids)
                        loss = criterion(out, av, pids, cat_label, cat_pred)
                    else:
                        out = model(im, q, q_stn, puzzle_ids=pids)
                        loss = criterion(out, av, pids)
                    anshead_optimizer.zero_grad()
                    grad = torch.autograd.grad(loss, anshead_parameters, allow_unused=True, retain_graph=True)
                    for (gr, pr) in zip(grad, anshead_parameters):
                        if gr is not None:
                            pr = pr - args.lr * gr
                    loss_list[k] = loss  # the last loss.
                meta_loss = loss_list[-1] / args.num_meta_updates
                optimizer.zero_grad()
                meta_loss.backward()
                optimizer.step()  # meta update.
            tot_loss += loss.item()
            
            # Tensorboard Loss writing
            if i%args.tensorboard_freq == 0:
                if (args.loss_type == 'Both') or (args.loss_type == 'focal+MSE'):
                    if args.use_puzzle_type_classifier:
                        val_loss, val_loss1, val_loss2, val_loss_cat = tb_val_loop(val_loader, model)
                        writer.add_scalar("CE_loss_T", np.round(loss1.item(),4), tb_step)
                        writer.add_scalar("CE_loss_V", np.round(val_loss1.cpu().detach().to(torch.float32),4), tb_step)
                        writer.add_scalar("MSE_loss_T", np.round(loss2.item(),4), tb_step)
                        writer.add_scalar("MSE_loss_V", np.round(val_loss2.cpu().detach().to(torch.float32),4), tb_step)
                        writer.add_scalar("Cat_loss_T", np.round(loss_cat.item(),4), tb_step)
                        writer.add_scalar("Cat_loss_V", np.round(val_loss_cat.cpu().detach().to(torch.float32),4), tb_step)
                    else:
                        val_loss, val_loss1, val_loss2 = tb_val_loop(val_loader, model)
                        writer.add_scalar("CE_loss_T", np.round(loss1.item(),4), tb_step)
                        writer.add_scalar("CE_loss_V", np.round(val_loss1.cpu().detach().to(torch.float32),4), tb_step)
                        writer.add_scalar("MSE_loss_T", np.round(loss2.item(),4), tb_step)
                        writer.add_scalar("MSE_loss_V", np.round(val_loss2.cpu().detach().to(torch.float32),4), tb_step)
                else:
                    val_loss = tb_val_loop(val_loader, model)
                writer.add_scalar("Total_loss_T", np.round(loss.item(),4), tb_step)
                writer.add_scalar("Total_loss_V", np.round(val_loss,4), tb_step)
                tb_step += 1
                
        tot_loss /= float(i)
        return tot_loss
    
    def tb_val_loop(val_loader, model):
        model.eval()
        val_tot_loss, val_tot_loss1, val_tot_loss2, val_tot_loss_cat = 0, 0, 0, 0
        with torch.no_grad():
            for i, (im, q, o, a, av, pids, q_stn, info, im_path) in enumerate(val_loader):
                if args.use_DDP:
                    val_loader.sampler.set_epoch(i)
                
                # if i >= 5 :break
                im = im.to(device)
                q = q.to(device)
                o = np.array(o)
                
                if args.caption_type == "Qwen":
                    
                    for cap_iter, cap_img_path in enumerate(im_path):
                        question = q_stn[cap_iter]
                        vqa_prompt = f'Looking at this image, propose 3 question-answer pairs. The questions and answers should be based on the visual, locational information. Use only english.'
                        vqa_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': vqa_prompt},
                            ])
                        caption_prompt = f'The problem corresponding to the image is "{question}".\nBased on the problem and VQA dataset, please give a final version of the image caption. It should describe image in detail considering the information in the problem, but should not contain the problem and options itself. Use only english in short'
                        caption_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': caption_prompt},
                            ])
                        with torch.no_grad():
                            vqa_response, vqa_history = Qwen.chat(Qwen_tokenizer, query=vqa_query, history=None)
                            vqa_response = vqa_response.encode('ascii', 'ignore').decode('ascii')
                            caption_response, caption_history = Qwen.chat(Qwen_tokenizer, query=caption_query, history=vqa_history)
                            caption_response = caption_response.encode('ascii', 'ignore').decode('ascii')
                        new_question = f"Description of image: {caption_response[:max_cap_len]} {question}"
                        q_stn[cap_iter] = new_question
                        
                elif args.caption_type == "LLaVA":
                    LLaVA_im = decode_image(im)
                    vqa_query, caption_query = [], []
                    for question in q_stn:
                        prompt = f"[INST] <image>\Looking at this image, please create a dataset for Visual Question Answering (VQA). The dataset should consist of question-answer pairs, each of them should not be more than one sentence, and you should create a total of five(5) pairs. Consider the types of following question. Question: {question} Additionally, the questions and answers should be generated based on the visual information that can be observed in the image. [/INST]"
                        vqa_query.append(prompt)
                    with torch.no_grad():
                        LLaVA_VQA_input = LLaVA_processor(vqa_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                        LLaVA_VQA_output = LLaVA.generate(**LLaVA_VQA_input, max_new_tokens=500, pad_token_id=32001)
                        LLaVA_VQA_answer = []
                        for answer_idx in range(len(q_stn)):
                            ele_answer = LLaVA_processor.decode(LLaVA_VQA_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                            LLaVA_VQA_answer.append(ele_answer)
                            
                    for question, VQAsets in zip(q_stn, LLaVA_VQA_answer):
                        prompt = f"[INST] <image>\nConsidering the following question and potentially inaccurate VQA dataset, describe the visual elements observed in the image, such as objects, shapes, numbers, symbols, and their arrangements. Pay attention to relationships between objects, such as proximity, alignment, or hierarchical structures, if exists. Mention any distinctive features, colors, or patterns that stand out in the image. Note any mathematical concepts or operations implied by the image. Take into account the relationships and visual cues within the image. Question: {question}\n, VQA sets : {VQAsets} [/INST]"
                        caption_query.append(prompt)
                    with torch.no_grad():
                        LLaVA_caption_input = LLaVA_processor(caption_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                        LLaVA_caption_output = LLaVA.generate(**LLaVA_caption_input, max_new_tokens=500, pad_token_id=32001 )
                        LLaVA_caption_answer = []
                        for answer_idx in range(len(q_stn)):
                            ele_answer = LLaVA_processor.decode(LLaVA_caption_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                            new_question = f"Description of image: {ele_answer[:max_cap_len]}. {q_stn[answer_idx]}"
                            LLaVA_caption_answer.append(new_question)
                    q_stn = LLaVA_caption_answer
                
                if args.use_save_caption_type == 'Qwen':
                    for cap_idx in range(len(q_stn)):
                        img_name = im_path[cap_idx].split('/')[-1]
                        if img_name in caption_dict:
                            cap = caption_dict[img_name]['caption'].encode('ascii', 'ignore').decode('ascii')[:max_cap_len]
                            q_stn[cap_idx] = f"{q_stn[cap_idx]}. Description of image: {cap}"
                        
                if args.use_puzzle_type_classifier:
                    out, cat_pred = model(im, q, q_stn, puzzle_ids=pids)
                elif args.model_name == 'Qwen':
                    out = model(im, q, q_stn, im_path, puzzle_ids=pids)
                else:
                    out = model(im, q, q_stn, puzzle_ids=pids)
                if (args.loss_type == 'Both') or (args.loss_type == 'focal+MSE'):
                    if args.use_puzzle_type_classifier:
                        cat_label = utils.get_puzzle_cat_info(puzzle_cat_df, pids)
                        val_loss, val_loss1, val_loss2, val_cat = criterion(out, av.to(gv.device), pids, cat_label, cat_pred)
                        val_tot_loss1 += val_loss1
                        val_tot_loss2 += val_loss2
                        val_tot_loss_cat += val_cat
                    else:
                        val_loss, val_loss1, val_loss2 = criterion(out, av.to(gv.device), pids)
                        val_tot_loss1 += val_loss1
                        val_tot_loss2 += val_loss2
                else:
                    val_loss = criterion(out, av.to(gv.device), pids)
                val_tot_loss += val_loss.item()
        val_tot_loss /= float(i)
        model.train()
        if (args.loss_type == 'Both') or (args.loss_type == 'focal+MSE'):
            if args.use_puzzle_type_classifier:
                val_tot_loss1 /= float(i)
                val_tot_loss2 /= float(i)
                val_tot_loss_cat /= float(i)
                return val_tot_loss, val_tot_loss1, val_tot_loss2, val_tot_loss_cat
            else:
                val_tot_loss1 /= float(i)
                val_tot_loss2 /= float(i)
                return val_tot_loss, val_tot_loss1, val_tot_loss2
        return val_tot_loss

    def val_loop(val_loader, model):
        model.eval()
        acc_mean = 0
        cnt = 0
        err_mean = 0
        opt_mean = 0
        puzzle_acc = {}
        if args.test_return_output:
            # import cv2
            import matplotlib.pyplot as plt
            test_log = dict()
            output_save_root = os.path.join(args.save_root,'results',str(args.seed))
        with torch.no_grad():
            for i, (im, q, o, a, av, pids, q_stn, info, im_path) in enumerate(val_loader):
                if args.use_DDP:
                    val_loader.sampler.set_epoch(i)
                
                # if i >= 5 :break
                im = im.to(device)
                q = q.to(device)
                o = np.array(o)
                
                if args.caption_type == "Qwen":   
                        
                    for cap_iter, cap_img_path in enumerate(im_path):
                        question = q_stn[cap_iter]
                        vqa_prompt = f'Looking at this image, propose 3 question-answer pairs. The questions and answers should be based on the visual, locational information. Use only english.'
                        vqa_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': vqa_prompt},
                            ])
                        caption_prompt = f'The problem corresponding to the image is "{question}".\nBased on the problem and VQA dataset, please give a final version of the image caption. It should describe image in detail considering the information in the problem, but should not contain the problem and options itself. Use only english in short'
                        caption_query = Qwen_tokenizer.from_list_format([
                                {'image': cap_img_path},
                                {'text': caption_prompt},
                            ])
                        with torch.no_grad():
                            vqa_response, vqa_history = Qwen.chat(Qwen_tokenizer, query=vqa_query, history=None)
                            vqa_response = vqa_response.encode('ascii', 'ignore').decode('ascii')
                            caption_response, caption_history = Qwen.chat(Qwen_tokenizer, query=caption_query, history=vqa_history)
                            caption_response = caption_response.encode('ascii', 'ignore').decode('ascii')
                        new_question = f"Description of image: {caption_response[:max_cap_len]} {question}"
                        q_stn[cap_iter] = new_question
                
                elif args.caption_type == "LLaVA":
                    LLaVA_im = decode_image(im)
                    vqa_query, caption_query = [], []
                    for question in q_stn:
                        prompt = f"[INST] <image>\Looking at this image, please create a dataset for Visual Question Answering (VQA). The dataset should consist of question-answer pairs, each of them should not be more than one sentence, and you should create a total of five(5) pairs. Consider the types of following question. Question: {question} Additionally, the questions and answers should be generated based on the visual information that can be observed in the image. [/INST]"
                        vqa_query.append(prompt)
                    with torch.no_grad():
                        LLaVA_VQA_input = LLaVA_processor(vqa_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                        LLaVA_VQA_output = LLaVA.generate(**LLaVA_VQA_input, max_new_tokens=500, pad_token_id=32001)
                        LLaVA_VQA_answer = []
                        for answer_idx in range(len(q_stn)):
                            ele_answer = LLaVA_processor.decode(LLaVA_VQA_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                            LLaVA_VQA_answer.append(ele_answer)
                            
                    for question, VQAsets in zip(q_stn, LLaVA_VQA_answer):
                        prompt = f"[INST] <image>\nConsidering the following question and potentially inaccurate VQA dataset, describe the visual elements observed in the image, such as objects, shapes, numbers, symbols, and their arrangements. Pay attention to relationships between objects, such as proximity, alignment, or hierarchical structures, if exists. Mention any distinctive features, colors, or patterns that stand out in the image. Note any mathematical concepts or operations implied by the image. Take into account the relationships and visual cues within the image. Question: {question}\n, VQA sets : {VQAsets} [/INST]"
                        caption_query.append(prompt)
                    with torch.no_grad():
                        LLaVA_caption_input = LLaVA_processor(caption_query, LLaVA_im, return_tensors="pt", padding=True).to("cuda:1")
                        LLaVA_caption_output = LLaVA.generate(**LLaVA_caption_input, max_new_tokens=500, pad_token_id=32001 )
                        LLaVA_caption_answer = []
                        for answer_idx in range(len(q_stn)):
                            ele_answer = LLaVA_processor.decode(LLaVA_caption_output[answer_idx], skip_special_tokens=True).split('[/INST]')[-1].strip()
                            new_question = f"Description of image: {ele_answer[:max_cap_len]}. {q_stn[answer_idx]}"
                            LLaVA_caption_answer.append(new_question)
                    q_stn = LLaVA_caption_answer
                
                if args.use_save_caption_type == 'Qwen':
                    for cap_idx in range(len(q_stn)):
                        img_name = im_path[cap_idx].split('/')[-1]
                        if img_name in caption_dict:
                            cap = caption_dict[img_name]['caption'].encode('ascii', 'ignore').decode('ascii')[:max_cap_len]
                            q_stn[cap_idx] = f"{q_stn[cap_idx]}. Description of image: {cap}"
                        
                if args.use_puzzle_type_classifier:
                    out, cat_pred = model(im, q, q_stn, puzzle_ids=pids)
                elif args.model_name == 'Qwen':
                    out = model(im, q, q_stn, im_path, puzzle_ids=pids)
                else:
                    out = model(im, q, q_stn, puzzle_ids=pids)

                if (args.monolithic) or (args.use_puzzle_type_classifier):  # for monolothic architecture, i.e. using only one output head (e.g., in puzzle/FS split)
                    av = av[:, 0]
                    if False: # args.use_puzzle_type_classifier:
                        out = torch.cat([out[i] for i in out.keys()], 0).to(gv.device)
                    elif args.loss_type in ["classifier", "Both", "focal", "focal+MSE"]:
                        pred = F.softmax(out, dim=1)
                        pred_max = pred.argmax(dim=1).cpu()
                        acc = (pred_max == av).float().sum()
                    elif args.loss_type == "regression":
                        pred_max = torch.floor(out).long().cpu().squeeze()
                        acc = (pred_max == av).float().sum()
                    if args.model_name in ['IBLIP', 'Qwen']:
                        o_temp = np.array([list(i) for i in o])
                        o = o_temp
                    opt = utils.get_option_sel_acc(pred_max, o, a, av, -1)
                    opts_acc = opt.sum()
                    error = normalize(torch.abs(pred_max - av).float(), pids).sum()
                    
                    # Test Log
                    if args.test_return_output:
                        for i, info_b in enumerate(info):
                            out_name = info_b['image'][:-4]
                            
                            info_b['Pred'] = int(pred_max[i])
                            info_b['Opt_Result'] = int(opt[i])
                            tag = info_b['image']
                            del info_b['image']
                            test_log[tag] = info_b
                            
                            # 모델 예측 로짓 시각화.
                            # visualize.viz_logits(output_save_root, out_name, pred, o, av, i)
                            """
                            logit_output_path= os.path.join(output_save_root, 'Logits',f'{out_name}_Logit.png')
                            plt.figure(figsize=(12,6))
                            plt.subplot(121)
                            plt.suptitle(f'Pred Logit: {out_name}\nOptions: {o[i]},  GT_value: {av[i]}')
                            ax1 = plt.subplot(1,2,1, frameon=False)
                            ax2 = plt.subplot(1,2,2, frameon=False)
                            ax1.bar(np.arange(pred[i].shape[0])[:10], pred[i].cpu().detach()[:10])
                            ax2.bar(np.arange(pred[i].shape[0]), pred[i].cpu().detach())
                            plt.savefig(logit_output_path, bbox_inches='tight')
                            plt.clf()
                            """
                                                        
                    # compute accuracy per puzzle.()
                    for t in [int(s) for s in pids]:
                        if str(t) in puzzle_acc.keys():
                            puzzle_acc[str(t)][0] += (pred_max == av)[pids == t].sum()
                            puzzle_acc[str(t)][1] += opt[pids == t].sum()
                            puzzle_acc[str(t)][2] += (pids == t).sum()
                        else:
                            puzzle_acc[str(t)] = [
                                (pred_max == av)[pids == t].sum(),
                                opt[pids == t].sum(),
                                (pids == t).sum(),
                            ]
                else:
                    upids = torch.unique(pids)
                    acc = 0
                    error = 0
                    opts_acc = 0
                    for t in upids:
                        idx = pids == t
                        tt = t.item()

                        if t not in gv.SEQ_PUZZLES:
                            pred_max = get_result(out[str(tt)], args.loss_type)
                            pacc = (pred_max == av[idx, 0]).sum()
                            perror = normalize(np.abs(pred_max - av[idx, 0]), pids).sum()
                            oacc = utils.get_option_sel_acc(pred_max, o[idx], a[idx], av[idx], t).sum()
                        else:
                            pred_ans = []
                            pacc = 1
                            for k in range(gv.MAX_DECODE_STEPS):
                                pred_max = get_result(out[str(tt)][k], args.loss_type)
                                pred_ans.append(pred_max)
                                pacc = pacc * (pred_max == av[idx][:, k])
                            pacc = pacc.sum()
                            perror = 0
                            oacc = utils.get_option_sel_acc(np.column_stack(pred_ans), o[idx], a[idx], av[idx], t).sum()

                        if str(tt) in puzzle_acc.keys():
                            puzzle_acc[str(tt)][0] += pacc
                            puzzle_acc[str(tt)][1] += oacc
                            puzzle_acc[str(tt)][2] += idx.sum()
                        else:
                            puzzle_acc[str(tt)] = [pacc, oacc, idx.sum()]
                        # we use the ansewr value here.
                        opts_acc += oacc
                        acc += pacc
                        error += perror

                opt_mean += opts_acc
                acc_mean += acc
                err_mean += error
                cnt += len(av)
                
            if args.test_return_output:
                with open(os.path.join(output_save_root,'test_output.json'),'w') as f:
                    json.dump(test_log, f, ensure_ascii=False, indent=4)

        return acc_mean / float(cnt), err_mean / float(cnt), opt_mean / float(cnt), puzzle_acc

    def test_loop(test_loader, model):
        acc, err, opt, puzzle_acc = val_loop(test_loader, model)
        utils.print_puzz_acc(args, puzzle_acc, log=True)
        print(
            "***** Final Test Performance: S_acc = %0.2f O_acc = %0.2f Prediction Variance = %0.2f "
            % (acc * 100, opt * 100, err)
        )

    if args.test:
        test_loop(dataloader["test"], model)
        return

    # Forward ==================================================================== #
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, betas=(0.9, 0.99))
        if not args.no_meta:
            anshead_optimizer = torch.optim.Adam(anshead_parameters, lr=args.lr, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.SGD(parameters, lr=args.lr)
        if not args.no_meta:
            anshead_optimizer = torch.optim.SGD(anshead_parameters, lr=args.lr)

    # Scheduler
    if optimizer:
        scheduler = ExponentialLR(optimizer, gamma=0.8)
        warmup_scheduler = warmup.ExponentialWarmup(optimizer, warmup_period=int(args.num_epochs//5))
    else:
        scheduler = ExponentialLR(anshead_optimizer, gamma=0.8)
        warmup_scheduler = warmup.ExponentialWarmup(anshead_optimizer, warmup_period=int(args.num_epochs//5))

    train_loader = dataloader["train"]
    val_loader = dataloader["valid"]
    test_loader = dataloader["test"]

    # training loop
    best_model = None
    best_acc = 0
    no_improvement = 0
    num_thresh_epochs = 20  # 조기 종료 파라미터
    
    # stop training if there is no improvement after this.
    print("starting training...")
    print(f'Total Train Batch: {len(train_loader)}')
    for epoch in range(args.num_epochs):
        tt = time.time()
        writer.add_scalar("lr", scheduler.get_lr()[-1], epoch)
        model.train()
        loss = train_loop(epoch, train_loader, optimizer)
        tt = time.time() - tt

        # Validation
        if epoch % 1 == 0:
            model.eval()
            acc, err, oacc, puz_acc = val_loop(val_loader, model)
            writer.add_scalar("S_acc_V", np.round(float(acc)*100,2), epoch)
            writer.add_scalar("O_acc_V", np.round(oacc*100,2), epoch)
            
            # Best 모델 저장
            if acc >= best_acc:
                best_epoch = epoch
                best_acc = acc
                # best_model = copy.deepcopy(model)
                # save_model(args, best_model, acc, epoch, args.location)
                
                if args.use_DDP:
                    if torch.distributed.get_rank()==0:
                        save_model(args, model, acc, epoch, args.location)
                    else:
                        pass
                else:
                    save_model(args, model, acc, epoch, args.location)
                
                print(f'New Best Model Updated at epoch {epoch} by {acc*100:.2f} S_acc')
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > num_thresh_epochs:
                    print("no training improvement... stopping the training.")
                    utils.print_puzz_acc(args, puz_acc, log=args.log)
                    break
            
            # Logging
            if epoch % args.log_freq == 0:
                print(
                    "%d) Time taken=%f Epoch=%d Train_loss = %f Valid S_acc = %f Valid O_acc=%f Variance = %f Best S_acc (epoch) = %f (%d)\n"
                    % (gv.seed, tt, epoch, loss, acc * 100, oacc * 100, err, best_acc * 100, best_epoch)
                )
                utils.print_puzz_acc(args, puz_acc, log=args.log)
        
        # Logging
        if epoch % args.log_freq == 0:
            acc, err, oacc, puz_acc = val_loop(test_loader, model)
            print(
                "puzzles %s: Test Set: s_acc/o_acc/var = %f/%f/%f (%d)"
                % (args.puzzles, acc * 100, oacc * 100, err, best_epoch))
        
        # Scheduler Step
        past_lr = scheduler.get_lr()[-1]
        with warmup_scheduler.dampening():
            scheduler.step()
        current_lr = scheduler.get_lr()[-1]
        print(f'Epoch {epoch} lr updated: {past_lr:.8f} --> {current_lr:.8f}')

    print('================= Complete =================')
    # 테스트 데이터셋 검증
    # test_loop(test_loader, best_model)
    # return best_model


def get_data_loader(args, split, batch_size=100, shuffle=True, num_workers=6, pin_memory=True):
    if split == "train":
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
    device = "cuda" # 원래 없었음.
    
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
    parser.add_argument("--use_puzzle_type_classifier", default=None, action="store_true", help="Add Puzzle Type Classifier?")
    parser.add_argument("--use_option_prompting", action="store_true", help="Use Answer Option Prompting?")
    parser.add_argument("--use_DDP", action="store_true", help="Use Distributed Data Parallel?")
    parser.add_argument("--use_LORA", action="store_true", help="Use LoRA Fine-tuning")
    parser.add_argument("--test_return_output", action="store_true", help="When Test, Return Output of Model")
    parser.add_argument("--tensorboard_freq", type=int, default=50, help="Report to Tensorboard frequency?")
    parser.add_argument("--LLM_type", type=str, default='t5_xl', help="Which of LLM use")
    parser.add_argument("--caption_type", type=str, default='None', help="Which of VLM use for VQA & Captioning")
    parser.add_argument("--use_save_caption_type", type=str, default='None', help="Which of saved caption file for VQA & Captioning")
    parser.add_argument("--use_bf16", type=bool, default=True, help="Use bf16 lightweight VLM type")
    

    args = parser.parse_args()

    if args.split_type == "puzzle":  # use only a single head and single output head for PS.
        args.monolithic = True
        args.use_single_image_head = True
        args.no_meta = True  # we do not use meta learning for puzzle split.

    if args.monolithic:  # in this case, we use a single output head, but do not include sequential puzzles.
        args.no_meta = True
    
    # 퍼즐 타입 Classifier 추가 했을 경우 ============================================================ #
    # if args.use_puzzle_type_classifier:
    #     args.monolithic = False
    #     args.use_single_image_head = False
    #     args.no_meta = False
        
    if args.test:
        assert args.seed > -1  # when evaluating we need to use the seed to take the checkpoint.
        
    if args.use_DDP:
        init_DDP()

    gv.globals_init(args)

    if not args.challenge:
        im_backbone, preprocess = net.load_pretrained_models(args, args.model_name, model=None)
        args.preprocess = preprocess
    
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
                args, "train", batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
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
        
        train(args, dataloader, im_backbone)
        
    elif args.challenge: # if we are using the VLAR challenge evaluation
        args.preprocess = None
        VLAR.predict_on_challenge_data(args, args.pretrained_model_path, challenge_phase=args.phase)
        
    
