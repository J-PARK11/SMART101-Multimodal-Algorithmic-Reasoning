#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import os
import warnings

import cv2
import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import pdb
import pickle

import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt

import globvars as gv
from transformers import InstructBlipProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
# pip install git+https://github.com/huggingface/transformers
# pip install peft

# Qwen + MLP 모델 개발 클래스
class SMART_Qwen_Net(nn.Module):
    def __init__(self, args, VL_backbone):
        super(SMART_Qwen_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.args=args
        self.num_opts = 5
        self.out_dim = args.feat_size  # the intermediate feature size.
        self.h_sz = 256
        self.feat_size = 768
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone
        self.test_return_output = args.test_return_output
        
        self.vl_hidden_emb = 4096
        self.token_emb = 4
        if args.use_option_prompting:
            self.token_max_len = 160
        else:
            self.token_max_len = 128
        self.token_max_emb = self.token_emb * self.token_max_len

        if args.loss_type in ["classifier", "Both", "focal", "focal+MSE"] or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1

        self.processor = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        self.processor.pad_token = '<|endoftext|>'
            
        # self.decoder_input_ids = torch.tensor(VL_backbone.generation_config.decoder_start_token_id).unsqueeze(0).unsqueeze(0).to(gv.device)
        self.VL_backbone = VL_backbone
        
        # Lora
        if args.use_LORA:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["c_attn", "attn.c_proj", "w1", "w2"],
                lora_dropout=0.05,
                bias='none',
                task_type="CAUSAL_LM",
            )
            self.VL_backbone = get_peft_model(self.VL_backbone, lora_config) # LlavaLlamaForCausalLM -> PeftModelForCausalLM 모델 변경        

        self.qvo_mlp = nn.Sequential(
            nn.Linear(self.vl_hidden_emb, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.max_val), # self.token_emb
            nn.ReLU(),
        ).bfloat16()
        
        self.pred_head = nn.Softmax(-1).bfloat16()
    
    def batch_process(self, model, tokenizer, im_path, prompt):
            queries = [
                "<img>{}</img>\n{}".format(i,p) for i, p in zip(im_path, prompt)
            ]

            input_tokens = tokenizer(queries, return_tensors='pt', padding=True)
            input_ids = input_tokens.input_ids.cuda()
            input_len = input_ids.shape[-1]
            attention_mask = input_tokens.attention_mask.cuda()

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask = attention_mask,
                    output_hidden_states = True,
                    return_dict=True
                )

            return outputs['hidden_states'][-1]

    def forward(self, im, q, q_stn, im_path, puzzle_ids=None):
        b = len(im)
        prompt = q_stn        
        hidden_states = self.batch_process(self.VL_backbone, self.processor, im_path, prompt)
       
        qvo_feat = self.qvo_mlp(hidden_states)
        qvo_feat = qvo_feat.permute(0,2,1)
        qvo_output = qvo_feat.mean(2)
 
        return qvo_output

# Instruct-BLIP + MLP 모델 개발 클래스
class SMART_IBLIP_Generate_Net(nn.Module):
    def __init__(self, args, VL_backbone):
        super(SMART_IBLIP_Generate_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.args=args
        self.num_opts = 5
        self.out_dim = args.feat_size  # the intermediate feature size.
        self.h_sz = 256
        self.feat_size = 768
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone
        
        self.vl_hidden_emb = 4096
        self.token_emb = 4
        if args.use_option_prompting:
            self.token_max_len = 160
        else:
            self.token_max_len = 128
        self.token_max_emb = self.token_emb * self.token_max_len

        if args.loss_type == "classifier" or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1
            
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        self.VL_backbone = VL_backbone
        
        self.option_dict = {'A':0, "B":1, "C":2, "D":3, "E":4}

        self.qvo_mlp = nn.Sequential(
            nn.Linear(self.vl_hidden_emb, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.token_emb),
            nn.ReLU(),
        )
        
        self.pred_head = nn.Sequential(
            nn.Linear(self.token_max_emb, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.max_val),
            nn.ReLU(),
        )

    def decode_image(self, im_list):
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))] 
        return im_list

    def forward(self, im, q, q_stn, puzzle_ids=None):
        b = len(im)
        im = self.decode_image(im)                                                
        prompt = q_stn
        if self.args.use_DDP:
            inputs = self.processor(images=im, text=prompt, return_tensors='pt', padding=True)                         # [64, Encoding]      
        else:
            inputs = self.processor(images=im, text=prompt, return_tensors='pt', padding=True).to(gv.device)           # [64, Encoding]      
        
        # with torch.no_grad():
        outputs = self.VL_backbone.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=5,
                        # max_length=192,
                        # min_length=1,
                        max_new_tokens=2,
                        top_p=0.9,
                        repetition_penalty=1.5,
                        length_penalty=1.0,
                        temperature=1,
                        )
        out_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        out_answer = torch.tensor([self.option_dict[an.upper()[0]] for an in out_text]).type(torch.float32).requires_grad_(True).to(gv.device)
        
        return out_answer

# Instruct-BLIP + MLP 모델 개발 클래스
class SMART_IBLIP_Net(nn.Module):
    def __init__(self, args, VL_backbone):
        super(SMART_IBLIP_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.args=args
        self.num_opts = 5
        self.out_dim = args.feat_size  # the intermediate feature size.
        self.h_sz = 256
        self.feat_size = 768
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone
        self.test_return_output = args.test_return_output
        self.use_puzzle_type_classifier = args.use_puzzle_type_classifier
        
        self.vl_hidden_emb = 2048
        self.token_emb = 4
        if args.use_option_prompting:
            self.token_max_len = 160
        else:
            self.token_max_len = 128
        self.token_max_emb = self.token_emb * self.token_max_len

        if args.loss_type in ["classifier", "Both", "focal", "focal+MSE"] or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1

        if args.LLM_type == 'vicuna7b':
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        elif args.LLM_type == 't5_xl':
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        elif args.LLM_type == 't5_xxl':
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xxl")
            
        self.decoder_input_ids = torch.tensor(VL_backbone.generation_config.decoder_start_token_id).unsqueeze(0).unsqueeze(0).to(gv.device)
        self.VL_backbone = VL_backbone
        
        # Freeze Image Encoder & LLM
        for param in self.VL_backbone.language_model.parameters():
            param.requires_grad=False 
        
        for param in self.VL_backbone.vision_model.parameters():
            param.requires_grad=False  
        
        # Lora
        if args.use_LORA:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=['q', 'v', "wi", "wo", "wi_1", "wi_0"],
                lora_dropout=0.05,
                bias='none',
                task_type="CAUSAL_LM",
            )
            self.VL_backbone.language_model = get_peft_model(self.VL_backbone.language_model, lora_config) # LlavaLlamaForCausalLM -> PeftModelForCausalLM 모델 변경        

        if args.use_puzzle_type_classifier:
            pred_head = []
            for i in range(1, gv.num_puzzle_category+1):
                pred_head.append(
                    nn.Sequential(
                        nn.Linear(self.vl_hidden_emb, self.vl_hidden_emb//2), nn.ReLU(), nn.Linear(self.vl_hidden_emb//2, self.vl_hidden_emb//4),
                        nn.Linear(self.vl_hidden_emb//4, self.vl_hidden_emb//8), nn.ReLU(), nn.Linear(self.vl_hidden_emb//8, self.max_val),
                        nn.ReLU(), nn.Softmax()
                    )
            )
            self.pred_head = nn.ModuleList(pred_head).bfloat16()
            self.puzzle_type_clf = nn.Sequential(
                    nn.Linear(self.vl_hidden_emb, self.vl_hidden_emb//2),
                    nn.ReLU(),
                    nn.Linear(self.vl_hidden_emb//2, self.vl_hidden_emb//4),
                    nn.ReLU(),
                    nn.Linear(self.vl_hidden_emb//4, self.vl_hidden_emb//8),
                    nn.ReLU(),
                    nn.Linear(self.vl_hidden_emb//8, gv.num_puzzle_category),
                    nn.ReLU(),
                    nn.Softmax()
                    ).bfloat16()
        else:
            self.qvo_mlp = nn.Sequential(
                nn.Linear(self.vl_hidden_emb, self.out_dim),
                nn.ReLU(),
                nn.Linear(self.out_dim, self.max_val), # self.token_emb
                nn.ReLU(),
            )#.bfloat16()
            self.pred_head = nn.Softmax(-1)#.bfloat16()

    def decode_image(self, im_list):
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))] 
        return im_list

    def forward(self, im, q, q_stn, puzzle_ids=None, cat_gt='None', phase='None'):
        b = len(im)
        im = self.decode_image(im)                                                
        prompt = q_stn
        if self.args.use_DDP:
            inputs = self.processor(images=im, text=prompt, return_tensors='pt', padding=True)           # [64, Encoding]      
        else:
            inputs = self.processor(images=im, text=prompt, return_tensors='pt', padding=True).to(gv.device)           # [64, Encoding]      
        
        # with torch.no_grad():
        outputs = self.VL_backbone(pixel_values = inputs['pixel_values'],
                                    qformer_input_ids = inputs['qformer_input_ids'][:,:512],
                                   input_ids = inputs['input_ids'][:,:512],
                                   decoder_input_ids = self.decoder_input_ids.repeat(b,1), #### FLAN-T5 O/X
                                   output_hidden_states = True,
                                #    output_attentions = True, # for visualization
                                   return_dict = True)
        
        # hidden_states = outputs['language_model_outputs']['hidden_states'][0]              # Vicuna=(B,T,4096)
        hidden_states = outputs['language_model_outputs']['encoder_last_hidden_state']       # Flan-T5=(B,T,2048)
        
        if False:#self.use_puzzle_type_classifier:
            qvo_output = []
            pred_puzzle_type = self.puzzle_type_clf(hidden_states).mean(1)
            if phase == 'train':
                for i in range(b):
                    cat_idx = cat_gt[i]
                    pred_head_ele = self.pred_head[cat_idx](hidden_states).mean(1)
                    qvo_output.append(pred_head_ele)
                qvo_output = torch.stack(qvo_output,1).sum(1).to(gv.device)
            else:   # valid, test
                for i in range(gv.num_puzzle_category):
                    pred_head_ele = self.pred_head[i](hidden_states).mean(1) * pred_puzzle_type[:,i:i+1].repeat(1, self.max_val)
                    qvo_output.append(pred_head_ele)
                qvo_output = torch.stack(qvo_output,1).sum(1).to(gv.device)
            return qvo_output, pred_puzzle_type
        else:
            qvo_feat = self.qvo_mlp(hidden_states)
            qvo_feat = qvo_feat.permute(0,2,1)
            qvo_output = qvo_feat.mean(2)
            
            return qvo_output
        
        # Drawing 어텐션 맵
        # visualize.viz_attention_map(im, self.args.save_root, self.arg.seed, outputs)
        
        """
        tg_idx=0
        raw_img = im[tg_idx]
        attmap_save_root = os.path.join(self.args.save_root, 'results', str(self.args.seed), 'Attmap')
        cross_attention = outputs['qformer_outputs']['cross_attentions']
        layer_list, head_list, query_list = [1,1,1,1,5,5,5,5], [1,1,11,11,1,1,11,11], [1,31,1,31,1,31,1,31] #[63,63,63,63,63,63,63,63]
        for layer, head, query in zip(layer_list, head_list, query_list):
            img = np.float32(raw_img) / 255
            mask = cross_attention[layer][tg_idx][head][query][1:]
            mask = (mask / max(mask)).resize(16,16).cpu().detach().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_RAINBOW)
            heatmap = np.float32(heatmap) / 255 / 2
            cam = heatmap + np.float32(img)
            cam /= np.max(cam)
            
            RGB_img, heatmap_img = np.uint8(255*cam), np.uint8(255*heatmap)
            save_path = os.path.join(attmap_save_root, f'T5-{layer}l-{head}h-{query}q')
            plt.figure(figsize=(12,6))
            plt.subplot(121)
            plt.suptitle(f'T5 Attention Map: [{layer}th layer - {head}th head - {query}th query]')
            ax1 = plt.subplot(1,2,1, frameon=False)
            ax2 = plt.subplot(1,2,2, frameon=False)
            ax1.imshow(RGB_img)
            ax2.imshow(heatmap_img)
            plt.savefig(save_path, bbox_inches='tight')
            mask,heatmap,cam = None, None, None
        """
        return qvo_output


# Vision and Language pretrained models. e.g., FLAVA model.
class SMART_VL_Net(nn.Module):
    def __init__(self, args, VL_backbone):
        super(SMART_VL_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.num_opts = 5
        self.out_dim = args.feat_size  # the intermediate feature size.
        self.h_sz = 256
        self.feat_size = 768
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone

        if args.loss_type == "classifier" or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1

        self.processor = args.preprocess
        self.VL_backbone = VL_backbone
        self.create_puzzle_head(args)

        self.q_MLP = nn.Sequential(
            nn.Linear(self.feat_size, self.h_sz),
            nn.ReLU(),
            nn.Linear(self.h_sz, self.out_dim),
            nn.ReLU(),
        )

        self.qv_MLP = nn.Sequential(
            nn.Linear(self.feat_size, self.h_sz),
            nn.ReLU(),
            nn.Linear(self.h_sz, self.out_dim),
            nn.ReLU(),
        )

        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),  # for flava its *2.
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        if self.monolithic:
            self.qvo_fusion = nn.Sequential(nn.Linear(self.out_dim, self.max_val))
        else:
            self.create_puzzle_tail(args)

    def create_puzzle_head(self, args):
        if args.use_single_image_head:
            self.im_encoder = nn.Sequential(
                nn.Linear(self.feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
            )
        else:
            self.puzzle_ids = args.puzzle_ids
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
                    )
                )
            self.im_encoder = nn.ModuleList(im_encoder)

    def create_puzzle_tail(self, args):
        self.puzzle_ids = args.puzzle_ids
        ans_decoder = [
            nn.Sequential(nn.Linear(self.out_dim, 1))
        ]  # start with a dummy as we are 1-indexed wrt puzzle ids.
        for pid in range(1, gv.num_puzzles + 1):
            num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)] if args.loss_type == "classifier" else 1
            if int(pid) not in gv.SEQ_PUZZLES:
                ans_decoder.append(
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.ReLU(),
                        nn.Linear(self.out_dim, num_classes),
                    )
                )
            else:
                ans_decoder.append(nn.LSTM(self.out_dim, num_classes, num_layers=1, batch_first=True))
        self.ans_decoder = nn.ModuleList(ans_decoder)

    def process(self, images, text):
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            max_length=77,
            padding=True,
            return_codebook_pixels=True,
            return_image_mask=True,
        )
        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
        inputs = inputs.to(gv.device)
        return inputs

    def encode_image(self, im_feat, pids=None):
        if self.use_single_image_head:
            y = self.im_encoder(im_feat)
        else:
            y = torch.zeros(len(im_feat), im_feat.shape[1], self.out_dim).to(gv.device)
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.to(gv.device)
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[int(self.puzzle_ids[t])](im_feat[idx]))
        return y

    def encode_image_and_text(self, qv_feat):
        x = F.relu(self.qv_MLP(qv_feat))
        return x

    def encode_text(self, q_feat):
        x = F.relu(self.q_MLP(q_feat))
        return x

    def decode_image(self, im_list):
        """convert torch tensor images back to Image bcos VL FLAVA model works with images."""
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))]  # convert im
        return im_list

    def decode_text(self, text):
        tt = text.cpu()
        text = [
            " ".join([self.vocab.idx2word[int(j)] for j in tt[i][1 : torch.nonzero(tt[i])[-1]]]) for i in range(len(tt))
        ]
        return text

    def seq_decoder(self, decoder, feat):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(gv.MAX_DECODE_STEPS):
            try:
                out[k], hx = decoder(feat, hx)
            except:
                pdb.set_trace()
        return out

    def decode_individual_puzzles(self, feat, pids):
        upids = torch.unique(pids)
        out_feats = {}
        for t in range(len(upids)):
            idx = pids == upids[t]
            key = str(upids[t].item())
            if upids[t] not in gv.SEQ_PUZZLES:
                out_feats[key] = self.ans_decoder[int(key)](feat[idx])
            else:
                out_feats[key] = self.seq_decoder(self.ans_decoder[int(key)], feat[idx])
        return out_feats

    def forward(self, im, q, q_stn, puzzle_ids=None):
        im = self.decode_image(im)                  # [64, PIL]
        q_text = self.decode_text(q)                # [64, text]
        inputs = self.process(im, q_text)           # [64, Encoding]
        """
        64 Batch's
        Encoding(num_tokens=77,
        attributes=[ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing])]
            - ids: 텍스트를 구성하는 토큰 ID [word + padding]
            - tokens: 텍스트를 구성하는 토큰들의 실제 텍스트 [word + padding]
            - offsets: 텍스트를 구성하는 토큰들의 offset [offset=(i,j)]
            - attention_mask: 텍스트를 구성하는 토큰들의 Att mask [1,1,1,...,0,0,0]
            - special_tokens_mask: 텍스트를 구성하는 토큰 중 eos, pad, cls와 같은 특수 토큰 마스크 [1,0,0,...,1,1,1]
            - overflowing: 아마 텍스트 길이가 77이 넘을 때 뭐가 들어가는 리스트인 듯.
        """
        if self.train_backbone:
            outputs = self.VL_backbone(**inputs)
        else:
            with torch.no_grad():
                outputs = self.VL_backbone(**inputs)

        im_feat = outputs.image_embeddings  # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
        q_feat = outputs.text_embeddings  # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768
        #        qv_feat_mm = outputs.multimodal_embeddings # Batch size X (Number of image patches + Text Sequence Length + 3) X Hidden size => 2 X 275 x 768
        # Multimodal embeddings can be used for multimodal tasks such as VQA

        im_feat = self.encode_image(im_feat, puzzle_ids)
        q_feat = self.encode_text(q_feat)

        qv_feat = self.qv_fusion(torch.cat([im_feat.mean(1), q_feat.mean(1)], dim=1))

        if self.monolithic:
            qv_feat = qv_feat.unsqueeze(1)
            qvo_feat = self.qvo_fusion(qv_feat).squeeze()
        else:
            qvo_feat = self.decode_individual_puzzles(qv_feat, puzzle_ids)

        return qvo_feat


# Vision backbones and language backbones.
class SMART_Net(nn.Module):
    def __init__(self, args, im_backbone=None):
        super(SMART_Net, self).__init__()
        vocab_path = args.vocab_path
        if vocab_path != 'none':
            with open(vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
            vocab_len = len(self.vocab)
        else:
            vocab_len = 5862

        self.num_opts = 5
        self.out_dim = args.feat_size  #  64 #
        self.h_sz = 256  # 256 #128 #
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.loss_type = args.loss_type
        self.monolithic = args.monolithic
        self.use_single_image_head = args.use_single_image_head
        self.train_backbone = args.train_backbone
        self.word_embed = args.word_embed
        self.is_challenge = args.challenge
        
        # 내가 추가한 Arguement ============== #
        self.use_puzzle_type_classifier = args.use_puzzle_type_classifier
        self.puzzle_category_dict = gv.puzzle_category_dict

        if args.loss_type == "classifier" or args.loss_type == "puzzle_tails":
            self.max_val = gv.MAX_VAL + 1
        elif args.loss_type == "regression":
            self.max_val = 1

        # image backbones.
        if args.model_name[:6] == "resnet":
            self.im_feat_size = im_backbone.fc.weight.shape[1]
            modules = list(im_backbone.children())[:-1]
            self.im_cnn = nn.Sequential(*modules)
        elif args.model_name in ["alexnet", "vgg"]:
            im_backbone.classifier[-1] = nn.Identity()
            self.im_cnn = im_backbone
            self.im_encoder = nn.Linear(im_backbone.classifier[-3].weight.shape[1], self.out_dim)
        elif args.model_name in ["swin_t"]:
            self.im_feat_size = 768
            self.im_cnn = im_backbone
            self.im_cnn.head = nn.Identity()
        elif args.model_name in ["swin_b"]:
            self.im_feat_size = 1024
            self.im_cnn = im_backbone
            self.im_cnn.head = nn.Identity()
        elif args.model_name in ["vit"]:
            self.im_feat_size = 768
            self.im_cnn = im_backbone
            self.im_cnn.heads.head = nn.Identity()
        elif args.model_name in ["mae"]:
            self.preprocess = args.preprocess
            self.im_cnn = lambda x: self.process_MAE(x)  # inputs = feature_extractor(images=image, return_tensors="pt")
            self.im_backbone = im_backbone
            self.im_feat_size = 768
        elif args.model_name in ["cross_transformer"]:  # when using a vision transformer model.
            from vit_pytorch.crossformer import CrossFormer

            self.im_cnn = CrossFormer(
                num_classes=256,  # number of output classes
                dim=(64, 128, 256, 512),  # dimension at each stage
                depth=(2, 2, 8, 2),  # depth of transformer at each stage
                global_window_size=(8, 4, 2, 1),  # global window sizes at each stage
                local_window_size=7,  # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
            )

            self.im_feat_size = 256
        else:
            raise "unknown model_name %s" % (args.model_name)

        self.create_puzzle_head(args)

        # language backbones
        if self.use_clip_text:
            self.q_encoder, _ = clip.load("ViT-B/32", device=gv.device)
            self.clip_dim = 512
            self.q_MLP = nn.Sequential(
                nn.Linear(self.clip_dim, self.h_sz), nn.ReLU(), nn.Linear(self.h_sz, self.out_dim)
            )
        else:
            if args.word_embed == "standard":
                self.q_emb = nn.Embedding(len(self.vocab), self.h_sz, max_norm=1)
                self.q_lstm = nn.LSTM(self.h_sz, self.h_sz, num_layers=2, batch_first=True, bidirectional=True)
            else:
                word_dim = gv.word_dim
                self.q_emb = nn.Identity()
                self.q_lstm = nn.GRU(word_dim, self.h_sz, num_layers=1, batch_first=True, bidirectional=True)
            self.q_MLP = nn.Linear(self.h_sz * 2, self.out_dim)

        self.o_encoder = nn.Sequential(
            nn.Embedding(vocab_len, self.out_dim, max_norm=1),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        if self.monolithic:
            self.qvo_fusion = nn.Sequential(nn.Linear(self.out_dim, self.max_val))
        else:
            self.create_puzzle_tail(args)

    def process_MAE(self, x):
        x = self.decode_image(x)  # get from tensor to PIL images
        inputs = self.preprocess(images=x, return_tensors="pt").to(gv.device)
        outputs = self.im_backbone(**inputs)
        return outputs.last_hidden_state.mean(1)

    def create_puzzle_head(self, args):
        if args.use_single_image_head:
            self.im_encoder = nn.Sequential(
                nn.Linear(self.im_feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
            )
        elif args.use_puzzle_type_classifier:
            self.puzzle_ids = args.puzzle_ids   # [101]
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzle_category+1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.im_feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
                    )
            )
            self.im_encoder = nn.ModuleList(im_encoder)
            pz_type_init_dim=2048
            self.puzzle_type_clf = nn.Sequential(
                    nn.Linear(pz_type_init_dim, pz_type_init_dim//2),
                    nn.ReLU(),
                    nn.Linear(pz_type_init_dim//2, pz_type_init_dim//4),
                    nn.ReLU(),
                    nn.Linear(pz_type_init_dim//4, pz_type_init_dim//8),
                    nn.ReLU(),
                    nn.Linear(pz_type_init_dim//8, gv.num_puzzle_category),
                    nn.Softmax()
                    )
        else:
            self.puzzle_ids = args.puzzle_ids   # [101]
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.im_feat_size, self.out_dim), nn.ReLU(), nn.Linear(self.out_dim, self.out_dim)
                    )
            )
            self.im_encoder = nn.ModuleList(im_encoder)

    def create_puzzle_tail(self, args):
        self.puzzle_ids = args.puzzle_ids
        ans_decoder = [
            nn.Sequential(nn.Linear(self.out_dim, 1))
        ]  # start with a dummy as we are 1-indexed wrt puzzle ids.
        
        if args.use_puzzle_type_classifier:
            num_classes = self.max_val if args.loss_type == "classifier" else 1
            for cid in range(1, gv.num_puzzle_category + 1):  # self.puzzle_category_dict:
                ans_decoder.append(
                        nn.Sequential(
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, num_classes),
                        )
                    )
            ans_decoder.append(nn.LSTM(self.out_dim, num_classes, num_layers=1, batch_first=True))
                
        else:
            for pid in range(1, gv.num_puzzles + 1):  # self.puzzle_ids:
                num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)] if args.loss_type == "classifier" else 1
                if int(pid) not in gv.SEQ_PUZZLES:
                    ans_decoder.append(
                        nn.Sequential(
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, num_classes),
                        )
                    )
                else:
                    ans_decoder.append(nn.LSTM(self.out_dim, num_classes, num_layers=1, batch_first=True))
        self.ans_decoder = nn.ModuleList(ans_decoder)

    def decode_image(self, im_list):
        """convert torch tensor images back to Image bcos VL FLAVA model works with images."""
        #        im_list = (im_list +1)/2. # this is in range [0, 1].
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [Image.fromarray(im_list[ii]) for ii in range(len(im_list))]  # convert im
        return im_list

    def save_grad_hook(self):
        self.vis_grad = None

        def bwd_hook(module, in_grad, out_grad):
            self.vis_grad = out_grad

        return bwd_hook

    def save_fwd_hook(self):
        self.vis_conv = None

        def fwd_hook(__, _, output):
            self.vis_conv = output

        return fwd_hook

    def encode_image(self, im, pids=None):
        if self.train_backbone:
            x = self.im_cnn(im).squeeze()
        else:
            with torch.no_grad():
                x = self.im_cnn(im).squeeze()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.use_single_image_head:
            y = self.im_encoder(x)
        elif self.use_puzzle_type_classifier:
            self.pred_pz_logit = self.puzzle_type_clf(x)
            self.pred_pz_type = torch.argmax(self.pred_pz_logit, dim=1)
            y = torch.zeros(len(im), self.out_dim).to(gv.device)
            for t in range(len(self.puzzle_category_dict)):
                idx = (self.pred_pz_type == t)
                idx = idx.to(gv.device)
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[t+1](x[idx]))       
        else:
            y = torch.zeros(len(im), self.out_dim).to(gv.device)
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.to(gv.device)
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[int(self.puzzle_ids[t])](x[idx]))

        return y

    def decode_text(self, text):
        get_range = lambda x: range(1, x) if x < 70 else range(x - 70 + 4, x)
        tt = text.cpu()
        text = [
            " ".join([self.vocab.idx2word[int(j)] for j in tt[i][get_range(torch.nonzero(tt[i])[-1])]])
            for i in range(len(tt))
        ]
        return text

    def encode_text(self, text):
        if self.word_embed == "standard":
            x = self.q_emb(text)
            x, (h, _) = self.q_lstm(x.float())
            x = F.relu(self.q_MLP(x.mean(1)))
        elif self.word_embed == "gpt" or "bert" or "glove":
            if not self.is_challenge: # for challenge, we already encode in the dataloader.
                text = self.decode_text(text)
                q_enc = torch.zeros(len(text), gv.max_qlen, gv.word_dim).to(gv.device)
                for ii, tt in enumerate(text):
                    q_feat = gv.word_embed(tt)
                    q_enc[ii, : min(gv.max_qlen, len(q_feat)), :] = q_feat
                x, (h, _) = self.q_lstm(q_enc.float())
                x = F.relu(self.q_MLP(x.mean(1)))
            else:
                q_enc = text
                x, (h, _) = self.q_lstm(q_enc.float())
                x = F.relu(self.q_MLP(x.mean(1)))
        else:
            x = gv.word_embed(text)

        return x

    def seq_decoder(self, decoder, feat):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(gv.MAX_DECODE_STEPS):
            try:
                out[k], hx = decoder(feat, hx)
            except:
                pdb.set_trace()
        return out

    def decode_individual_category(self, feat, pids):
        pids = pids.to(gv.device)
        ucids = torch.unique(self.pred_pz_type)
        out_feats = {}
        seq_key = int(gv.num_puzzle_category+1)
        for t in ucids:
            idx = (self.pred_pz_type == t)
            key = int(t+1)
            
            # if pids[idx][0] not in gv.SEQ_PUZZLES:
            #     out_feats[key] = self.ans_decoder[key](feat[idx])
            # else:
            #     out_feats[seq_key] = self.seq_decoder(self.ans_decoder[seq_key], feat[idx])
            out_feats[key] = self.ans_decoder[key](feat[idx])
        return out_feats

    def decode_individual_puzzles(self, feat, pids):
        upids = torch.unique(pids)
        out_feats = {}
        for t in range(len(upids)):
            idx = pids == upids[t]
            key = str(upids[t].item())
            if upids[t] not in gv.SEQ_PUZZLES:
                out_feats[key] = self.ans_decoder[int(key)](feat[idx])
            else:
                out_feats[key] = self.seq_decoder(self.ans_decoder[int(key)], feat[idx])
        return out_feats

    def forward(self, im, q, q_stn, puzzle_ids=None):
        im_feat = self.encode_image(im, puzzle_ids)                     # [64, 128]
        q_feat = self.encode_text(q)                                    # [64, 128]
        qv_feat = self.qv_fusion(torch.cat([im_feat, q_feat], dim=1))   # [64, 128]
        if self.monolithic:
            qv_feat = qv_feat.unsqueeze(1)                              # [64, 1, 128]
            qvo_feat = self.qvo_fusion(qv_feat).squeeze()               # [64, 257]
            return qvo_feat
        elif self.use_puzzle_type_classifier:
            qvo_feat = self.decode_individual_category(qv_feat, puzzle_ids)
            return qvo_feat, self.pred_pz_logit
        else:
            qvo_feat = self.decode_individual_puzzles(qv_feat, puzzle_ids)
            return qvo_feat


def load_pretrained_models(args, model_name, model=None):

    if args.test and model is not None:
        model_path = os.path.join(args.location, "ckpt_%s_%s_%s.pth" % (args.model_name, args.word_embed, args.seed))
        print("test: loading checkpoint %s ..." % (model_path))
        if ("IBLIP" in args.model_name) or (args.model_name == 'Qwen'):
            model = torch.load(model_path, map_location=torch.device(gv.device))
            return model
        else:
            checkpoint = torch.load(model_path, map_location=torch.device(gv.device))
            model.load_state_dict(checkpoint["net"], strict=True)
            return model
    
    if args.challenge and model is not None:
        model_path = args.pretrained_model_path
        print("challenge: loading checkpoint %s ..." % (model_path))
        if ("IBLIP" in args.model_name) or (args.model_name == 'Qwen'):
            model = torch.load(model_path, map_location=torch.device(gv.device))
            return model
        else:
            checkpoint = torch.load(model_path, map_location=torch.device(gv.device))
            model.load_state_dict(checkpoint["net"], strict=True)
            return model

    preprocess = None
    if args.model_name in ["resnet18"]:
        model = models.__dict__[args.model_name](pretrained=True)
    elif args.model_name in ["resnet50"]:  # use_resnet:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        preprocess = weights.transforms()
        model = resnet50(weights=None)
        pretrained_weights = torch.load("./checkpoints/RAW/resnet50-11ad3fa6.pth")
        model.load_state_dict(pretrained_weights, strict=True)
    elif args.model_name == "swin_t":  # use_vit:
        from torchvision.models import Swin_T_Weights, swin_t

        weights = Swin_T_Weights.IMAGENET1K_V1
        model = swin_t(weights=weights)
        preprocess = weights.transforms()
    elif args.model_name == "swin_b":  # use_vit:
        from torchvision.models import Swin_B_Weights, swin_b

        weights = Swin_B_Weights.IMAGENET1K_V1
        model = swin_b(weights=weights)
        preprocess = weights.transforms()
    elif args.model_name == "vit":
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1  # ViT_B_16_Weights.DEFAULT #
        model = vit_b_16(weights=weights)
        preprocess = weights.transforms()
    elif args.model_name == "flava":
        from transformers import FlavaForPreTraining, FlavaProcessor
        
        model = FlavaForPreTraining.from_pretrained("facebook/flava-full").eval()
        preprocess = FlavaProcessor.from_pretrained("facebook/flava-full")
    elif args.model_name == "clip":
        import clip
        model, preprocess = clip.load("ViT-B/32", device=gv.device)
    elif args.model_name == "mae":
        from transformers import AutoFeatureExtractor, ViTMAEModel

        feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        preprocess = feature_extractor
    elif args.model_name == 'IBLIP':
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        if args.test:
            model = None
        else:
            if args.LLM_type == "vicuna7b":
                model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to(gv.device)
            elif args.LLM_type == "t5_xl":
                model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl", ignore_mismatched_sizes=True).to(gv.device)
            elif args.LLM_type == "t5_xxl":
                model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xxl", ignore_mismatched_sizes=True).to(gv.device)
            if args.use_bf16:
                model.to(torch.bfloat16)
        preprocess = None
    elif args.model_name == 'IBLIP_GEN':
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        if args.test:
            model = None
        else:    
            model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").to(gv.device)
        preprocess = None
    
    elif args.model_name == 'Qwen':
        if args.test:
            model = None
        else:
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="cuda", trust_remote_code=True).eval()
            if args.use_bf16:
                model.to(torch.bfloat16)
        preprocess = None
        
    else:
        print("model name is %s: not loading pre-trained model." % (args.model_name))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location=torch.device(gv.device))

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith("module.encoder") and not k.startswith("module.encoder.fc"):
                    # remove prefix
                    state_dict[k[len("module.encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    return model, preprocess
