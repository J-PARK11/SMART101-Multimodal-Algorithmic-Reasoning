#!/usr/bin/env python3
# Copyright (c) 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: MIT
#
import torch
import torch.nn as nn
import torch.nn.functional as F

import globvars as gv

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        
        # inputs: 모델의 로짓 출력, targets: 실제 레이블
        CE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-CE_loss)  # p_t를 계산
        F_loss = self.alpha * (1-pt)**self.gamma * CE_loss
        return F_loss.mean()

class Criterion(nn.Module):
    def __init__(self, args):
        super(Criterion, self).__init__()
        self.monolithic = args.monolithic  # just one classifier
        self.loss_type = args.loss_type
        self.use_puzzle_type_classifier = args.use_puzzle_type_classifier
        self.model_name = args.model_name
        
        if args.loss_type == "classifier":
            self.criterion = nn.CrossEntropyLoss()
        elif args.loss_type == "regression":
            self.criterion = nn.L1Loss()
        elif args.loss_type == "Both":
            self.criterion1 = nn.CrossEntropyLoss()
            self.criterion2 = nn.L1Loss()
        elif args.loss_type == "focal":
            self.criterion = FocalLoss()
        elif args.loss_type == "focal+MSE":
            self.criterion1 = FocalLoss()
            self.criterion2 = nn.L1Loss()
            
        if self.use_puzzle_type_classifier:
            self.cat_loss = nn.CrossEntropyLoss()

    def compute_loss(self, a, b, pids, cat_label=None, cat_pred_logit=None):
        if (self.loss_type == 'Both') or (self.loss_type == 'focal+MSE'):
            if self.use_puzzle_type_classifier:
                a_max = a.argmax(dim=1)
                alpha, beta, gamma = 0.75, 0.5, 0.5
                loss1 = self.criterion1(a, b[:, 0])
                loss2 = self.criterion2(a_max.type(torch.float32), b[:, 0])
                loss_cat = self.cat_loss(cat_pred_logit, torch.tensor(cat_label).to(gv.device))
                loss = alpha*loss1 + beta*loss2 + gamma*loss_cat
                return loss, loss1, loss2, loss_cat
            else:
                a_max = a.argmax(dim=1)
                alpha, beta = 0.75, 0.25
                loss1 = self.criterion1(a, b[:, 0])
                loss2 = self.criterion2(a_max.type(torch.float32), b[:, 0])
                loss = alpha*loss1 + beta*loss2
                return loss, loss1, loss2
        
        elif self.monolithic:
            loss = self.criterion(a, b[:, 0])
        elif cat_pred_logit != None:
            cat_label = torch.tensor(cat_label).to(gv.device)
            cat_pred = torch.argmax(cat_pred_logit,1)
            loss, cat_loss = 0, 0
            for key in a.keys():
                idx = (cat_pred == (key-1))
                if key != 9:
                    loss += self.criterion(
                        a[key], b[idx, 0])
                    
                # else:
                #     seq_loss = 0
                #     for i in range(len(a[key])):
                #         seq_loss += self.criterion(a[key][i], b[idx, i])  # .long()
                #     loss += seq_loss
            cat_loss = self.cat_loss(cat_pred_logit, cat_label)   
            loss += cat_loss
            loss = loss / len(b)
                
        else:
            loss = 0
            for key in a.keys():
                idx = pids == int(key)
                if int(key) not in gv.SEQ_PUZZLES:
                    loss += self.criterion(
                        a[key], b[idx, 0]
                    )  # risky if idx and key entries are not matched. but then we will encouter an exception.
                else:
                    seq_loss = 0
                    for i in range(len(a[key])):
                        seq_loss += self.criterion(a[key][i], b[idx, i])  # .long()
                    seq_loss /= len(a[key])
                    loss += seq_loss
            loss = loss / len(a.keys())
        return loss

    def forward(self, a, b, pids=None, cat_label=None, cat_pred=None):
        if (self.loss_type == "classifier") or (self.loss_type == "focal"):
            loss = self.compute_loss(a, b.long(), pids, cat_label, cat_pred)
        elif self.loss_type == "regression":
            loss = self.compute_loss(a, b.float(), pids, cat_label, cat_pred)
        elif (self.loss_type == "Both") or (self.loss_type == "focal+MSE"):
            if self.use_puzzle_type_classifier:
                loss, loss1, loss2, loss_cat = self.compute_loss(a, b.long(), pids, cat_label, cat_pred)
                return loss, loss1, loss2, loss_cat
            else:
                loss, loss1, loss2 = self.compute_loss(a, b.long(), pids, cat_label, cat_pred)
                return loss, loss1, loss2
        else:
            raise "Unknown loss type: use classifer/regression"
        return loss
