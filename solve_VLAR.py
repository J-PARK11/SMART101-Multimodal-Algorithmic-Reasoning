#!/usr/bin/env python3
"""
    VLAR Challenge: Submission Code Demo.
    ************************************
    
    This is a starter code demonstrating how to format your pre-trained model for evaluation. 
    It shows where to expect to read the test puzzle images from, and how to produce 
    the output, which can be evaluated using our evaluation code. 
    
    Please see the predict_on_challenge_data() function below for details. Formally, the code shows four steps:
        1) To read the puzzles (see data_loader.py, SMART_Challenge_Data class)
        2) Get/load the prediction/solution model: get_SMART_solver_model())
        3) Run the prediction model on the test puzzles and collect responses: make_predictions()
        4) Collect the responses in a json file for evaluation: make_response_json()
    
    For this demo, we provide a pretrained ResNet-50 + BERT pre-trained model traiend
    on the SMART-101 dataset in the puzzle_split mode. This model is provided in ./checkpoints/ckpt_resnet50_bert_212.pth
    
    See scripts.sh file for the commandlines to train the model on SMART-101 dataset and how to run the model on the VLAR challenge
    val and test datasets. 
    
    Specifically, note that the VLAR-val.json and VLAR-test.json files containing the VLAR challenge puzzles
    are assumed to be kept in /dataset/ folder, and a method should write the responses to /submission/submission.json
    as described in make_predictions() below. 
    
    Note
    ----
    In this demo, we do not use the answer candidate options within the model. However, 
    a user may chose to have additional inputs to the model for taking in the options.
    
    For questions: contact the VLAR organizers at vlariccv23@googlegroups.com
"""
import torch
import numpy as np
import json
import os
import time
import net
import data_loader as dl
import globvars as gv

def get_SMART_solver_model(args, model_path):
    """ A dummy function that needs to be implemented to get a prediction model """
    if args.challenge and (args.phase == 'val' or args.phase == 'test'):
        model = net.load_pretrained_models(args, args.model_name, model=True)
    else:
        raise Exception("SMART solver needs to be in the Challenge mode!")
    
    return model
    
def make_predictions(challenge_loader, model):
    responses = {}
    with torch.no_grad():
        for i, (im, q, q_stn, opts, pid) in enumerate(challenge_loader):
            im = im.to(gv.device)
            q = q.to(gv.device)
            out = model(im, q, q_stn, puzzle_ids=pid)
            pred_max = out.argmax().cpu().numpy()
            try:
                selected_opt = np.abs(np.array([int(opt[0]) for opt in opts])-pred_max).argmin() # answers are digits.
            except:
                selected_opt = np.abs(np.array([ord(opt[0]) for opt in opts])-pred_max).argmin() # result is a letter
            responses[str(pid[0].item())] = chr(ord('A') + selected_opt)
    return responses

def make_response_json(challenge_loader, responses):
    puz_cnt = 0
    if not os.path.exists(gv.VLAR_CHALLENGE_submission_root):
        os.mkdir(gv.VLAR_CHALLENGE_submission_root)
    with open(os.path.join(gv.VLAR_CHALLENGE_submission_root, 'submission.json'), 'w') as pred_json:
        pred_json.write('{ \"VLAR\": [') # header.
        for i, (_, _, _, _, pid) in enumerate(challenge_loader):
            puz = {'Id': str(pid[0].item()), 'Answer': responses[str(pid[0].item())]}
            if puz_cnt > 0:
                pred_json.write(',\n')
            json.dump(puz, pred_json, indent = 6)
            puz_cnt += 1
        pred_json.write(']\n}')
    
    return 0

def get_data_loader(args, split, batch_size=100, shuffle=True, num_workers=6, pin_memory=True):
    assert(split == 'challenge')
    dataset = dl.SMART_Challenge_Data(args, split)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=None,
    )
    return data_loader

def predict_on_challenge_data(args, pretrained_model_path, challenge_phase='val'):
    args.puzzles_file = 'VLAR-val.json' if challenge_phase == 'val' else 'VLAR-test.json'
        
    print('loading model ...')
    model = get_SMART_solver_model(args, pretrained_model_path) # provide the model for evaluation.
    model.eval()
    model.to(gv.device)
    
    challenge_loader = get_data_loader(args, "challenge", batch_size=1, shuffle=False, num_workers=0) 
    
    print('making predictions using the model')
    responses = make_predictions(challenge_loader, model) # call the model.forward()
    
    print('writing the model responses to file')
    make_response_json(challenge_loader, responses) # dump the model predicted answers into a json file for evaluation.
    
    print('done!', flush=True)
    # NOTE: Do no remove this without this evaluation script will not run.
    time.sleep(320) # sleep for 5 minutes to allow the evaluation script to run.
