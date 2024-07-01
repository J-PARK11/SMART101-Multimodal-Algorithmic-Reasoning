import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def viz_attention_map(im, save_root, seed, outputs):
    """_summary_
        Return Prediction Result in json file
    Args:
        im (tensor): Tensor including Image = (B, image)
        save_root (str): the root of directory for saving visualization
        seed (int): seed number
        outputs (tensor): outputs of InstructBLIP last hidden states

    Returns:
        None
    """
    tg_idx=0
    raw_img = im[tg_idx]
    attmap_save_root = os.path.join(save_root, 'results', str(seed), 'Attmap')
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
    return

def viz_logits(save_root, img_name, pred, o, av, batch_idx):
    """_summary_
        Save Prediction Logits=Probability by png file
    Args:
        save_root (str): the root of directory for saving visualization
        img_name (str): the name of the image
        pred (tensor): the logits of prediction head = (B, 257=class)
        o (tensor): tensor of options of question = (B, str)
        av (tensor): tensor of GT Value = (B, int)
        batch_idx (int): the index of batch to visualize

    Returns:
        None
    """
    logit_output_path= os.path.join(save_root, 'Logits',f'{img_name}_Logit.png')
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.suptitle(f'Pred Logit: {img_name}\nOptions: {o[batch_idx]},  GT_value: {av[batch_idx]}')
    ax1 = plt.subplot(1,2,1, frameon=False)
    ax2 = plt.subplot(1,2,2, frameon=False)
    ax1.bar(np.arange(pred[batch_idx].shape[0])[:10], pred[batch_idx].cpu().detach()[:10])
    ax2.bar(np.arange(pred[batch_idx].shape[0]), pred[batch_idx].cpu().detach())
    plt.savefig(logit_output_path, bbox_inches='tight')
    plt.clf()
    return None

def viz_result_json(info, pred_max, opt, log):
    """_summary_
        Return Prediction Result in json file
    Args:
        info (dict): Dictionary including Question Meta data = (B, {Pred, Opt_Result, image})
        pred_max (tensor): Actual Prediction Value = (B, int)
        opt (tensor): Boolean of Option Prediction Match = (B, Bool)
        log (txt): Load txt file by pallete for reporting log.

    Returns:
        log (txt): Return updated new txt file 
    """
    for i, info_b in enumerate(info):
        out_name = info_b['image'][:-4]
        info_b['Pred'] = int(pred_max[i])
        info_b['Opt_Result'] = int(opt[i])
        tag = info_b['image']
        del info_b['image']
        log[tag] = info_b
    return log