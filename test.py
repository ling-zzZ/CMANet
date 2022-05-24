# -*- coding: utf-8 -*-
"""
Created on Mon May 23 01:21:08 2022

@author: ling-zzZ
"""

import os
import sys
from models.model import CMANet
import cv2
from PIL import Image
import torch
import argparse
import numpy as np


def getresults(left, func):

    in_h, in_w = left.shape[:2]

    left_img = cv2.resize(left, (outw, outh), interpolation=cv2.INTER_CUBIC)
    left_img = cv2.cvtColor(left_img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    left_img = (left_img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    left_img = left_img.transpose(2, 0, 1).astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_left = torch.from_numpy(left_img).to(device).unsqueeze(0)

    with torch.no_grad():
        output = func(batch_left)

    pred_depth = output.squeeze().cpu().data.numpy()
    if np.min(pred_depth) < 0:
        pred_depth = (pred_depth -np.min(pred_depth))/np.max(pred_depth) * 65535.0
    else:
        pred_depth = pred_depth /np.max(pred_depth) * 65535.0
    pred_depth = cv2.resize(pred_depth, (in_w, in_h), interpolation=cv2.INTER_LINEAR)

    res = pred_depth.astype(np.uint16)

    return res

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters share common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_ckpt(model, ckpt_name):

    checkpoint = torch.load(ckpt_name)

    pretrained_dict = remove_prefix(checkpoint, 'module.')
    model.load_state_dict(pretrained_dict)

def main():
    parser = argparse.ArgumentParser(description="DPT")

    parser.add_argument('--test_data', default='./img/img_03170_c0_1303398750047535us.jpg')
    parser.add_argument('--model_set', default='./cmanet_season_weight.pkl')
    parser.add_argument('--size', default='576x768')
    parser.add_argument('--num', default='10000', type=int)
    parser.add_argument('--name', default='', help="using for result dir")

    args = parser.parse_args()
    global outh
    global outw
    outh, outw = [int(e) for e in args.size.split('x')]
    
    net = CMANet(is_train=False)
    load_ckpt(net, args.model_set)
    img_raw = cv2.imread(args.test_data, cv2.IMREAD_COLOR)
    depth = getresults(img_raw, net)
#    depth = Image.fromarray(depth)
#    depth.save('test_img_depth.png')     
    cv2.imwrite('./test_img_depth.png', depth)

if __name__ == "__main__":
    main()