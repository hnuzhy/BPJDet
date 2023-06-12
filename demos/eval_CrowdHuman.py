
'''
CrowdHuman: A Benchmark for Detecting Human in a Crowd

https://www.crowdhuman.org/

https://arxiv.org/abs/1805.00123
'''

#!/usr/bin/env python
# coding: utf-8
import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import os
import json
import torch
import argparse
import yaml
import cv2
import math
import os.path as osp
import numpy as np
import pandas as pd

import brambox
from tqdm import tqdm
from collections import defaultdict
from brambox.stat._matchboxes import match_det, match_anno
from brambox.stat import coordinates, mr_fppi, ap, pr, threshold, fscore, peak, lamr

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import post_process_batch

import xml.etree.ElementTree as ET

def safe_div(x,y):
    if y == 0:
        return 0
    return x / y

def get_moda(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()

    dets_per_frame = anno.groupby('image').filter(lambda x: any(x['ignore'] == 0))
    dets_per_frame = dets_per_frame.groupby('image').size().to_dict()
    # Other param for finding matched anno
    crit = coordinates.pdollar if ignore else coordinates.iou
    label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
    matched_dets = match_det(det, anno, threshold, criteria=crit,
                            class_label=label, ignore=2 if ignore else 0)
    fp_per_im = matched_dets[matched_dets.fp==True].groupby('image').size().to_dict()
    tp_per_im = matched_dets[matched_dets.tp==True].groupby('image').size().to_dict()
    valid_anno = anno[anno.ignore == False].groupby('image').size().to_dict()
    assert valid_anno.keys() == tp_per_im.keys()

    moda_ = []
    for k, _ in valid_anno.items():
        n_gt = valid_anno[k]
        miss = n_gt-tp_per_im[k]
        fp = fp_per_im[k]
        moda_.append(safe_div((miss+fp), n_gt))
    return 1 - np.mean(moda_)

def get_modp(det, anno, threshold=0.2, ignore=None):
    if ignore is None:
        ignore = anno.ignore.any()
    # Compute TP/FP
    if not {'tp', 'fp'}.issubset(det.columns):
        crit = coordinates.pdollar if ignore else coordinates.iou
        label = len({*det.class_label.unique(), *anno.class_label.unique()}) > 1
        det = match_anno(det, anno, threshold, criteria=crit, class_label=label, ignore=2 if ignore else 0)
    elif not det.confidence.is_monotonic_decreasing:
        det = det.sort_values('confidence', ascending=False)
    modp = det.groupby('image')['criteria'].mean().mean()
    return modp

def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation script of BPJDet on CroHD train-set')  # val-set not released
    parser.add_argument('-p', '--root-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/JointBP_CrowdHuman_head.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.6, help='Matching IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    
    args = parser.parse_args()
    
    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    assert data['dataset'] == "CrowdHuman", "we now only consider models trained on CrowdHuman!!!"
    
    data['conf_thres_part'] = args.conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = args.iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = args.match_iou  # whether a body-part in matched with one body bbox
    
    '''pre-downloading model weight and CroHD train-set'''
    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    
    # Brambox eval related
    pd_dict_v1 = defaultdict(list)  # using body_association
    pd_dict_v2 = defaultdict(list)  # not using body_association
    gt_dict = defaultdict(list)

    # root_path: /datasdc/zhouhuayi/dataset/CrowdHuman/
    image_path = os.path.join(args.root_path, "images/val/")
    json_path = os.path.join(args.root_path, "JointBodyPart/crowdhuman_coco_val_head.json")
    
    
    '''update gt_dict'''
    anno_dict = json.load(open(json_path, "r"))
    imgs_dict_list = anno_dict['images']
    annos_dict_list = anno_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(annos_dict_list)
    for imgs_dict in tqdm(imgs_dict_list):
        img_name = imgs_dict["file_name"]
        image_id = str(imgs_dict['id'])
        anno_BFJDet_list = images_labels_dict[image_id]
        for anno_BFJDet_instance in anno_BFJDet_list:
            [xmin, ymin, head_w, head_h] = anno_BFJDet_instance["bbox"]
            gt_dict['image'].append(img_name)
            gt_dict['class_label'].append('head')
            gt_dict['id'].append(0)
            gt_dict['x_top_left'].append(float(xmin))
            gt_dict['y_top_left'].append(float(ymin))
            gt_dict['width'].append(float(head_w))
            gt_dict['height'].append(float(head_h))
            gt_dict['ignore'].append(False)

    '''load images in image_path'''
    dataset = LoadImages(image_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)

    print("[start]\t", image_path, len(dataset))

    for index in tqdm(range(len(dataset))):
        
        (single_path, img, im0, _) = next(dataset_iter)
        img_name = osp.split(single_path)[-1]
        # img_name = str(sub_id) + "_" + osp.split(single_path)[-1]  # to distinguish sub_folder
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model(img, augment=True, scales=args.scales)[0]
        body_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=[0], num_offsets=data['num_offsets'])
        part_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=list(range(1, 1 + data['num_offsets']//2)), num_offsets=data['num_offsets'])
        
        # Post-processing of body and part detections
        bboxes, points, scores, imgids, parts_dict, _ = post_process_batch(
            data, img, [], [[im0.shape[:2]]], body_dets, part_dets)
        
        '''update pd_dict_v1 and pd_dict_v2'''
        # using body_association
        for point in points:
            f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
            if f_score == 0:
                continue  # this body bbox has no matched head bbox
            [x1, y1, x2, y2] = f_bbox
            pd_dict_v1['image'].append(img_name)
            pd_dict_v1['class_label'].append('head')
            pd_dict_v1['id'].append(0)
            pd_dict_v1['x_top_left'].append(x1)
            pd_dict_v1['y_top_left'].append(y1)
            pd_dict_v1['width'].append(x2 - x1)
            pd_dict_v1['height'].append(y2 - y1)
            pd_dict_v1['confidence'].append(f_score)
        if img_name not in pd_dict_v1['image']:  # no objects are detected in this image
            pd_dict_v1['image'].append(img_name)
            pd_dict_v1['class_label'].append('head')
            pd_dict_v1['id'].append(0)
            pd_dict_v1['x_top_left'].append(0)
            pd_dict_v1['y_top_left'].append(0)
            pd_dict_v1['width'].append(0)
            pd_dict_v1['height'].append(0)
            pd_dict_v1['confidence'].append(0)
            
        # not using body_association
        if len(parts_dict) != 0:
            key_img_id = list(parts_dict.keys())[0]  # batch size is 1
            head_bboxes = parts_dict[key_img_id]
            for head_bbox in head_bboxes:
                [x1, y1, x2, y2, conf, cls] = head_bbox
                pd_dict_v2['image'].append(img_name)
                pd_dict_v2['class_label'].append('head')
                pd_dict_v2['id'].append(0)
                pd_dict_v2['x_top_left'].append(float(x1))
                pd_dict_v2['y_top_left'].append(float(y1))
                pd_dict_v2['width'].append(float(x2) - float(x1))
                pd_dict_v2['height'].append(float(y2) - float(y1))
                pd_dict_v2['confidence'].append(float(conf))
        if img_name not in pd_dict_v2['image']:  # no objects are detected in this image
            pd_dict_v2['image'].append(img_name)
            pd_dict_v2['class_label'].append('head')
            pd_dict_v2['id'].append(0)
            pd_dict_v2['x_top_left'].append(0)
            pd_dict_v2['y_top_left'].append(0)
            pd_dict_v2['width'].append(0)
            pd_dict_v2['height'].append(0)
            pd_dict_v2['confidence'].append(0)
            
    print("[end]\t", image_path, len(dataset))

    print("bbox number in gt_dict:", len(gt_dict['image']))
    print("bbox number in pd_dict_v1 (using body_association):", len(pd_dict_v1['image']))
    print("bbox number in pd_dict_v2 (not using body_association):", len(pd_dict_v2['image']))
    
    gt_df = pd.DataFrame(gt_dict)
    gt_df['image'] = gt_df['image'].astype('category')
    
    # gather the stats from all processes (v1)
    pd_df = pd.DataFrame(pd_dict_v1)
    pd_df['image'] = pd_df['image'].astype('category')
    pr_ = pr(pd_df, gt_df, ignore=True)
    ap_ = ap(pr_)
    f1_ = fscore(pr_)
    f1_ = f1_.fillna(0)
    threshold_ = peak(f1_)
    moda = get_moda(pd_df, gt_df, threshold=0.2, ignore=True)
    modp = get_modp(pd_df, gt_df, threshold=0.2, ignore=True)

    result_dict_v1 = {'p': round(pr_['precision'].values[-1], 5), 'r': round(pr_['recall'].values[-1], 5), 
        'f1': round(threshold_.f1, 5), 'moda': round(moda, 5), 'modp': round(modp, 5), 'AP': round(ap_, 5)}
    print(result_dict_v1)


    # gather the stats from all processes (v2)
    pd_df = pd.DataFrame(pd_dict_v2)
    pd_df['image'] = pd_df['image'].astype('category')
    pr_ = pr(pd_df, gt_df, ignore=True)
    ap_ = ap(pr_)
    f1_ = fscore(pr_)
    f1_ = f1_.fillna(0)
    threshold_ = peak(f1_)
    moda = get_moda(pd_df, gt_df, threshold=0.2, ignore=True)
    modp = get_modp(pd_df, gt_df, threshold=0.2, ignore=True)

    result_dict_v2 = {'p': round(pr_['precision'].values[-1], 5), 'r': round(pr_['recall'].values[-1], 5), 
        'f1': round(threshold_.f1, 5), 'moda': round(moda, 5), 'modp': round(modp, 5), 'AP': round(ap_, 5)}
    print(result_dict_v2)
            