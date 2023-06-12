import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

import numpy as np
import torch
from tqdm import tqdm
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.augmentations import letterbox
from utils.general import check_dataset, check_file, check_img_size, \
    non_max_suppression, scale_coords, set_logging, colorstr, xyxy2xywh
from utils.torch_utils import select_device, time_sync
import tempfile
import cv2
import pickle

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from utils.bp_eval import body_part_association_evaluation


def cal_inside_iou(bigBox, smallBox):  # body_box, part_box
    # calculate small rectangle inside big box ratio, calSmallBoxInsideRatio
    [Ax0, Ay0, Ax1, Ay1] = bigBox[0:4]
    [Bx0, By0, Bx1, By1] = smallBox[0:4]
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        # return crossArea/(areaA + areaB - crossArea)
        return crossArea/areaB  # range [0, 1]
    

def post_process_batch(data, imgs, paths, shapes, body_dets, part_dets, num_states=0):

    batch_bboxes, batch_points, batch_scores, batch_imgids = [], [], [], []
    batch_parts_dict = {}
    img_indexs = []
    
    # process each image in batch
    for si, (bdet, pdet) in enumerate(zip(body_dets, part_dets)):
        nbody, npart = bdet.shape[0], pdet.shape[0]
        
        if nbody:  # one batch
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            
            # img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si
            if data['dataset'] in ["CityPersons", "CrowdHuman", "BodyHands", "HumanParts", "ContactHands"]:
                img_id = int(osp.splitext(osp.split(path)[-1])[0].split("_")[-1]) if path else si

            scores = bdet[:, 4].cpu().numpy()  # body detection score
            bboxes = scale_coords(imgs[si].shape[1:], bdet[:, :4], shape).round().cpu().numpy()
            if num_states:  
                points = scale_coords(imgs[si].shape[1:], bdet[:, -data['num_offsets']-num_states:-num_states], shape).cpu().numpy()
                points = points.reshape((nbody, -1, 2))
                states = bdet[:, -num_states:].cpu().numpy()
                states = states.reshape((nbody, -1, 4))  # hand contact states: [NC, SC, PC, OC]+[NC, SC, PC, OC]
                points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5)), states), axis=-1)  # n*c*2 --> n*c*11
            else:
                points = scale_coords(imgs[si].shape[1:], bdet[:, -data['num_offsets']:], shape).cpu().numpy()
                points = points.reshape((nbody, -1, 2))
                # points = np.concatenate((points, np.zeros((nbody, points.shape[1], 1))), axis=-1)  # n*c*2 --> n*c*3
                points = np.concatenate((points, np.zeros((nbody, points.shape[1], 5))), axis=-1)  # n*c*2 --> n*c*7
                
            batch_parts_dict[str(img_id)] = []
            if npart:
                pdet[:, :4] = scale_coords(imgs[si].shape[1:], pdet[:, :4].clone(), shape)
                pdet_slim = pdet[:, :6].cpu().numpy()
                # pdet_pts = scale_coords(imgs[si].shape[1:], pdet[:, -data['num_offsets']:].clone(), shape).cpu().numpy()
                # pdet_pts = pdet_pts.reshape((npart, -1, 2))
                if num_states:
                    pdet_states = pdet[:, -num_states:].cpu().numpy()
                    pdet_states = pdet_states.reshape((npart, -1, 4))
                
                left_pdet = []
                matched_part_ids = [-1 for i in range(points.shape[0])]  # points shape is n*c*7, add in 2022-12-09
                for id, (x1, y1, x2, y2, conf, cls) in enumerate(pdet_slim):
                    p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))  # the body-part's part bbox center point
                    part_pts = points[:, int(cls - 1)]
                    dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                    pt_match = np.argmin(dist)

                    '''association alg version 4.0'''   
                    tmp_iou = cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])  # add in 2022-12-11, body-part must inside the body
                    if conf > part_pts[pt_match][2] and tmp_iou > data['match_iou_thres']:  # add in 2022-12-09, we fetch the part bbox with highest conf
                        if num_states:
                            bsw = data['body_state_w']
                            [bs1, bs2, bs3, bs4] = part_pts[pt_match][-4:] * bsw  # [NC, SC, PC, OC] for body-hand1
                            [ps1, ps2, ps3, ps4] = pdet_states[id, 0, :] * (1 - bsw)  # [NC, SC, PC, OC] for hand1
                            [s1, s2, s3, s4] = [bs1+ps1, bs2+ps2, bs3+ps3, bs4+ps4]
                            part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2, s1, s2, s3, s4]
                        else:
                            part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        matched_part_ids[pt_match] = id  # only for dataset BodyHands without left/right label of hands

                    '''association alg version 3.0'''  
                    # part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                    # if conf > part_pts[pt_match][2]:  # 2022-12-09, we feteh the part bbox with highest conf
                        # part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        # matched_part_ids[pt_match] = id
  
                    '''association alg version 2.0'''  
                    # b_x1, b_y1, b_x2, b_y2 = bboxes[pt_match]  # matched body bbox
                    # b_xc, b_yc = pdet_pts[id, 0]  # the body-part's corresponding body bbox center point 
                    # body_part_dist = (((b_x1+b_x2)/2.0 - b_xc)**2 + ((b_y1+b_y2)/2.0 - b_yc)**2)**(0.5)
                    
                    '''association alg version 1.0'''
                    # if dist[pt_match] < data['dist_thre'] and body_part_dist < data['dist_thre']:
                        # this body-part has been matched with one body bbox center point
                        # part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                    # else:
                        # this body-part has not been matched with any body bbox center point
                        # left_pdet.append([pdet_slim[id], pdet_pts[id, 0]])  # left unmatched [x1, y1, x2, y2, conf, cls, xc, yc]
                    
                    
                    # put all detected body part bboxes into their image_dict
                    batch_parts_dict[str(img_id)].append([x1, y1, x2, y2, conf, cls])
                
                
                if data['dataset'] in ["BodyHands", "ContactHands"]:  # They have no left/right anno, we have to update points[:, 1] 
                    for id, (x1, y1, x2, y2, conf, _) in enumerate(pdet_slim):
                        if id in matched_part_ids:
                            continue  # this part id has been matched into the hand1 type, here matching for hand2 type
                        p_xc, p_yc = np.mean((x1, x2)), np.mean((y1, y2))  # the body-part's part bbox center point
                        part_pts = points[:, 1]  # For dataset BodyHands, we have two hand parts
                        dist = np.linalg.norm(part_pts[:, :2] - np.array([[p_xc, p_yc]]), axis=-1)
                        pt_match = np.argmin(dist)
                        
                        '''association alg version 3.0'''
                        # part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        # if conf > part_pts[pt_match][2]:  # improve in 2022-12-09, we feteh the part bbox with highest conf
                            # part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]
                        
                        '''association alg version 4.0'''
                        tmp_iou = cal_inside_iou(bboxes[pt_match], [x1, y1, x2, y2])  # add in 2022-12-11, body-part must inside the body
                        if conf > part_pts[pt_match][2] and tmp_iou > data['match_iou_thres']:  # add in 2022-12-09, we fetch the part bbox with highest conf
                            if num_states:
                                bsw = data['body_state_w']
                                [bs1, bs2, bs3, bs4] = part_pts[pt_match][-4:] * bsw  # [NC, SC, PC, OC] for body-hand2
                                [ps1, ps2, ps3, ps4] = pdet_states[id, 1, :] * (1 - bsw)  # [NC, SC, PC, OC] for hand2
                                [s1, s2, s3, s4] = [bs1+ps1, bs2+ps2, bs3+ps3, bs4+ps4]
                                part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2, s1, s2, s3, s4]
                            else:
                                part_pts[pt_match] = [p_xc, p_yc, conf, x1, y1, x2, y2]  # update points[:, int(cls - 1), 7]

                    
            batch_bboxes.extend(bboxes)
            batch_points.extend(points)
            batch_scores.extend(scores)
            batch_imgids.extend([img_id] * len(scores))
            
            img_indexs.append(si)
        # else:
            # print("This image has no object detected!")
        
    return batch_bboxes, batch_points, batch_scores, batch_imgids, batch_parts_dict, img_indexs
    
            
@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=16,  # batch size
        imgsz=1280,  # inference size (pixels)
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        conf_thres=0.01,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        scales=[1],
        flips=[None],
        rect=False,
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        compute_loss=None,
        pad=0,
        json_name='',
        ):

    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Data
        data = check_dataset(data)  # check
        
    if data['dataset'] == "CityPersons":  
        # data['dist_thre'] = 200  # the largest dist threshold for matching, large than it will not be replaced
        data['conf_thres'] = 0.01  # the larger conf threshold for filtering body detection proposals
        data['iou_thres'] = 0.6  # the smaller iou threshold for filtering body detection proposals
        data['conf_thres_part'] = 0.02  # the larger conf threshold for filtering body-part detection proposals
        data['iou_thres_part'] = 0.3  # the smaller iou threshold for filtering body-part detection proposals
        
    if data['dataset'] == "CrowdHuman" or data['dataset'] == "BodyHands" \
        or data['dataset'] == "HumanParts" or data['dataset'] == "ContactHands":
        # data['dist_thre'] = 100
        data['conf_thres'] = 0.05  # CrowdHuman, BodyHands and HumanParts have more dense instance labels
        data['iou_thres'] = 0.6
        data['conf_thres_part'] = 0.1  # CrowdHuman, BodyHands and HumanParts have more dense instance labels
        data['iou_thres_part'] = 0.3
        
    data['match_iou_thres'] = 0.6  # whether a body-part in matched with one body bbox
    data['body_state_w'] = 0.4  # human body contact state weight, which is for ContactHands
    
    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
        
    # Configure
    model.eval()
    nc = int(data['nc'])  # number of classes
    num_offsets = int(data['num_offsets'])
    if 'num_states' in data:
        num_states = int(data['num_states'])
    else:
        num_states = 0

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], data['labels'], imgsz, batch_size, gs, 
            pad=pad, rect=rect, prefix=colorstr(f'{task}: '), num_offsets=num_offsets, num_states=num_states)[0]

    seen = 0
    mp, mr, map50, mAP, mAP_part, map50_part, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    if num_states:
        loss = torch.zeros(5, device=device)
    else:
        loss = torch.zeros(4, device=device)
    json_dump, json_dump_part_coco, json_dump_part_mr = [], [], []
    
    if data['dataset'] == "HumanParts":  # for map_sub cal (Subordination Metric) defined by Hier-R-CNN
        json_dump_part_coco_sub = []
    
    pbar = tqdm(dataloader, desc='Processing {} images'.format(task))
    for batch_i, (imgs, targets, paths, shapes) in enumerate(pbar):
        t_ = time_sync()
        imgs = imgs.to(device, non_blocking=True)
        # imgs_ori = imgs.clone()
        imgs = imgs.half() if half else imgs.float()  # uint8 to fp16/32
        imgs /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(imgs, augment=True, scales=scales, flips=flips)
        t1 += time_sync() - t

        # Compute loss
        if train_out:  # only computed if no scale / flipping
            if compute_loss:
                loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls, bpl, (cts)

        t = time_sync()
        
        # Run NMS
        # left_out = non_max_suppression(out, conf_thres, iou_thres, 
            # multi_label=False, agnostic=single_cls, num_offsets=data['num_offsets'])
        # body_dets = [d[d[:, 5] == 0] for d in left_out]  # [xyxy, conf, cls, part_points], cls = 0
        # part_dets = [d[d[:, 5] > 0] for d in left_out]  # [xyxy, conf, cls, part_points], cls = 1 or larger numbers
        
        body_dets = non_max_suppression(out, data['conf_thres'], data['iou_thres'], classes=[0],
            multi_label=False, agnostic=single_cls, num_offsets=num_offsets+num_states)
        part_dets = non_max_suppression(out, data['conf_thres_part'], data['iou_thres_part'], 
            # classes=list(range(1, 1 + data['num_offsets']//2)),
            classes=list(range(1, data['nc'])),
            multi_label=False, agnostic=single_cls, num_offsets=num_offsets+num_states)
        
        
        # Post-processing of body and part detections
        bboxes, points, scores, imgids, parts_dict, img_indexs = post_process_batch(
            data, imgs, paths, shapes, body_dets, part_dets, num_states=num_states)

        t2 += time_sync() - t
        seen += len(imgs)
        
        for i, (bbox, point, score, img_id) in enumerate(zip(bboxes, points, scores, imgids)):
        
            # img = imgs_ori[img_indexs[i]].cpu().numpy()
            # img = img[::-1].transpose((1, 2, 0)) # RGB to BGR, CHW to HWC
            # img = np.ascontiguousarray(img, dtype=np.uint8)
            # print(si, img.shape)
            
            bbox_new = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]  # [x0, y0, x1, y1] --> [x0, y0, w, h]

            # https://github.com/AibeeDetect/BFJDet/tree/main/eval_cp
            if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":  # data['num_offsets'] is 2
                # f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                # f_bbox = [f_bbox[0], f_bbox[1], f_bbox[2]-f_bbox[0], f_bbox[3]-f_bbox[1]]
                # f_bbox = f_bbox if f_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet

                # json_dump.append({
                    # 'image_id': img_id,
                    # 'category_id': 1,  # only one class 'person'
                    # 'bbox': [round(float(t), 3) for t in bbox_new],
                    # 'score': round(float(score), 3),  # person body score
                    # 'f_bbox': [round(float(t), 3) for t in f_bbox],  # the single bbox of body part (face or head)
                    # 'f_score': round(float(f_score), 3),  # the score of body part (face or head)
                # })

                # [x0, y0, x1, y1] = bbox
                # cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), thickness=2)
                # [px0, py0, px1, py1] = f_bbox
                # if px0 != 0 and py0 != 0:
                    # cv2.rectangle(img, (int(px0), int(py0)), (int(px1), int(py1)), (0, 255, 0), thickness=2)
                    # cv2.line(img, (int(x0), int(y0)), (int(px0), int(py0)), (255,255,0), thickness=2)
                # cv2.imwrite("./debug/"+Path(paths[img_indexs[i]]).stem+".jpg", img)

                '''t_bbox is suitable for face or head or any other bodypart'''
                t_score, t_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                t_bbox = [t_bbox[0], t_bbox[1], t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]]
                t_bbox = t_bbox if t_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet 

                json_dump.append({
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                    't_bbox': [round(float(t), 3) for t in t_bbox],  # the single bbox of body part (face or head)
                    't_score': round(float(t_score), 3),  # the score of body part (face or head)
                })
            
            
            if data['dataset'] == "BodyHands":  # data['num_offsets'] is 4, BodyHands does not label left-right
                lh_score, lh_bbox = point[0][2], point[0][3:]  # hand1 part, bbox format [x1, y1, x2, y2]
                lh_bbox = [lh_bbox[0], lh_bbox[1], lh_bbox[2]-lh_bbox[0], lh_bbox[3]-lh_bbox[1]]
                lh_bbox = lh_bbox if lh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet
                
                rh_score, rh_bbox = point[1][2], point[1][3:]  # hand2 part, bbox format [x1, y1, x2, y2]
                rh_bbox = [rh_bbox[0], rh_bbox[1], rh_bbox[2]-rh_bbox[0], rh_bbox[3]-rh_bbox[1]]
                rh_bbox = rh_bbox if rh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet 

                json_dump.append({
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                    'h1_bbox': [round(float(t), 3) for t in lh_bbox],  # the single bbox of body hand1 part
                    'h1_score': round(float(lh_score), 3),  # the score of body part (hand1)
                    'h2_bbox': [round(float(t), 3) for t in rh_bbox],  # the single bbox of body hand2 part
                    'h2_score': round(float(rh_score), 3),  # the score of body part (hand2)
                })

            if data['dataset'] == "ContactHands":  # data['num_offsets'] is 4, data['num_states'] is 8
                lh_score, lh_bbox = point[0][2], point[0][3:7]  # hand1 part, bbox format [x1, y1, x2, y2]
                lh_bbox = [lh_bbox[0], lh_bbox[1], lh_bbox[2]-lh_bbox[0], lh_bbox[3]-lh_bbox[1]]
                lh_bbox = lh_bbox if lh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet
                lh_state = point[0][7:]  # [NC, SC, PC, OC] for hand1
                
                rh_score, rh_bbox = point[1][2], point[1][3:7]  # hand2 part, bbox format [x1, y1, x2, y2]
                rh_bbox = [rh_bbox[0], rh_bbox[1], rh_bbox[2]-rh_bbox[0], rh_bbox[3]-rh_bbox[1]]
                rh_bbox = rh_bbox if rh_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet
                rh_state = point[1][7:]  # [NC, SC, PC, OC] for hand2

                json_dump.append({
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                    'h1_bbox': [round(float(t), 3) for t in lh_bbox],  # the single bbox of body hand1 part
                    'h1_score': round(float(lh_score), 3),  # the score of body part (hand1)
                    'h1_state': [round(float(t), 3) for t in lh_state],  # the contact state of hand1, range 0~1
                    'h2_bbox': [round(float(t), 3) for t in rh_bbox],  # the single bbox of body hand2 part
                    'h2_score': round(float(rh_score), 3),  # the score of body part (hand2)
                    'h2_state': [round(float(t), 3) for t in rh_state],  # the contact state of hand2, range 0~1
                })


            if data['dataset'] == "HumanParts":  # data['num_offsets'] is 12 = 2*6
                temp_dict = {
                    'image_id': img_id,
                    'category_id': 1,  # only one class 'person'
                    'bbox': [round(float(t), 3) for t in bbox_new],
                    'score': round(float(score), 3),  # person body score
                }
                dict_keys_list = ["h_bbox", "f_bbox", "lh_bbox", "rh_bbox", "lf_bbox", "rf_bbox"]
                for cls_ind in range(6):  # (head, face, lefthand, righthand, leftfoot, rightfoot)
                    t_score, t_bbox = point[cls_ind][2], point[cls_ind][3:]  # bodypart, bbox format [x1, y1, x2, y2]
                    t_bbox = [t_bbox[0], t_bbox[1], t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]]
                    t_bbox = t_bbox if t_score != 0 else [0, 0, 1, 1]  # this format is defined in BFJDet
                    temp_dict[dict_keys_list[cls_ind]] = [round(float(t), 3) for t in t_bbox]
                    temp_dict[dict_keys_list[cls_ind].replace("bbox", "score")] = round(float(t_score), 3)
                    
                    if t_score != 0:
                        json_dump_part_coco_sub.append({
                            'image_id': img_id,
                            'category_id': int(cls_ind+1),  # class of subset body part, 1~6
                            'bbox': [round(float(t), 3) for t in t_bbox],  # [x0, y0, w, h]
                            'score': float(t_score),  # body part score
                        })
                    
                assert data['part_type'] == "head", "We now only using head as the bodypart for mMR cal!"
                temp_dict['t_bbox'] = temp_dict["h_bbox"]
                temp_dict['t_score'] = temp_dict["h_score"]
                
                json_dump.append(temp_dict)


        imgids_rm_dup = list(set(imgids))
        for img_id in imgids_rm_dup:
            part_bbox_list = parts_dict[str(img_id)]
            for part_bbox in part_bbox_list:
                [x1, y1, x2, y2, conf, cls] = part_bbox
                json_dump_part_coco.append({
                    'image_id': img_id,
                    'category_id': int(cls),  # class of body part, e.g., [1,] for 'head' or 'face' and 'hands'(no l/r)
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x0, y0, w, h]
                    'score': float(conf),  # body part score
                })
                json_dump_part_mr.append({
                    'image_id': img_id,
                    'category_id': int(cls+1),  # class of body part, e.g., [2,] for 'head' or 'face' and 'hands'(no l/r)
                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],  # [x0, y0, w, h]
                    'score': float(conf),  # body part score
                })
                    
        # if batch_i > 2: break  # for prediction results debugging
        
        
    if not training:  # save json
        save_dir, weights_name = osp.split(weights)
        if not json_name:
            json_name = '{}_{}_c{}_i{}.json'.format(
                task, osp.splitext(weights_name)[0], data['conf_thres'], data['iou_thres'])
        else:
            if not json_name.endswith('.json'):
                json_name += '.json'
        json_path = osp.join(save_dir, json_name)
    else:
        tmp = tempfile.NamedTemporaryFile(mode='w+b')
        json_path = tmp.name + '.json'
    json_path_part_coco = json_path[:-5]+"_bodypart_coco.json"
    json_path_part_mr = json_path[:-5]+"_bodypart_mr.json"
    
    with open(json_path, 'w') as f:
        json.dump(json_dump, f)
    with open(json_path_part_coco, 'w') as f:
        json.dump(json_dump_part_coco, f)
    with open(json_path_part_mr, 'w') as f:
        json.dump(json_dump_part_mr, f)

    if len(json_dump) == 0:
        error_list = [0, 0, 0]
        return (mp, mr, map50, mAP, map50_part, mAP_part, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list

    if task in ('train', 'val'):
    
        if data['dataset'] != "HumanParts":
            print("###### person bbox mAP:", len(json_dump))
            if len(json_dump) != 0:
                annot = osp.join(data['path'], data['{}_annotations'.format(task)])
                coco = COCO(annot)
                result = coco.loadRes(json_path)
                eval = COCOeval(coco, result, iouType='bbox')
                # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
                eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                mAP, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
                
            print("###### bodypart bbox mAP:", len(json_dump_part_coco))
            if len(json_dump_part_coco) != 0:
                annot_part = osp.join(data['path'], data['{}_annotations_part'.format(task)])
                coco = COCO(annot_part)
                result = coco.loadRes(json_path_part_coco)
                eval = COCOeval(coco, result, iouType='bbox')
                # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
                eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                mAP_part, map50_part = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
                
        else:
            per_cls_ap_list = np.zeros((7, 7))  # (1 person and 6 bodyparts) * (6 kinds of ap and 1 num) for map
            per_cls_ap_list_sub = np.zeros((7, 7))  # (1 person and 6 bodyparts) * (6 kinds of ap and 1 num) for map sub
            
            '''person map'''
            print("###### person bbox mAP:", len(json_dump))
            if len(json_dump) != 0:
                annot = osp.join(data['path'], data['{}_annotations'.format(task)])
                coco = COCO(annot)
                result = coco.loadRes(json_path)
                eval = COCOeval(coco, result, iouType='bbox')
                # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
                eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
                eval.evaluate()
                eval.accumulate()
                eval.summarize()
                # (mAP@0.5:0.95, mAP@0.5, mAP@0.75, mAP@0.5:0.95/Small, mAP@0.5:0.95/Medium, mAP@0.5:0.95/Large,)
                mAP, map50, map75, mapS, mapM, mapL = eval.stats[:6]
                per_cls_ap_list[0,:] = np.array([mAP, map50, map75, mapS, mapM, mapL, len(json_dump)])
                per_cls_ap_list_sub[0,:] = np.array([mAP, map50, map75, mapS, mapM, mapL, len(json_dump)])
            
            '''6 bodyparts map'''
            print("###### bodypart bbox mAP:", len(json_dump_part_coco))
            if len(json_dump_part_coco) != 0:
                cls_pred_dict = {}
                for pred_dict in json_dump_part_coco:
                    cls_id = pred_dict['category_id']  # 1~6
                    pred_dict['category_id'] = 1  # we set all class as 1 in their gt_anno json files
                    if cls_id in cls_pred_dict:
                        cls_pred_dict[cls_id].append(pred_dict)
                    else:
                        cls_pred_dict[cls_id] = [pred_dict]
                        
                for cls_ind, cls_name in enumerate(data['names']):
                    if cls_ind == 0:
                        continue  # person class
                    if cls_ind not in cls_pred_dict:
                        print("###### bodypart bbox mAP (%s):"%(cls_name), "This bodypart has none of one bbox detected!")
                        continue
                    else:
                        print("###### bodypart bbox mAP (%s):"%(cls_name), len(cls_pred_dict[cls_ind]))
                        json_path_part_coco_cls = json_path[:-5] + "_bodypart_coco_%s.json"%(cls_name)
                        with open(json_path_part_coco_cls, 'w') as f:
                            json.dump(cls_pred_dict[cls_ind], f)
                        
                        annot_part_path = data['{}_annotations_part'.format(task)].replace("bodypart", cls_name)
                        annot_part = osp.join(data['path'], annot_part_path)
                        coco = COCO(annot_part)
                        result = coco.loadRes(json_path_part_coco_cls)
                        eval = COCOeval(coco, result, iouType='bbox')
                        # eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
                        eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
                        eval.evaluate()
                        eval.accumulate()
                        eval.summarize()
                        # (mAP@0.5:0.95, mAP@0.5, mAP@0.75, mAP@0.5:0.95/Small, mAP@0.5:0.95/Medium, mAP@0.5:0.95/Large,)
                        mAP_part, map50_part, map75_part, mapS_part, mapM_part, mapL_part = eval.stats[:6]  
                        per_cls_ap_list[cls_ind,:] = np.array([mAP_part, map50_part, map75_part, 
                            mapS_part, mapM_part, mapL_part, len(cls_pred_dict[cls_ind])])
            
            '''6 bodyparts map sub'''
            cal_map_sub = True
            if cal_map_sub:  # for map_sub cal (Subordination Metric) defined by Hier-R-CNN
                print("###### bodypart bbox mAP[sub]:", len(json_dump_part_coco_sub))
                if len(json_dump_part_coco_sub) != 0:
                    cls_pred_dict = {}
                    for pred_dict in json_dump_part_coco_sub:
                        cls_id = pred_dict['category_id']  # 1~6
                        pred_dict['category_id'] = 1  # we set all class as 1 in their gt_anno json files
                        if cls_id in cls_pred_dict:
                            cls_pred_dict[cls_id].append(pred_dict)
                        else:
                            cls_pred_dict[cls_id] = [pred_dict]
                            
                    for cls_ind, cls_name in enumerate(data['names']):
                        if cls_ind == 0:
                            continue  # person class
                        if cls_ind not in cls_pred_dict:
                            print("###### bodypart bbox mAP[sub] (%s):"%(cls_name), "This bodypart has none of one bbox detected!")
                            continue
                        else:
                            print("###### bodypart bbox mAP[sub] (%s):"%(cls_name), len(cls_pred_dict[cls_ind]))
                            json_path_part_coco_cls = json_path[:-5] + "_bodypart_coco_%s.json"%(cls_name) # 
                            with open(json_path_part_coco_cls, 'w') as f:
                                json.dump(cls_pred_dict[cls_ind], f)
                            
                            annot_part_path = data['{}_annotations_part'.format(task)].replace("bodypart", cls_name)
                            annot_part = osp.join(data['path'], annot_part_path)
                            coco = COCO(annot_part)
                            result = coco.loadRes(json_path_part_coco_cls)
                            eval = COCOeval(coco, result, iouType='bbox')
                            eval.params.imgIds = [int(Path(x).stem.split("_")[-1]) for x in dataloader.dataset.img_files]
                            eval.evaluate()
                            eval.accumulate()
                            eval.summarize()
                            # (mAP@0.5:0.95, mAP@0.5, mAP@0.75, mAP@0.5:0.95/Small, mAP@0.5:0.95/Medium, mAP@0.5:0.95/Large,)
                            mAP_part_sub, map50_part_sub, map75_part_sub, mapS_part_sub, mapM_part_sub, mapL_part_sub = eval.stats[:6]  
                            per_cls_ap_list_sub[cls_ind,:] = np.array([
                                mAP_part_sub, map50_part_sub, map75_part_sub, 
                                mapS_part_sub, mapM_part_sub, mapL_part_sub, len(cls_pred_dict[cls_ind])])
                
                '''6 bodyparts map sub print'''        
                for cls_ind, cls_name in enumerate(data['names']):  # including person class
                    mAP_sub, map50_sub, map75_sub, mapS_sub, mapM_sub, mapL_sub, num_sub = per_cls_ap_list_sub[cls_ind]
                    print("Per Category AP@s (%s):\t %.4f, %.4f, %.4f, %.4f, %.4f, %.4f; %d"%(
                        cls_name, mAP_sub, map50_sub, map75_sub, mapS_sub, mapM_sub, mapL_sub, num_sub))
                mAP_sub, map50_sub, map75_sub, mapS_sub, mapM_sub, mapL_sub, _ = np.mean(per_cls_ap_list_sub, 0)
                print("All Category AP@s (%s):\t %.4f, %.4f, %.4f, %.4f, %.4f, %.4f; %d"%(
                    "all", mAP_sub, map50_sub, map75_sub, mapS_sub, mapM_sub, mapL_sub, len(json_dump_part_coco_sub) ))
            
            '''6 bodyparts map print'''
            for cls_ind, cls_name in enumerate(data['names']):  # including person class
                mAP, map50, map75, mapS, mapM, mapL, num_all = per_cls_ap_list[cls_ind]
                print("Per Category AP@a (%s):\t %.4f, %.4f, %.4f, %.4f, %.4f, %.4f; %d"%(
                    cls_name, mAP, map50, map75, mapS, mapM, mapL, num_all))
            mAP, map50, map75, mapS, mapM, mapL, _ = np.mean(per_cls_ap_list, 0)
            print("All Category AP@a (%s):\t %.4f, %.4f, %.4f, %.4f, %.4f, %.4f; %d"%(
                "all", mAP, map50, map75, mapS, mapM, mapL, len(json_dump_part_coco) ))
            
            # return values for best model choosing: person_mAP_avg, person_mAP50_avg, head_mAP, head_mAP50
            mAP, map50, mAP_part, map50_part = per_cls_ap_list[0,0], per_cls_ap_list[0,1], per_cls_ap_list[1,0], per_cls_ap_list[1,1]


        if data['dataset'] == "CityPersons":
            if len(json_dump) != 0 and len(json_dump_part_mr) != 0:
                MR_body_list, MR_part_list, mMR_list, MR_body, MR_part, mMR_avg = body_part_association_evaluation(
                    json_path, json_path_part_mr, data)
            else:
                MR_body_list, MR_part_list, mMR_list = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
                MR_body, MR_part, mMR_avg = 0, 0, 0

            print("[MR_body_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                MR_body_list[0], MR_body_list[1], MR_body_list[2], MR_body_list[3] ))
            print("[MR_part_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                MR_part_list[0], MR_part_list[1], MR_part_list[2], MR_part_list[3] ))
            print("[mMR_all_list]: Reasonable: %.3f, Bare: %.3f, Partial: %.3f, Heavy: %.3f"%(
                mMR_list[0], mMR_list[1], mMR_list[2], mMR_list[3] ))
            print("[MR_body, MR_part, mMR]: %.3f, %.3f, %.3f"%(MR_body, MR_part, mMR_avg))
            error_list = [MR_body, MR_part, mMR_avg]

        if data['dataset'] == "CrowdHuman" or data['dataset'] == "HumanParts":
            if len(json_dump) != 0 and len(json_dump_part_mr) != 0:
                AP_body, MR_body, AP_part, MR_part, mMR_list, mMR_avg = body_part_association_evaluation(
                    json_path, json_path_part_mr, data)
            else:
                AP_body, MR_body, AP_part, MR_part, mMR_avg = 0, 0, 0, 0, 0
                mMR_list = [0, 0, 0, 0]  # "Reasonable", "Small", "Heavy", "All"
            
            print("[AP@.5&MR]: AP_body: %.3f, AP_part: %.3f, MR_body: %.3f, MR_part: %.3f, mMR_avg: %.3f"%(
                AP_body, AP_part, MR_body, MR_part, mMR_avg ))
            print("[mMR_list]: Reasonable: %.3f, Small: %.3f, Heavy: %.3f, All: %.3f"%(
                mMR_list[0], mMR_list[1], mMR_list[2], mMR_list[3] ))
            error_list = [MR_body, MR_part, mMR_avg]
            # error_list = [MR_body, MR_part, mMR_list[-1]]  # All
            # error_list = [MR_body, MR_part, mMR_list[0]]  # Reasonable
            
        if data['dataset'] == "BodyHands":
            print("[BodyHands]: using <Cond. Accuracy> and <Joint AP> instead of <MR_body>, <MR_part> and <mMR> !")
            
            if len(json_dump) != 0:
                ap_dual, ap_single = body_part_association_evaluation(json_path, json_path_part_mr, data)
            else:
                ap_dual, ap_single = 0, 0
            
            print("AP_Dual(Joint-AP): %.3f, AP_Single: %.3f"%(ap_dual, ap_single))
            error_list = [1, 1, 1]
        
        if data['dataset'] == "ContactHands":
            print("[ContactHands]: using many <APs> and <mAP> instead of <MR_body>, <MR_part> and <mMR> !")

            if len(json_dump) != 0:
                hand_AP, NC_AP, SC_AP, PC_AP, OC_AP, mAP_contact = body_part_association_evaluation(
                    json_path, json_path_part_mr, data)
            else:
                hand_AP, NC_AP, SC_AP, PC_AP, OC_AP, mAP_contact = 0, 0, 0, 0, 0, 0
            
            print("APs(hand; NC, SC, PC, OC): %.4f; %.4f, %.4f, %.4f, %.4f, mAP_contact: %.4f"%(
                hand_AP, NC_AP, SC_AP, PC_AP, OC_AP, mAP_contact))
            error_list = [1, 1, 1]
            
    if training:
        tmp.close()

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training and task != 'test':
        os.rename(json_path, osp.splitext(json_path)[0] + '_ap{:.4f}.json'.format(mAP))
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.3fms pre-process, %.3fms inference, %.3fms NMS per image at shape {shape}' % t)

    model.float()  # for training
    # return (mp, mr, map50, mAP, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t  # for compatibility with train
    return (mp, mr, map50, mAP, map50_part, mAP_part, *(loss.cpu() / len(dataloader)).tolist()), np.zeros(nc), t, error_list


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', default='yolov5s6.pt')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=1280, help='inference size (pixels)')
    parser.add_argument('--task', default='val', help='train, val, test')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--flips', type=int, nargs='+', default=[-1])
    parser.add_argument('--rect', action='store_true', help='rectangular input image')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--pad', type=int, default=0, help='padding for two-stage inference')
    parser.add_argument('--json-name', type=str, default='', help='optional name for saved json file')

    opt = parser.parse_args()
    opt.flips = [None if f == -1 else f for f in opt.flips]
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
