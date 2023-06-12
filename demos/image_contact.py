import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())

import torch
import argparse
import yaml
import cv2
import math
import os.path as osp
import numpy as np

import os
import json

from utils.torch_utils import select_device
from utils.general import check_img_size, scale_coords, non_max_suppression
from utils.datasets import LoadImages
from models.experimental import attempt_load
from val import post_process_batch

colors_list = [
        # [255, 0, 0], [255, 127, 0], [255, 255, 0], [127, 255, 0], [0, 255, 0], [0, 255, 127], 
        # [0, 255, 255], [0, 127, 255], [0, 0, 255], [127, 0, 255], [255, 0, 255], [255, 0, 127],
        [255, 127, 0], [127, 255, 0], [0, 255, 127], [0, 127, 255], [127, 0, 255], [255, 0, 127],
        [255, 255, 255],
        [127, 0, 127], [0, 127, 127], [127, 127, 0], [127, 0, 0], [127, 0, 0], [0, 127, 0],
        [127, 127, 127],
        [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0],
        [0, 0, 0],
        [255, 127, 255], [127, 255, 255], [255, 255, 127], [127, 127, 255], [255, 127, 127], [255, 127, 127],
    ]  # 27 colors

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/JointBP_CityPersons_face.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.6, help='Matching IoU threshold')
    parser.add_argument('--body_state_w', type=float, default=0.4, help='body state threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--line-thick', type=int, default=1, help='thickness of lines')
    
    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict
        
    assert data['dataset'] == "ContactHands", "We now only support models trained on ContactHands!!!"
    num_states = data['num_states']  # should be 8 of two hands with 4 states for each hand
    
    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    data['match_iou_thres'] = args.match_iou  # whether a body-part in matched with one body bbox
    data['body_state_w'] = args.body_state_w  # human body contact state weight, which is for ContactHands
    
    print(args.img_path, len(dataset))
    for index in range(len(dataset)):
        
        (single_path, img, im0, _) = next(dataset_iter)
        
        if '_res' in single_path or '_vis' in single_path:
            continue
        
        print(index, single_path, "\n")
        
        img = torch.from_numpy(img).to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        out_ori = model(img, augment=True, scales=args.scales)[0]
        body_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=[0], num_offsets=data['num_offsets']+num_states)
        part_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=list(range(1, 1 + data['num_offsets']//2)), num_offsets=data['num_offsets']+num_states)
        
        # Post-processing of body and part detections
        bboxes, points, scores, imgids, parts_dict, _ = post_process_batch(
            data, img, [], [[im0.shape[:2]]], body_dets, part_dets, num_states=num_states)
        
        # args.line_thick = max(im0.shape[:2]) // 1280 + 3
        # args.line_thick = max(im0.shape[:2]) // 1000 + 3
        
        '''
        https://github.com/cvlab-stonybrook/ContactHands
        --sc 0.5 --pc 0.4 --oc 0.6

        The parameters sc, pc, oc denote thresholds for Self-Contact, Other-Person-Contact, 
        and Object-Contact, respectively.

        The thresholds are in the range [0.0, 1.0]. Lower thresholds increases the recall and 
        higher thresholds increases precision. Choose them according to your need.

        If the predicted contact state score for all three of them are less than the 
        corresponding thresholds, the contact state No-Contact will be choosen.
        '''
        # sc_thre, pc_thre, oc_thre = 0.5, 0.4, 0.6  # ContactHands thresholds
        sc_thre, pc_thre, oc_thre = 0.5, 0.6, 0.4  # our thresholds
        sc_colors = [(250,225,170), (193,153,245), (76,76,255), (173,248,255)]  # [NC, SC, PC, OC], BGR for opencv2
        alpha = 0.8
        
        im0_overlay = im0.copy()
        
        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            color = colors_list[i%len(colors_list)]
            [x1, y1, x2, y2] = bbox
            cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)  # body bbox
            cv2.rectangle(im0_overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)  # body bbox
            
            h1_score, h1_bbox, h1_state = point[0][2], point[0][3:7], point[0][7:]  # bbox format [x1, y1, x2, y2]
            if h1_score != 0:  # this body bbox has the matched hand1 bbox
                [px1, py1, px2, py2] = h1_bbox

                [NC, SC, PC, OC] = h1_state  # [NC, SC, PC, OC] for hand1
                sc_state = 1 if SC > sc_thre else 0
                pc_state = 1 if PC > pc_thre else 0
                oc_state = 1 if OC > oc_thre else 0
                nc_state = 0 if (sc_state or pc_state or oc_state) else 1
                '''visualization v1'''
                # state_str = str(nc_state) + str(sc_state) + str(pc_state) + str(oc_state)
                # cv2.putText(im0, state_str, (int(px1), int(py1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    # color, 1, cv2.LINE_AA)
                '''visualization v2'''
                state_str = ""  # may have more than one state
                if nc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[0], -1)
                    state_str += "NC,"
                if sc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[1], -1)
                    state_str += "SC,"
                if pc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[2], -1)
                    state_str += "PC,"
                if oc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[3], -1)
                    state_str += "OC,"
                cv2.putText(im0_overlay, state_str[:-1], (int(px1+2), int(py2-2)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,0), 1, cv2.LINE_AA)
                    
                cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
                cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
  
            h2_score, h2_bbox, h2_state = point[1][2], point[1][3:7], point[1][7:]  # bbox format [x1, y1, x2, y2]
            if h2_score != 0:  # this body bbox has the matched hand1 bbox
                [px1, py1, px2, py2] = h2_bbox

                [NC, SC, PC, OC] = h2_state  # [NC, SC, PC, OC] for hand2
                sc_state = 1 if SC > sc_thre else 0
                pc_state = 1 if PC > pc_thre else 0
                oc_state = 1 if OC > oc_thre else 0
                nc_state = 0 if (sc_state or pc_state or oc_state) else 1
                '''visualization v1'''
                # state_str = str(nc_state) + str(sc_state) + str(pc_state) + str(oc_state)
                # cv2.putText(im0, state_str, (int(px1), int(py1)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                    # color, 1, cv2.LINE_AA)
                '''visualization v2'''
                state_str = ""  # may have more than one state
                if nc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[0], -1)
                    state_str += "NC,"
                if sc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[1], -1)
                    state_str += "SC,"
                if pc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[2], -1)
                    state_str += "PC,"
                if oc_state:
                    cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), sc_colors[3], -1)
                    state_str += "OC,"
                cv2.putText(im0_overlay, state_str[:-1], (int(px1+2), int(py2-2)), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0,0,0), 1, cv2.LINE_AA)
                    
                cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
                cv2.rectangle(im0_overlay, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)

        image_new = cv2.addWeighted(im0_overlay, alpha, im0, 1 - alpha, 0)   
                
        # cv2.imwrite(single_path[:-4]+"_res_contact.jpg", im0)
        cv2.imwrite(single_path[:-4]+"_res_contact.jpg", image_new)
