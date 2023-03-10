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
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--line-thick', type=int, default=2, help='thickness of lines')
    parser.add_argument('--counting', type=int, default=0, help='0 or 1, plot counting')

    args = parser.parse_args()

    with open(args.data) as f:
        data = yaml.safe_load(f)  # load data dict

    device = select_device(args.device, batch_size=1)
    print('Using device: {}'.format(device))

    model = attempt_load(args.weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(args.imgsz, s=stride)  # check image size
    dataset = LoadImages(args.img_path, img_size=imgsz, stride=stride, auto=True)
    dataset_iter = iter(dataset)
    
    # if data['dataset'] == "CityPersons":
        # data['dist_thre'] = 200  # the largest dist threshold for matching, large than it will not be replaced
    # else:
        # data['dist_thre'] = 50
    data['conf_thres_part'] = args.conf_thres  # the larger conf threshold for filtering body-part detection proposals
    data['iou_thres_part'] = args.iou_thres  # the smaller iou threshold for filtering body-part detection proposals
    data['match_iou_thres'] = args.match_iou  # whether a body-part in matched with one body bbox
    
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
            classes=[0], num_offsets=data['num_offsets'])
        part_dets = non_max_suppression(out_ori, args.conf_thres, args.iou_thres, 
            classes=list(range(1, 1 + data['num_offsets']//2)), num_offsets=data['num_offsets'])
        
        # Post-processing of body and part detections
        bboxes, points, scores, _, _, _ = post_process_batch(
            data, img, [], [[im0.shape[:2]]], body_dets, part_dets)
        
        # args.line_thick = max(im0.shape[:2]) // 1280 + 3
        args.line_thick = max(im0.shape[:2]) // 1000 + 3
        
        instance_counting = 0
        
        for i, (bbox, point, score) in enumerate(zip(bboxes, points, scores)):
            [x1, y1, x2, y2] = bbox

            # https://github.com/AibeeDetect/BFJDet/tree/main/eval_cp
            # cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=args.line_thick)
            # if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":  # data['num_offsets'] is 2
                # f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                # if f_score != 0:
                    # [px1, py1, px2, py2] = f_bbox
                    # cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), thickness=args.line_thick)
                    # cv2.line(im0, (int(x1), int(y1)), (int(px1), int(py1)), (255,0,255), thickness=args.line_thick)  # magenta

            # if data['dataset'] == "BodyHands":  # data['num_offsets'] is 4
                # lh_score, lh_bbox = point[0][2], point[0][3:]  # left-hand part, bbox format [x1, y1, x2, y2]
                # if lh_score != 0:
                    # [px1, py1, px2, py2] = lh_bbox
                    # cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), thickness=args.line_thick)
                    # cv2.line(im0, (int(x1), int(y1)), (int(px1), int(py1)), (255,0,255), thickness=args.line_thick)  # magenta
                
                # rh_score, rh_bbox = point[1][2], point[1][3:]  # right-hand part, bbox format [x1, y1, x2, y2]
                # if rh_score != 0:
                    # [px1, py1, px2, py2] = rh_bbox
                    # cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), (0, 255, 0), thickness=args.line_thick)
                    # cv2.line(im0, (int(x1), int(y1)), (int(px1), int(py1)), (255,0,255), thickness=args.line_thick)  # magenta
            
            
            color = colors_list[i%len(colors_list)]
            
            
            if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":  # data['num_offsets'] is 2
                f_score, f_bbox = point[0][2], point[0][3:]  # bbox format [x1, y1, x2, y2]
                if data['part_type'] == "head" and f_score == 0:  # for the body-head pair, we must have a detected head
                    continue
                    
                instance_counting += 1               
                    
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)
                if f_score != 0:
                    [px1, py1, px2, py2] = f_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)

            if data['dataset'] == "BodyHands":  # data['num_offsets'] is 4
                cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=args.line_thick)
                lh_score, lh_bbox = point[0][2], point[0][3:]  # left-hand part, bbox format [x1, y1, x2, y2]
                if lh_score != 0:
                    [px1, py1, px2, py2] = lh_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
                
                rh_score, rh_bbox = point[1][2], point[1][3:]  # right-hand part, bbox format [x1, y1, x2, y2]
                if rh_score != 0:
                    [px1, py1, px2, py2] = rh_bbox
                    cv2.rectangle(im0, (int(px1), int(py1)), (int(px2), int(py2)), color, thickness=args.line_thick)
        
        if args.counting:
            cv2.putText(im0, "Num:"+str(instance_counting), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0,0,255), 2, cv2.LINE_AA)
            
        # cv2.imwrite(single_path[:-4]+"_res.jpg", im0)
        cv2.imwrite(single_path[:-4]+"_res_%s.jpg"%(data['part_type']), im0)

            
        
  