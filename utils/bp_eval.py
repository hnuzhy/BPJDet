
import os
import json
import numpy as np
import copy
from tqdm import tqdm

from utils.eval.eval_mmr_cityperson import eval_mmr
from utils.eval.eval_mr_cityperson import eval_mr

from utils.eval.compute_APMR import compute_APMR
from utils.eval.compute_MMR import compute_MMR
from utils.eval.misc_utils import save_json_lines

from utils.eval.eval_bodyhands import voc_eval


def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict

def sort_images_by_image_id(imgs_dict_list):
    imginfo_imgid_dict = {}  # dict of dict list
    for i, image_dict in enumerate(imgs_dict_list):
        image_id = str(image_dict['id'])  # the 'image_id'
        assert image_id not in imginfo_imgid_dict, "this image has appeared twice!"+str(image_dict)
        imginfo_imgid_dict[image_id] = image_dict
    return imginfo_imgid_dict

def calculate_bbox_iou(bboxA, bboxB, format='xyxy'):
    if format == 'xywh':  # xy is in top-left, wh is size
        [Ax, Ay, Aw, Ah] = bboxA[0:4]
        [Ax0, Ay0, Ax1, Ay1] = [Ax, Ay, Ax+Aw, Ay+Ah]
        [Bx, By, Bw, Bh] = bboxB[0:4]
        [Bx0, By0, Bx1, By1] = [Bx, By, Bx+Bw, By+Bh]
    if format == 'xyxy':
        [Ax0, Ay0, Ax1, Ay1] = bboxA[0:4]
        [Bx0, By0, Bx1, By1] = bboxB[0:4]
        
    W = min(Ax1, Bx1) - max(Ax0, Bx0)
    H = min(Ay1, By1) - max(Ay0, By0)
    if W <= 0 or H <= 0:
        return 0
    else:
        areaA = (Ax1 - Ax0)*(Ay1 - Ay0)
        areaB = (Bx1 - Bx0)*(By1 - By0)
        crossArea = W * H
        return crossArea/(areaA + areaB - crossArea)


def pred_boxes_dump_BFJDet(pred_json_path, anno_json_path, part_type="body"):
    
    pred_boxes_dict_list = json.load(open(pred_json_path, "r"))
    imgid_preds_dict = sort_labels_by_image_id(pred_boxes_dict_list)
    
    anno_dict = json.load(open(anno_json_path, "r"))
    anno_imgs_dict_list = anno_dict["images"]
    info_imgid_dict = sort_images_by_image_id(anno_imgs_dict_list)
    
    pred_boxes_dict_list_new = []
    
    # refer https://github.com/AibeeDetect/BFJDet/blob/main/tools/test.py
    if part_type == "body":
        class_tag = 1
    elif part_type == "face" or part_type == "head":
        class_tag = 2
        
    for img_id, preds_dict_list in imgid_preds_dict.items():
        temp_dict = {}
        temp_dict["ID"] = info_imgid_dict[img_id]['file_name'],  # str of image name
        temp_dict["height"] = info_imgid_dict[img_id]['height'],  # used in clip_all_boader()
        temp_dict["width"] = info_imgid_dict[img_id]['width'],  # used in clip_all_boader()
        dtboxes = []
        for preds_dict in preds_dict_list:
            assert preds_dict['category_id'] == class_tag, "unmatched category_id:"+pred_json_path
            dtboxes.append({
                'box': preds_dict['bbox'],
                'score': preds_dict['score'],
                'tag': class_tag
            })
        temp_dict["dtboxes"] = dtboxes
        pred_boxes_dict_list_new.append(temp_dict)

    json_path_root, json_path_name = os.path.split(pred_json_path)
    json_path_new = os.path.join(json_path_root, "dump-" + json_path_name) # for matching utils.eval.database.py
    
    # with open(json_path_new, "w") as json_write:
        # json.dump(pred_boxes_dict_list_new, json_write)
    
    save_json_lines(pred_boxes_dict_list_new, json_path_new)
    
    return json_path_new


# process COCO format predictions into BodyHands format
def process_predictions(json_path_body, val_json_path):

    pred_boxes_dict_list = json.load(open(json_path_body, "r"))
    imgid_preds_dict = sort_labels_by_image_id(pred_boxes_dict_list)
    
    anno_dict = json.load(open(val_json_path, "r"))
    anno_imgs_dict_list = anno_dict["images"]
    info_imgid_dict = sort_images_by_image_id(anno_imgs_dict_list)
    
    all_predictions_lines = []
    
    for img_id, preds_dict_list in imgid_preds_dict.items():
        img_name = info_imgid_dict[img_id]['file_name'] # str of image name
        
        # remove .jpg, e.g., 'test_4677_1_1140.jpg' --> 'test_4677_1_1140'
        # remove the last image_id added by us for the BPJDet task, e.g., 'test_4677_1_1140' --> 'test_4677_1'
        item_list = img_name[:-4].split("_")  
        img_name_real = ""
        for item in item_list[:-1]:
            img_name_real += item + "_"
        img_name_real = img_name_real[:-1]
        
        for preds_dict in preds_dict_list:
            bx1, by1, bw, bh = preds_dict['bbox']
            bx2, by2 = bx1 + bw, by1 + bh
            bx1, by1 = bx1 +1 , by1 + 1
            
            h1_score = preds_dict['h1_score']
            if h1_score != 0:
                hx1, hy1, hw, hh = preds_dict['h1_bbox']
                hx2, hy2 = hx1 + hw, hy1 + hh
                hx1, hy1 = hx1 +1 , hy1 + 1
                save_line = f"{img_name_real} {h1_score} {hx1} {hy1} {hx2} {hy2} {bx1} {by1} {bx2} {by2}"
                all_predictions_lines.append(save_line)
                
            h2_score = preds_dict['h2_score']
            if h2_score != 0:
                hx1, hy1, hw, hh = preds_dict['h2_bbox']
                hx2, hy2 = hx1 + hw, hy1 + hh
                hx1, hy1 = hx1 +1 , hy1 + 1
                save_line = f"{img_name_real} {h2_score} {hx1} {hy1} {hx2} {hy2} {bx1} {by1} {bx2} {by2}"
                all_predictions_lines.append(save_line)
                
    return all_predictions_lines


def body_part_association_evaluation(json_path_body, json_path_part, data_cfg):

    # https://github.com/AibeeDetect/BFJDet/tree/main/eval_cp
    if data_cfg['dataset'] == "CityPersons":
        val_bhf_path = os.path.join(data_cfg["path"], data_cfg["val_bhf_path"])
        if data_cfg['part_type'] == "face":
            val_bf_path = os.path.join(data_cfg["path"], data_cfg["val_bf_path"])

            MR_body_list = eval_mr(val_bf_path, json_path_body, type='body')
            MR_body = sum(MR_body_list) / len(MR_body_list) # "Reasonable", "Bare", "Partial", "Heavy"
            MR_part_list = eval_mr(val_bf_path, json_path_part, type='face')
            MR_part = sum(MR_part_list) / len(MR_part_list) # "Reasonable", "Bare", "Partial", "Heavy"
            mMR_list = eval_mmr(val_bhf_path, json_path_body, body_part='face')
            mMR = sum(mMR_list) / len(mMR_list) # "Reasonable", "Bare", "Partial", "Heavy"
            
        if data_cfg['part_type'] == "head":
            val_bh_path = os.path.join(data_cfg["path"], data_cfg["val_bh_path"])

            MR_body_list = eval_mr(val_bh_path, json_path_body, type='body')
            MR_body = sum(MR_body_list) / len(MR_body_list) # "Reasonable", "Bare", "Partial", "Heavy"
            MR_part_list = eval_mr(val_bh_path, json_path_part, type='head')
            MR_part = sum(MR_part_list) / len(MR_part_list) # "Reasonable", "Bare", "Partial", "Heavy"
            mMR_list = eval_mmr(val_bhf_path, json_path_body, body_part='head')
            mMR = sum(mMR_list) / len(mMR_list) # "Reasonable", "Bare", "Partial", "Heavy"
            
        return MR_body_list, MR_part_list, mMR_list, MR_body, MR_part, mMR
        
        
    # https://github.com/AibeeDetect/BFJDet/blob/main/lib/evaluate/compute_MMR.py
    if data_cfg['dataset'] == "CrowdHuman":
        val_bhf_path = os.path.join(data_cfg["path"], data_cfg["val_bhf_path"])
        
        json_path_body_new = pred_boxes_dump_BFJDet(json_path_body, val_bhf_path, part_type="body")
        
        AP_body, MR_body = compute_APMR(json_path_body_new, val_bhf_path, 
            'box', if_face=False, body_part='body')

        if data_cfg['part_type'] == "face":
            json_path_part_new = pred_boxes_dump_BFJDet(json_path_part, val_bhf_path, part_type="face")
            AP_part, MR_part = compute_APMR(json_path_part_new, val_bhf_path, 
                'box', if_face=True, body_part='face')
                
            mMR_list = compute_MMR(json_path_body, val_bhf_path, body_part='face')
            mMR_avg = sum(mMR_list) / len(mMR_list)  # "Reasonable", "Small", "Heavy", "All"
            
        if data_cfg['part_type'] == "head":
            json_path_part_new = pred_boxes_dump_BFJDet(json_path_part, val_bhf_path, part_type="head")
            AP_part, MR_part = compute_APMR(json_path_part_new, val_bhf_path, 
                'box', if_face=True, body_part='head')
                
            mMR_list = compute_MMR(json_path_body, val_bhf_path, body_part='head')
            mMR_avg = sum(mMR_list) / len(mMR_list)  # "Reasonable", "Small", "Heavy", "All"
            
        return AP_body, MR_body, AP_part, MR_part, mMR_list, mMR_avg

    
    # https://github.com/cvlab-stonybrook/BodyHands/blob/main/bodyhands/evaluation/evaluator.py
    if data_cfg['dataset'] == "BodyHands":
        val_json_path = os.path.join(data_cfg['path'], data_cfg['val_annotations'])
        anno_file_template = os.path.join(data_cfg['path'], 'VOC2007', "Annotations", "{}.xml")
        image_set_path = os.path.join(data_cfg['path'], 'VOC2007', "ImageSets", "Main", "test.txt")
        cls_name = data_cfg['part_type']  # 'hand'
        is_2007_year = True  # True or False, we set it as True following BodyHands
        
        all_predictions_lines = process_predictions(json_path_body, val_json_path)
        
        rec_dual, prec_dual, ap_dual = voc_eval(all_predictions_lines, anno_file_template, image_set_path, 
            cls_name, ovthresh=0.5, use_07_metric=is_2007_year, single_metric=False)  # aps_dual_metric
        # rec_dual and prec_dual are list
        
        rec_single, prec_single, ap_single = voc_eval(all_predictions_lines, anno_file_template, image_set_path, 
            cls_name, ovthresh=0.5, use_07_metric=is_2007_year, single_metric=True)  # aps_single_metric
        # rec_single and prec_single are list
        
        return ap_dual * 100 , ap_single * 100
