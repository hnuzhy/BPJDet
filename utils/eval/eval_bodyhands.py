
# https://github.com/cvlab-stonybrook/BodyHands/blob/main/bodyhands/evaluation/evaluator.py

import os
import numpy as np
import xml.etree.ElementTree as ET

from functools import lru_cache
from fvcore.common.file_io import PathManager


##############################################################################
#
# Below code is modified from
# https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/voc_eval.py
# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

"""Python implementation of the PASCAL VOC devkit's AP evaluation code."""


@lru_cache(maxsize=None)
def parse_rec(filename):
    """Parse a PASCAL VOC xml file."""
    with PathManager.open(filename) as f:
        tree = ET.parse(f)
    
    hand_annotations = {}
    body_annotations = {}
    for obj in tree.findall("object"):
        obj_struct = {}
        obj_struct["name"] = obj.find("name").text
        obj_struct["pose"] = obj.find("pose").text
        obj_struct["truncated"] = int(obj.find("truncated").text)
        obj_struct["difficult"] = int(obj.find("difficult").text)
        cls_ = obj.find("name").text
        bbox = obj.find("bndbox")
        obj_struct["bbox"] = [
            int(bbox.find("xmin").text),
            int(bbox.find("ymin").text),
            int(bbox.find("xmax").text),
            int(bbox.find("ymax").text),
        ]
        body_id = int(obj.find("body_id").text)
        if cls_ == "hand":
            if body_id in hand_annotations:
                    pass
            else:
                hand_annotations[body_id] = []
            hand_annotations[body_id].append(obj_struct)
        else:
            body_annotations[body_id] = [obj_struct] 

    objects = []
    for body_id in hand_annotations:
        body_ann = body_annotations[body_id][0]
        for hand_ann in hand_annotations[body_id]:
            hand_ann["body_box"] = body_ann["bbox"]
            objects.append(hand_ann)

    return objects


def voc_ap(rec, prec, use_07_metric=False):

    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


# def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, single_metric=False):
def voc_eval(detlines, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False, single_metric=False):

    with PathManager.open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    recs = {}
    for imagename in imagenames:
        recs[imagename] = parse_rec(annopath.format(imagename))

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        body_box = np.array([x["body_box"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {"bbox": bbox, "body_box": body_box, "difficult": difficult, "det": det}

    # detfile = detpath.format(classname)
    # with open(detfile, "r") as f:
        # lines = f.readlines()

    # splitlines = [x.strip().split(" ") for x in lines]
    splitlines = [x.strip().split(" ") for x in detlines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:6]] for x in splitlines]).reshape(-1, 4)
    body_BB = np.array([[float(z) for z in x[6:]] for x in splitlines]).reshape(-1, 4)

    sorted_ind = np.argsort(-confidence)
    BB = BB[sorted_ind, :]
    body_BB = body_BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    body_acc_count = 0
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        body_bb = body_BB[d, :].astype(float)
        ovmax = -np.inf
        body_ovmax = -np.inf
        BBGT = R["bbox"].astype(float)
        body_BBGT = R["body_box"].astype(float)

        if BBGT.size > 0:

            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            uni = (
                (bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
                + (BBGT[:, 2] - BBGT[:, 0] + 1.0) * (BBGT[:, 3] - BBGT[:, 1] + 1.0)
                - inters
            )

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:

                    body_bbgt_d = body_BBGT[jmax, :]

                    ixmin = np.maximum(body_bbgt_d[0], body_bb[0])
                    iymin = np.maximum(body_bbgt_d[1], body_bb[1])
                    ixmax = np.minimum(body_bbgt_d[2], body_bb[2])
                    iymax = np.minimum(body_bbgt_d[3], body_bb[3])
                    iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
                    ih = np.maximum(iymax - iymin + 1.0, 0.0)
                    inters = iw * ih

                    uni = (
                        (body_bb[2] - body_bb[0] + 1.0) * (body_bb[3] - body_bb[1] + 1.0)
                        + (body_bbgt_d[2] - body_bbgt_d[0] + 1.0) * (body_bbgt_d[3] - body_bbgt_d[1] + 1.0)
                        - inters
                        )

                    overlaps_body = inters / uni 
                    
                    if not single_metric:
                        tp[d] = 1.0
                        R["det"][jmax] = 1                    
                        if overlaps_body > 0.5:
                            body_acc_count += 1
                    else:
                        if overlaps_body > 0.5:
                            tp[d] = 1.0
                            R["det"][jmax] = 1 
                        else:
                            fp[d] = 1.0
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    if not single_metric:
        body_acc = (body_acc_count / max(tp)) * 100.0
        print("Body Accuracy corresponding to Dual Metric is:", round(body_acc, 4))
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap