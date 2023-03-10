import os, os.path as osp
import argparse
import numpy as np
import yaml
from tqdm import tqdm

from pycocotools.coco import COCO

def write_yolov5_labels(data):
    assert not osp.isdir(osp.join(data['path'], data['labels'])), \
        'Labels already generated. Remove or choose new name for labels.'

    splits = [osp.splitext(osp.split(data[s])[-1])[0] for s in ['train', 'val', 'test'] if s in data]
    annotations = [osp.join(data['path'], data['{}_annotations'.format(s)]) for s in ['train', 'val', 'test'] if s in data]
    test_split = [0 if s in ['train', 'val'] else 1 for s in ['train', 'val', 'test'] if s in data]
    img_txt_dir = osp.join(data['path'], data['labels'], 'img_txt')
    os.makedirs(img_txt_dir, exist_ok=True)

    for split, annot, is_test in zip(splits, annotations, test_split):
        img_txt_path = osp.join(img_txt_dir, '{}.txt'.format(split))
        labels_path = osp.join(data['path'], '{}/{}'.format(data['labels'], split))
        if not is_test:
            os.makedirs(labels_path, exist_ok=True)
        coco = COCO(annot)
        if not is_test:
            pbar = tqdm(coco.anns.keys(), total=len(coco.anns.keys()))
            pbar.desc = 'Writing {} labels to {}'.format(split, labels_path)
            for id in pbar:
                a = coco.anns[id]

                if a['image_id'] not in coco.imgs:
                    continue

                if 'train' in split and a['iscrowd']:
                    continue

                img_info = coco.imgs[a['image_id']]
                img_h, img_w = img_info['height'], img_info['width']
                x, y, w, h = a['bbox']
                xc, yc = x + w / 2, y + h / 2
                xc /= img_w
                yc /= img_h
                w /= img_w
                h /= img_h
                
                if data['dataset'] == "CityPersons" or data['dataset'] == "CrowdHuman":
                    if data['part_type'] == 'face':  # for the body-face joint detection task
                        part_bbox = a['f_bbox']
                    if data['part_type'] == 'head':
                        part_bbox = a['h_bbox']
                        
                    if len(part_bbox) != 0:
                        x_part, y_part, w_part, h_part = part_bbox
                        xc_part, yc_part = x_part + w_part / 2, y_part + h_part / 2
                        xc_part /= img_w
                        yc_part /= img_h
                        w_part /= img_w
                        h_part /= img_h
                        v_part = 1
                    else:  # not all body instances have their heads /faces part
                        xc_part, yc_part, w_part, h_part, v_part = 0, 0, 0, 0, 0
                    
                    yolov5_label_txt = '{}.txt'.format(osp.splitext(img_info['file_name'])[0])
                    with open(osp.join(labels_path, yolov5_label_txt), 'a') as f:
                        # body cls and its bbox annotation, pointing to face center (face may be invisible)
                        f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            0, xc, yc, w, h, xc_part, yc_part, v_part))
                        if v_part:
                            # body part (e.g., face) cls and its bbox annotation, pointing to body center (body is visible)
                            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                1, xc_part, yc_part, w_part, h_part, xc, yc, 1))
                            # f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {} {} {}\n'.format(
                                # 1, xc_part, yc_part, w_part, h_part, 0, 0, 0))
                
                if data['dataset'] == "BodyHands":
                    part1_bbox, part2_bbox = a['h1_bbox'], a['h2_bbox']
                    if len(part1_bbox) != 0:
                        x_part1, y_part1, w_part1, h_part1 = part1_bbox
                        xc_part1, yc_part1 = x_part1 + w_part1 / 2, y_part1 + h_part1 / 2
                        xc_part1 /= img_w
                        yc_part1 /= img_h
                        w_part1 /= img_w
                        h_part1 /= img_h
                        v_part1 = 1
                        if len(part2_bbox) != 0:
                            x_part2, y_part2, w_part2, h_part2 = part2_bbox
                            xc_part2, yc_part2 = x_part2 + w_part2 / 2, y_part2 + h_part2 / 2
                            xc_part2 /= img_w
                            yc_part2 /= img_h
                            w_part2 /= img_w
                            h_part2 /= img_h
                            v_part2 = 1
                        else:
                            xc_part2, yc_part2, w_part2, h_part2, v_part2 = 0, 0, 0, 0, 0
                    else:
                        xc_part1, yc_part1, w_part1, h_part1, v_part1 = 0, 0, 0, 0, 0
                        xc_part2, yc_part2, w_part2, h_part2, v_part2 = 0, 0, 0, 0, 0
                    
                    yolov5_label_txt = '{}.txt'.format(osp.splitext(img_info['file_name'])[0])
                    with open(osp.join(labels_path, yolov5_label_txt), 'a') as f:
                        # body cls and its bbox annotation, pointing to hand center (hand may be invisible)
                        f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                            0, xc, yc, w, h, xc_part1, yc_part1, v_part1, xc_part2, yc_part2, v_part2))
                        if v_part1:
                            # body part (e.g., hand) cls and its bbox annotation, pointing to body center (body is visible)
                            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                1, xc_part1, yc_part1, w_part1, h_part1, xc, yc, 1, 0, 0, 0))
                        if v_part2:
                            f.write('{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                                1, xc_part2, yc_part2, w_part2, h_part2, xc, yc, 1, 0, 0, 0))
                                # 2, xc_part2, yc_part2, w_part2, h_part2, xc, yc, 1, 0, 0, 0))  # BodyHands doesn't label L-R hand
                            
            pbar.close()

        with open(img_txt_path, 'w') as f:
            for img_info in coco.imgs.values():
                f.write(osp.join(data['path'], 'images', '{}'.format(split), img_info['file_name']) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/coco.yaml')
    args = parser.parse_args()

    assert osp.isfile(args.data), 'Data config file not found at {}'.format(args.data)

    with open(args.data, 'rb') as f:
        data = yaml.safe_load(f)
    write_yolov5_labels(data)