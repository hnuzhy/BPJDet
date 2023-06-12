
import os
import json
import copy
import shutil
import cv2

from tqdm import tqdm

import xml.etree.ElementTree as ET


def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict


def get_full_new_annotations(imgs_root, anno_path, save_imgs, debug=False):

    if os.path.exists(save_imgs):
        shutil.rmtree(save_imgs)
    os.mkdir(save_imgs)
    
    '''check_annotation_integrity'''
    print("Processing annotations of HumanParts by Hier-R-CNN [%s]..."%(anno_path))
    
    # For body_MR/AP, headpart_MR/AP, body-head_mMR and person_mAP cals when training, following BFJDet format
    anno_new_dict_person = {"type": "instances", "categories": [{"id": 1, "name": "person"}]}
    anno_new_dict_person['images'] = []
    anno_new_dict_person['annotations'] = []

    # For parts_mAP cal when testing, following COCO-Det format
    anno_new_dict_head = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_head["categories"][0]["name"] = "head"
    anno_new_dict_face = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_face["categories"][0]["name"] = "face"
    anno_new_dict_lefthand = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_lefthand["categories"][0]["name"] = "lefthand"
    anno_new_dict_righthand = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_righthand["categories"][0]["name"] = "righthand"
    anno_new_dict_leftfoot = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_leftfoot["categories"][0]["name"] = "leftfoot"
    anno_new_dict_rightfoot = copy.deepcopy(anno_new_dict_person)
    anno_new_dict_rightfoot["categories"][0]["name"] = "rightfoot"


    anno_dict = json.load(open(anno_path, "r"))
    
    imgs_dict_list = anno_dict['images']
    annos_dict_list = anno_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(annos_dict_list)
    
    print("The original images/instances number in HumanParts(Hier-R-CNN): %d / %d"%(
        len(imgs_dict_list), len(annos_dict_list)))
    
    for imgs_dict in tqdm(imgs_dict_list):
        img_name = imgs_dict["file_name"]
        img_h, img_w = imgs_dict["height"], imgs_dict["width"]
        image_id = str(imgs_dict['id'])
        if image_id not in images_labels_dict:
            continue  # this image has no person instances
        
        img_path_src = os.path.join(imgs_root, img_name)
        assert os.path.exists(img_path_src), "original image missing :%s"%(img_path_src)
        img_name_new = img_name[:-4] + "_" + image_id + ".jpg"  # embed the image_id in new img_name
        imgs_dict["file_name"] = img_name_new  # the new img_name should also been updated in the imgs_dict
        img_path_dst = os.path.join(save_imgs, img_name_new)
        # shutil.copy(img_path_src, img_path_dst)
        os.system("ln -s %s %s"%(img_path_src, img_path_dst))  # save soft-link of source image path
        
        if debug:
            img = cv2.imread(img_path_src)
        
        anno_HierRCNN_list = images_labels_dict[image_id]
        assert len(anno_HierRCNN_list) != 0, "Each image has at least one anno by HierRCNN! --> "+img_path_src
        ''' coco format of an instance_id in HierRCNN
        anno_HierRCNN_instance = {
            "hier": [426, 157, 443, 178, 1, 426, 167, 431, 176, 1, 0, 0, 0, 0, 0, 412, 213, 420, 216, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 5*6, (head, face, lefthand, righthand, leftfoot, rightfoot), [x1,y1,x2,y2,v]
            "segmentation": [], 
            "difficult": 0, 
            "id": 1, 
            "bbox": [412.0, 157.0, 54.0, 139.0],  # full body [x,y,w,h], x or y may be negative
            "image_id": 1, 
            "iscrowd": 0, 
            "category_id": 1,  # person category
            "area": 7506.0
        }
        the original defination in HumanParts by HierRCNN
        "categories": [
            {"id": 1, "supercategory": "person", "name": "person"}, 
            {"id": 2, "supercategory": "head", "name": "head"}, 
            {"id": 3, "supercategory": "face", "name": "face"}, 
            {"id": 4, "supercategory": "lefthand", "name": "lefthand"}, 
            {"id": 5, "supercategory": "righthand", "name": "righthand"}, 
            {"id": 6, "supercategory": "leftfoot", "name": "leftfoot"}, 
            {"id": 7, "supercategory": "rightfoot", "name": "rightfoot"}
        ]      
        '''
        left_HierRCNN_list = []
        for anno_HierRCNN_instance in anno_HierRCNN_list:

            instance_id = anno_HierRCNN_instance["id"]
            p_bbox = anno_HierRCNN_instance["bbox"]
            if debug:
                if p_bbox[0] < 0 or p_bbox[1] < 0 or p_bbox[0]+p_bbox[2] > img_w or p_bbox[1]+p_bbox[3] > img_h:
                    print("[warning] illegal <person> bbox!", p_bbox, img_w, img_h, instance_id, img_name)
            if p_bbox[0]+p_bbox[2] > img_w: p_bbox[2] = img_w - 1 - p_bbox[0]  # fix some illegal p_bbox in x1
            if p_bbox[1]+p_bbox[3] > img_h: p_bbox[3] = img_h - 1 - p_bbox[1]  # fix some illegal p_bbox in y1
            if p_bbox[0] < 0:  # fix some illegal p_bbox in x0
                p_bbox[2] += p_bbox[0]
                p_bbox[0] = 0
            if p_bbox[1] < 0:  # fix some illegal p_bbox in y1
                p_bbox[3] += p_bbox[1]
                p_bbox[1] = 0
            anno_HierRCNN_instance["bbox"] = p_bbox
            anno_HierRCNN_instance["area"] = round(p_bbox[-1] * p_bbox[-2], 4)
            anno_HierRCNN_instance["ignore"] = 0  # must give this key !!!
            anno_HierRCNN_instance["height"] = p_bbox[-1]  # must give this key !!!
            anno_HierRCNN_instance["vis_ratio"] = 1.0  # must give this key !!!
            all_bbox_list = anno_HierRCNN_instance["hier"]
            if not debug:
                anno_HierRCNN_instance.pop("hier", None)  # remove "hier"
            
            dict_keys_list = ["h_bbox", "f_bbox", "lh_bbox", "rh_bbox", "lf_bbox", "rf_bbox"]
            for cls_ind in range(6):  # (head, face, lefthand, righthand, leftfoot, rightfoot)
                t_bbox = all_bbox_list[5*cls_ind:5*(cls_ind+1)]
                # illegal part bbox are labeled as [0,0,0,0,0] by HierRCNN
                if t_bbox[-1] == 0:
                    # update bodypart bbox "t_bbox" with [] for our BPJDet method
                    # t_bbox = []  # this will lead bug when cal body-part mMR
                    
                    # update bodypart bbox "t_bbox" with [x1,y1,1,1] for our BPJDet method following BFJDet
                    t_bbox = [p_bbox[0], p_bbox[1], 1,1]  # must give this key's value !!!
                    
                else:
                    t_bbox = [t_bbox[0], t_bbox[1], t_bbox[2]-t_bbox[0], t_bbox[3]-t_bbox[1]]  # (x1,y1,x2,y2) --> (x,y,w,h)
                    if debug:
                        if t_bbox[0] < 0 or t_bbox[1] < 0 or t_bbox[0]+t_bbox[2] > img_w or t_bbox[1]+t_bbox[3] > img_h:
                            print("[warning] illegal <bodypart> bbox!", t_bbox, p_bbox, img_w, img_h, instance_id, img_name)
                    if t_bbox[0]+t_bbox[2] > img_w: t_bbox[2] = img_w - 1 - t_bbox[0]  # fix some illegal t_bbox in x1
                    if t_bbox[1]+t_bbox[3] > img_h: t_bbox[3] = img_h - 1 - t_bbox[1]  # fix some illegal t_bbox in y1
                    if t_bbox[0] < 0:  # fix some illegal t_bbox in x0
                        t_bbox[2] += t_bbox[0]
                        t_bbox[0] = 0
                    if t_bbox[1] < 0:  # fix some illegal t_bbox in y1
                        t_bbox[3] += t_bbox[1]
                        t_bbox[1] = 0
                    
                    temp_instance_dict = {
                        "segmentation": [], 
                        "difficult": 0, 
                        "id": instance_id * 7 + (cls_ind+1),  # 1 person and 6 bodyparts
                        "bbox": t_bbox,  # full bodypart [x,y,w,h]
                        "image_id": anno_HierRCNN_instance["image_id"], 
                        "iscrowd": 0, 
                        "category_id": 1,  # bodypart category, 1~6 are all set as 1
                        "area": round(t_bbox[-1] * t_bbox[-2], 4)  # update "area"
                    }
                    if cls_ind == 0:  anno_new_dict_head['annotations'].append(temp_instance_dict)
                    if cls_ind == 1:  anno_new_dict_face['annotations'].append(temp_instance_dict)
                    if cls_ind == 2:  anno_new_dict_lefthand['annotations'].append(temp_instance_dict)
                    if cls_ind == 3:  anno_new_dict_righthand['annotations'].append(temp_instance_dict)
                    if cls_ind == 4:  anno_new_dict_leftfoot['annotations'].append(temp_instance_dict)
                    if cls_ind == 5:  anno_new_dict_rightfoot['annotations'].append(temp_instance_dict)

                anno_HierRCNN_instance[dict_keys_list[cls_ind]] = t_bbox
            left_HierRCNN_list.append(anno_HierRCNN_instance)
            
    
        if debug:
            if not os.path.exists("./debug_HumanParts/"):
                os.mkdir("./debug_HumanParts/")
            if len(os.listdir("./debug_HumanParts/")) > 50:
                debug = False
            # [green, red, cyan, yellow, magenta, blue]
            colors_list = [(0,255,0),(0,0,255),(255,255,0),(0,255,255),(255,0,255),(255,0,0)]  
            for anno_instance in anno_HierRCNN_list:
                [x, y, w, h] = anno_instance["bbox"]
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (128,128,128), 2)  # gray, person bbox
                all_bbox_list = anno_instance["hier"]
                for cls_ind in range(6):  # (head, face, lefthand, righthand, leftfoot, rightfoot)
                    t_bbox = all_bbox_list[5*cls_ind:5*(cls_ind+1)]
                    if t_bbox[-1] != 0:
                        [tx1, ty1, tx2, ty2, tv] = t_bbox
                        color = colors_list[cls_ind]
                        cv2.rectangle(img, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color, 1)
            cv2.imwrite("./debug_HumanParts/"+img_name[:-4]+".jpg", img)
                
                
        anno_new_dict_person['images'].append(imgs_dict)  # images w/o person bboxes, we should also save it for cocoAPI
        assert len(left_HierRCNN_list) != 0, "One given image in HumanParts should at least have one bbox label!!!"           
        anno_new_dict_person['annotations'] += left_HierRCNN_list

        anno_new_dict_head['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
        anno_new_dict_face['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
        anno_new_dict_lefthand['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
        anno_new_dict_righthand['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
        anno_new_dict_leftfoot['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
        anno_new_dict_rightfoot['images'].append(imgs_dict)  # images w/o bodypart bboxes, we should also save it for cocoAPI
            
            
    print("The total images/instances number in new <person> annotation: %d / %d"%(
        len(anno_new_dict_person['images']), len(anno_new_dict_person['annotations'])))
    print("The left instances number in new <part: head> annotation: %d"%(len(anno_new_dict_head['annotations'])))
    print("The left instances number in new <part: face> annotation: %d"%(len(anno_new_dict_face['annotations'])))
    print("The left instances number in new <part: lefthand> annotation: %d"%(len(anno_new_dict_lefthand['annotations'])))
    print("The left instances number in new <part: righthand> annotation: %d"%(len(anno_new_dict_righthand['annotations'])))
    print("The left instances number in new <part: leftfoot> annotation: %d"%(len(anno_new_dict_leftfoot['annotations'])))
    print("The left instances number in new <part: rightfoot> annotation: %d"%(len(anno_new_dict_rightfoot['annotations'])))
    
    parts_dict_list = [anno_new_dict_head, anno_new_dict_face, 
        anno_new_dict_lefthand, anno_new_dict_righthand,
        anno_new_dict_leftfoot, anno_new_dict_rightfoot]
        
    return anno_new_dict_person, parts_dict_list

 
if __name__ == '__main__':

    dataset_root_path = "/datasdc/zhouhuayi/dataset/coco/"
    
    imgs_root_train = os.path.join(dataset_root_path, "images/train2017")
    imgs_root_val = os.path.join(dataset_root_path, "images/val2017")
    anno_path_train = os.path.join(dataset_root_path, "annotations_HumanParts/person_humanparts_train2017.json")
    anno_path_val = os.path.join(dataset_root_path, "annotations_HumanParts/person_humanparts_val2017.json")
    
    if not os.path.exists(os.path.join(dataset_root_path, "JointBodyPart")):
        os.mkdir(os.path.join(dataset_root_path, "JointBodyPart"))
        os.mkdir(os.path.join(dataset_root_path, "JointBodyPart/images"))
        
    save_imgs_train = os.path.join(dataset_root_path, "JointBodyPart/images/train")  # save soft-link of image path
    save_imgs_val = os.path.join(dataset_root_path, "JointBodyPart/images/val")  # save soft-link of image path

    save_anno_train_h = os.path.join(dataset_root_path, "JointBodyPart/humanparts_coco_train_person.json")
    save_anno_val_h = os.path.join(dataset_root_path, "JointBodyPart/humanparts_coco_val_person.json")
    save_anno_train_p = os.path.join(dataset_root_path, "JointBodyPart/humanparts_coco_train_bodypart.json")
    save_anno_val_p = os.path.join(dataset_root_path, "JointBodyPart/humanparts_coco_val_bodypart.json")



    img_anno_dict_train, save_anno_dict_parts_list = get_full_new_annotations(
        imgs_root_train, anno_path_train, save_imgs_train, debug=False)
        # imgs_root_train, anno_path_train, save_imgs_train, debug=True)  # checking bboxes
    with open(save_anno_train_h, "w") as dst_ann_file:
        json.dump(img_anno_dict_train, dst_ann_file)
    for cls_ind, cls_name in enumerate(['head', 'face', 'lefthand', 'righthand', 'leftfoot', 'rightfoot']):
        with open(save_anno_train_p.replace("bodypart", cls_name), "w") as dst_ann_file:
            json.dump(save_anno_dict_parts_list[cls_ind], dst_ann_file)

        
    img_anno_dict_val, save_anno_dict_parts_list = get_full_new_annotations(
        imgs_root_val, anno_path_val, save_imgs_val, debug=False)
    with open(save_anno_val_h, "w") as dst_ann_file:
        json.dump(img_anno_dict_val, dst_ann_file)
    for cls_ind, cls_name in enumerate(['head', 'face', 'lefthand', 'righthand', 'leftfoot', 'rightfoot']):
        with open(save_anno_val_p.replace("bodypart", cls_name), "w") as dst_ann_file:
            json.dump(save_anno_dict_parts_list[cls_ind], dst_ann_file)
        
        
'''
Processing annotations of HumanParts by Hier-R-CNN [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_train2017.json]...
The original images/instances number in HumanParts(Hier-R-CNN): 64115 / 257306
100%|████████████████████████████████| 64115/64115 [41:45<00:00, 25.59it/s]
The total images/instances number in new <person> annotation: 64115 / 257306
The left instances number in new <part: head> annotation: 223049
The left instances number in new <part: face> annotation: 153195
The left instances number in new <part: lefthand> annotation: 96078
The left instances number in new <part: righthand> annotation: 100205
The left instances number in new <part: leftfoot> annotation: 77997
The left instances number in new <part: rightfoot> annotation: 77870

Processing annotations of HumanParts by Hier-R-CNN [/datasdc/zhouhuayi/dataset/coco/annotations_HumanParts/person_humanparts_val2017.json]...
The original images/instances number in HumanParts(Hier-R-CNN): 2693 / 10777
100%|████████████████████████████████| 2693/2693 [02:16<00:00, 19.68it/s]
The total images/instances number in new <person> annotation: 2693 / 10777
The left instances number in new <part: head> annotation: 9351
The left instances number in new <part: face> annotation: 6913
The left instances number in new <part: lefthand> annotation: 4222
The left instances number in new <part: righthand> annotation: 4324
The left instances number in new <part: leftfoot> annotation: 3134
The left instances number in new <part: rightfoot> annotation: 3100
'''
    