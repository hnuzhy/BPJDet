
import os
import json
import copy
import shutil
import cv2

from tqdm import tqdm


def sort_labels_by_image_id(labels_list):
    images_labels_dict = {}
    for i, labels_dict in enumerate(labels_list):
        image_id = str(labels_dict['image_id'])
        if image_id in images_labels_dict.keys():
            images_labels_dict[image_id].append(labels_dict)
        else:
            images_labels_dict[image_id] = [labels_dict]
    return images_labels_dict

def get_full_new_annotations(anno_path, imgs_root, save_imgs, rm_ignore=False, debug=False):

    if os.path.exists(save_imgs):
        shutil.rmtree(save_imgs)
    os.mkdir(save_imgs)
    
    '''check_citypersons_annotation_integrity'''
    print("Processing annotations of CrowdHuman by BFJDet [%s]..."%(anno_path))

    anno_new_dict_person = {"type": "instances", "categories": [{"id": 1, "name": "person"}]}
    anno_new_dict_person['images'] = []
    anno_new_dict_person['annotations'] = []
    anno_new_dict_head = {"type": "instances", "categories": [{"id": 1, "name": "head"}]}
    anno_new_dict_head['images'] = []
    anno_new_dict_head['annotations'] = []
    anno_new_dict_face = {"type": "instances", "categories": [{"id": 1, "name": "face"}]}
    anno_new_dict_face['images'] = []
    anno_new_dict_face['annotations'] = []
    
    left_images_count = [0, 0, 0]  # for person, head and face

    anno_dict = json.load(open(anno_path, "r"))
    
    imgs_dict_list = anno_dict['images']
    annos_dict_list = anno_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(annos_dict_list)
    
    print("The original images/instances number in CrowdHuman(BFJDet): %d / %d"%(
        len(imgs_dict_list), len(annos_dict_list)))
    
    for imgs_dict in tqdm(imgs_dict_list):
        img_name = imgs_dict["file_name"]
        img_h, img_w = imgs_dict["height"], imgs_dict["width"]
        image_id = str(imgs_dict['id'])
        if image_id not in images_labels_dict:
            continue  # this image has no person instances
        
        img_path_src = os.path.join(imgs_root, img_name)
        assert os.path.exists(img_path_src), "orginal image missing :%s"%(img_path_src)
        img_name_new = img_name[:-4] + "_" + image_id + ".jpg"  # embed the image_id in new img_name
        imgs_dict["file_name"] = img_name_new  # the new img_name should also been updated in the imgs_dict
        img_path_dst = os.path.join(save_imgs, img_name_new)
        # shutil.copy(img_path_src, img_path_dst)
        os.system("ln -s %s %s"%(img_path_src, img_path_dst))  # save soft-link of source image path
        
        if debug:
            img = cv2.imread(img_path_src)
        
        anno_BFJDet_list = images_labels_dict[image_id]
        assert len(anno_BFJDet_list) != 0, "Each image has at least one anno by BFJDet! --> "+img_path_src
        ''' coco format of an instance_id in BFJDet
        anno_BFJDet_instance= {
            "segmentation": [], 
            "vbox": [959, 599, 38, 160],   # visible body [x,y,w,h] 
            "area": 6400, 
            "iscrowd": 0, 
            "height": 160, 
            "ignore": 0, 
            "image_id": 21, 
            "vis_ratio": 0.95, 
            "bbox": [959, 598, 40, 160],   # full body [x,y,w,h], x or y may be negative
            "category_id": 1, 
            "id": 567, 
            "h_bbox": [977, 600, 19, 21],  # head bbox [x,y,w,h], if not visible, the w and h will be 0
            "f_bbox": [978, 603, 11, 14]  # face bbox [x,y,w,h], if not visible, the w and h will be 0
        }
        '''
        left_BFJDet_list, left_head_instance_list, left_face_instance_list = [], [], []
        for anno_BFJDet_instance in anno_BFJDet_list:
        
            if rm_ignore and anno_BFJDet_instance["ignore"] == 1:
                # remove this person instance with "ignore" flag is 1 
                # This may be a pitcure of pitcure, a traffic sign or a super tiny person
                # print("[warning] ignore this instance:", img_name, anno_BFJDet_instance["id"])
                continue  
        
            instance_id = str(anno_BFJDet_instance["id"])
            p_bbox = anno_BFJDet_instance["bbox"]
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
            anno_BFJDet_instance["bbox"] = p_bbox
            anno_BFJDet_instance["area"] = round(p_bbox[-1] * p_bbox[-2], 4)
            anno_BFJDet_instance.pop("vbox", None)  # remove "vbox"
            anno_BFJDet_instance.pop("vis_ratio", None)  # remove "vis_ratio"
            
            h_bbox = anno_BFJDet_instance["h_bbox"]
            # illegal head bbox are labeled as [xxx, xxx, 1, 1] by BFJDet
            if h_bbox[2] == 1 and h_bbox[3] == 1:
                h_bbox = []  # update "h_bbox" with [] for our method
            else:
                if debug:
                    if h_bbox[0] < 0 or h_bbox[1] < 0 or h_bbox[0]+h_bbox[2] > img_w or h_bbox[1]+h_bbox[3] > img_h:
                        print("[warning] illegal <head> bbox!", h_bbox, p_bbox, img_w, img_h, instance_id, img_name)
                if h_bbox[0]+h_bbox[2] > img_w: h_bbox[2] = img_w - 1 - h_bbox[0]  # fix some illegal h_bbox in x1
                if h_bbox[1]+h_bbox[3] > img_h: h_bbox[3] = img_h - 1 - h_bbox[1]  # fix some illegal h_bbox in y1
                if h_bbox[0] < 0:  # fix some illegal h_bbox in x0
                    h_bbox[2] += h_bbox[0]
                    h_bbox[0] = 0
                if h_bbox[1] < 0:  # fix some illegal h_bbox in y1
                    h_bbox[3] += h_bbox[1]
                    h_bbox[1] = 0
                temp_instance_dict = copy.deepcopy(anno_BFJDet_instance)
                temp_instance_dict["bbox"] = h_bbox  # replace "bbox" with "h_bbox"
                temp_instance_dict["area"] = round(h_bbox[-1] * h_bbox[-2], 4)  # update "area"
                temp_instance_dict.pop("h_bbox", None)  # remove "h_bbox"
                temp_instance_dict.pop("f_bbox", None)  # remove "f_bbox"
                temp_instance_dict["area"] = round(h_bbox[-1] * h_bbox[-2], 4)  # update bbox area
                left_head_instance_list.append(temp_instance_dict)
            anno_BFJDet_instance["h_bbox"] = h_bbox
            
            f_bbox = anno_BFJDet_instance["f_bbox"]
            # illegal face bbox are labeled as [xxx, xxx, 1, 1] by BFJDet
            if f_bbox[2] == 1 and f_bbox[3] == 1:
                f_bbox = []  # update "f_bbox" with [] for our method
            else:
                if debug:
                    if f_bbox[0] < 0 or f_bbox[1] < 0 or f_bbox[0]+f_bbox[2] > img_w or f_bbox[1]+f_bbox[3] > img_h:
                        print("[warning] illegal <face> bbox!", f_bbox, p_bbox, img_w, img_h, instance_id, img_name)
                if f_bbox[0]+f_bbox[2] > img_w: f_bbox[2] = img_w - 1 - f_bbox[0]  # fix some illegal f_bbox in x1
                if f_bbox[1]+f_bbox[3] > img_h: f_bbox[3] = img_h - 1 - f_bbox[1]  # fix some illegal f_bbox in y1
                if f_bbox[0] < 0:  # fix some illegal f_bbox in x0
                    f_bbox[2] += f_bbox[0]
                    f_bbox[0] = 0
                if f_bbox[1] < 0:  # fix some illegal f_bbox in y1
                    f_bbox[3] += f_bbox[1]
                    f_bbox[1] = 0
                temp_instance_dict = copy.deepcopy(anno_BFJDet_instance)
                temp_instance_dict["bbox"] = f_bbox  # replace "bbox" with "f_bbox"
                temp_instance_dict["area"] = round(f_bbox[-1] * f_bbox[-2], 4)  # update "area"
                temp_instance_dict.pop("h_bbox", None)  # remove "h_bbox"
                temp_instance_dict.pop("f_bbox", None)  # remove "f_bbox"
                temp_instance_dict["area"] = round(f_bbox[-1] * f_bbox[-2], 4)  # update bbox area
                left_face_instance_list.append(temp_instance_dict)
            anno_BFJDet_instance["f_bbox"] = f_bbox
            
            left_BFJDet_list.append(anno_BFJDet_instance)
            
            
        if debug:
            if not os.path.exists("./debug_CrowdHuman/"):
                os.mkdir("./debug_CrowdHuman/")
            if len(os.listdir("./debug_CrowdHuman/")) >30:
                debug = False
            for anno_instance in anno_BFJDet_list:
                [x, y, w, h] = anno_instance["bbox"]
                h_bbox = anno_instance["h_bbox"]
                [hx, hy, hw, hh] = h_bbox if len(h_bbox) != 0 else [0, 0, 0, 0]
                f_bbox = anno_instance["f_bbox"]
                [fx, fy, fw, fh] = f_bbox if len(f_bbox) != 0 else [0, 0, 0, 0]
                if anno_instance["ignore"] == 0:
                    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)  # green
                    if hw != 0 and hh != 0:
                        cv2.rectangle(img, (int(hx), int(hy)), (int(hx+hw), int(hy+hh)), (0,255,255), 1)  # yellow
                    if fw != 0 and fh != 0:
                        cv2.rectangle(img, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (255,0,255), 1)  # magenta
                if anno_instance["ignore"] == 1 and (not rm_ignore):
                    cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,0,255), 2)  # red
            cv2.imwrite("./debug_CrowdHuman/"+img_name[:-4]+".jpg", img)
                
                
        anno_new_dict_person['images'].append(imgs_dict)  # images w/o person bboxes, we should also save it for cocoAPI
        if len(left_BFJDet_list) != 0:           
            left_images_count[0] += 1
            anno_new_dict_person['annotations'] += left_BFJDet_list

        anno_new_dict_head['images'].append(imgs_dict)  # images w/o head bboxes, we should also save it for cocoAPI
        if len(left_head_instance_list) != 0:
            left_images_count[1] += 1
            anno_new_dict_head['annotations'] += left_head_instance_list
            
        anno_new_dict_face['images'].append(imgs_dict)  # images w/o face bboxes, we should also save it for cocoAPI
        if len(left_face_instance_list) != 0:
            left_images_count[2] += 1
            anno_new_dict_face['annotations'] += left_face_instance_list
            
    print("The total images/instances number in new <person> annotation: %d / %d"%(
        left_images_count[0], len(anno_new_dict_person['annotations'])))
    print("The left images/instances number in new <head> annotation: %d / %d"%(
        left_images_count[1], len(anno_new_dict_head['annotations'])))
    print("The left images/instances number in new <face> annotation: %d / %d"%(
        left_images_count[2], len(anno_new_dict_face['annotations'])))
        
    return anno_new_dict_person, anno_new_dict_head, anno_new_dict_face


if __name__ == '__main__':
    
    dataset_root_path = "/datasdc/zhouhuayi/dataset/"
    
    imgs_root_train = os.path.join(dataset_root_path, "CrowdHuman/yolov5_format/images/train")  # 15000
    imgs_root_val = os.path.join(dataset_root_path, "CrowdHuman/yolov5_format/images/val")  # 4370
    anno_path_train = os.path.join(dataset_root_path, "CrowdHuman/BFJDet/instances_train_full_bhf_new.json")
    anno_path_val = os.path.join(dataset_root_path, "CrowdHuman/BFJDet/instances_val_full_bhf_new.json")

    save_imgs_train = os.path.join(dataset_root_path, "CrowdHuman/images/train")  # save soft-link of image path
    save_imgs_val = os.path.join(dataset_root_path, "CrowdHuman/images/val")  # save soft-link of image path
    
    save_anno_train = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_train_person.json")
    save_anno_val = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_val_person.json")
    save_anno_head_train = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_train_head.json")
    save_anno_head_val = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_val_head.json")
    save_anno_face_train = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_train_face.json")
    save_anno_face_val = os.path.join(dataset_root_path, "CrowdHuman/JointBodyPart/crowdhuman_coco_val_face.json")

    
    img_anno_dict_train, img_anno_dict_head_train, img_anno_dict_face_train = get_full_new_annotations(
        anno_path_train, imgs_root_train, save_imgs_train, rm_ignore=True, debug=False)
        # anno_path_train, imgs_root_train, save_imgs_train, rm_ignore=True, debug=True)  # checking bboxes
        # anno_path_train, imgs_root_train, save_imgs_train, rm_ignore=False, debug=True)  # checking ignore
    with open(save_anno_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_train, dst_ann_file)
    with open(save_anno_head_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_head_train, dst_ann_file)
    with open(save_anno_face_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_face_train, dst_ann_file)
    
    img_anno_dict_val, img_anno_dict_head_val, img_anno_dict_face_val = get_full_new_annotations(
        anno_path_val, imgs_root_val, save_imgs_val, rm_ignore=True, debug=False)
    with open(save_anno_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_val, dst_ann_file)
    with open(save_anno_head_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_head_val, dst_ann_file)
    with open(save_anno_face_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_face_val, dst_ann_file)
        
'''
Processing annotations of CrowdHuman by BFJDet [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_train_full_bhf_new.json]...
The original images/instances number in CrowdHuman(BFJDet): 15000 / 438792
100%|███████████████████████████████████████████████████████| 15000/15000 [10:48<00:00, 23.14it/s]
The total images/instances number in new <person> annotation: 15000 / 339565
The left images/instances number in new <head> annotation: 15000 / 339565
The left images/instances number in new <face> annotation: 14690 / 191575

Processing annotations of CrowdHuman by BFJDet [/datasdc/zhouhuayi/dataset/CrowdHuman/BFJDet/instances_val_full_bhf_new.json]...
The original images/instances number in CrowdHuman(BFJDet): 4370 / 127716
100%|████████████████████████████████████████████████████████| 4370/4370 [03:41<00:00, 19.73it/s]
The total images/instances number in new <person> annotation: 4370 / 99481
The left images/instances number in new <head> annotation: 4370 / 99481
The left images/instances number in new <face> annotation: 4282 / 57328

'''
    