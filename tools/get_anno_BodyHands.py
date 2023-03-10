
import os
import json
import copy
import shutil
import cv2

from tqdm import tqdm

import xml.etree.ElementTree as ET


def get_full_new_annotations(imgs_root, anno_path, split_txt, save_imgs, debug=False):

    if os.path.exists(save_imgs):
        shutil.rmtree(save_imgs)
    os.mkdir(save_imgs)
    
    '''check_citypersons_annotation_integrity'''
    print("Processing annotations of BodyHands [%s]..."%(split_txt))

    anno_new_dict_person = {"type": "instances", "categories": [{"id": 1, "name": "person"}]}
    anno_new_dict_person['images'] = []
    anno_new_dict_person['annotations'] = []
    anno_new_dict_hand = {"type": "instances", "categories": [{"id": 1, "name": "hand"}]}
    anno_new_dict_hand['images'] = []
    anno_new_dict_hand['annotations'] = []

    left_images_count = [0, 0]  # for person, hand
    
    image_names = open(split_txt, "r").readlines()
    annos_dict_list, total_body_cnt = [], 0
    for img_id, image_name in enumerate(image_names):
        image_name = image_name.strip()
        temp_anno_dict = {}
    
        img_anno_path = os.path.join(anno_path, image_name+".xml")
        tree = ET.parse(img_anno_path)
        root = tree.getroot()
        
        img_w = int(root.find('size').find('width').text)
        img_h = int(root.find('size').find('height').text)
        temp_anno_dict["width"] = img_w
        temp_anno_dict["height"] = img_h
        temp_anno_dict["file_name"] = image_name+".jpg"
        temp_anno_dict["id"] = img_id
        temp_anno_dict["anno"] = {}
        
        objects = root.findall('object')
        for object in objects:
            ins_name = object.find('name').text  # "body" or "hand"
            body_id = int(object.find('body_id').text)
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            if body_id not in temp_anno_dict["anno"]:
                temp_anno_dict["anno"][body_id] = []
            temp_anno_dict["anno"][body_id].append([ins_name, xmin, ymin, xmax, ymax])
                
        total_body_cnt += len(temp_anno_dict["anno"])
        annos_dict_list.append(temp_anno_dict)
    print("The original images/instances number in BodyHands: %d / %d"%(
        len(image_names), total_body_cnt))
    
    body_id_cnt, hand_id_cnt = 0, 0
    for annos_dict in tqdm(annos_dict_list):
        img_name = annos_dict["file_name"]
        img_h, img_w = annos_dict["height"], annos_dict["width"]
        image_id = str(annos_dict['id'])
        
        imgs_dict = {}
        imgs_dict["height"], imgs_dict["width"] = img_h, img_w
        imgs_dict["id"] = int(image_id)
        
        img_path_src = os.path.join(imgs_root, img_name)
        assert os.path.exists(img_path_src), "orginal image missing :%s"%(img_path_src)
        img_name_new = img_name[:-4] + "_" + image_id + ".jpg"  # embed the image_id in new img_name
        imgs_dict["file_name"] = img_name_new  # the new img_name should also been updated in the imgs_dict
        img_path_dst = os.path.join(save_imgs, img_name_new)
        # shutil.copy(img_path_src, img_path_dst)
        os.system("ln -s %s %s"%(img_path_src, img_path_dst))  # save soft-link of source image path
        
        if debug:
            img = cv2.imread(img_path_src)
        
        '''
        anno_instance= {
            "segmentation": [], 
            "area": 6400, 
            "iscrowd": 0, 
            "ignore": 0, 
            "image_id": int(), 
            "bbox": [x,y,w,h],   # full body bbox
            "category_id": 1, 
            "id": int(), 
            "h1_bbox": [x,y,w,h],  # hand1 bbox, if not visible, the bbox will be []
            "h2_bbox": [x,y,w,h],  # hand2 bbox, if not visible, the bbox will be []
        }
        '''
        new_body_instance_list, new_hand_instance_list = [], []
        for body_id, body_hand_list in annos_dict["anno"].items():
        
            anno_instance_body = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id), "h1_bbox": [], "h2_bbox": [] }
            anno_instance_hand1 = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id) }
            anno_instance_hand2 = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id) }
            for body_hand in body_hand_list:  # "body" or "hand"
                [ins_name, xmin, ymin, xmax, ymax] = body_hand
                if ins_name == "body":
                    body_id_cnt += 1
                    anno_instance_body["id"] = body_id_cnt
                    anno_instance_body["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
                    anno_instance_body["area"] = (xmax-xmin)*(ymax-ymin)
                if ins_name == "hand":
                    if "bbox" not in anno_instance_hand1:
                        hand_id_cnt += 1
                        anno_instance_hand1["id"] = hand_id_cnt
                        anno_instance_hand1["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
                        anno_instance_hand1["area"] = (xmax-xmin)*(ymax-ymin)
                    else:
                        hand_id_cnt += 1
                        anno_instance_hand2["id"] = hand_id_cnt
                        anno_instance_hand2["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
                        anno_instance_hand2["area"] = (xmax-xmin)*(ymax-ymin)
            
            '''BodyHands does not label left-right hands, but we should distinguish between the two.
            We thus always set hand1 on the left-img-canvas and hand2 on the right-img-canvas'''
            if "bbox" in anno_instance_hand1 and "bbox" in anno_instance_hand2:
                if anno_instance_hand1["bbox"][0] > anno_instance_hand2["bbox"][0]:
                    temp_bbox = anno_instance_hand1["bbox"]
                    temp_area = anno_instance_hand1["area"]
                    anno_instance_hand1["bbox"] = anno_instance_hand2["bbox"]
                    anno_instance_hand1["area"] = anno_instance_hand2["area"]
                    anno_instance_hand2["bbox"] = temp_bbox
                    anno_instance_hand2["area"] = temp_area
                
            assert "bbox" in anno_instance_body, "Each body_id must have a body bbox!\n"+str(annos_dict)
            if "bbox" in anno_instance_hand1:
                anno_instance_body["h1_bbox"] = anno_instance_hand1["bbox"]  # update h1_bbox
                new_hand_instance_list.append(anno_instance_hand1)
            if "bbox" in anno_instance_hand2:
                anno_instance_body["h2_bbox"] = anno_instance_hand2["bbox"]  # update h2_bbox
                new_hand_instance_list.append(anno_instance_hand2)
            new_body_instance_list.append(anno_instance_body)
            
            
        if debug:
            if not os.path.exists("./debug_BodyHands/"):
                os.mkdir("./debug_BodyHands/")
            if len(os.listdir("./debug_BodyHands/")) > 30:
                debug = False
            for anno_instance in new_body_instance_list:
                [x, y, w, h] = anno_instance["bbox"]
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)  # green
                h_bbox = anno_instance["h1_bbox"]
                [hx, hy, hw, hh] = h_bbox if len(h_bbox) != 0 else [0, 0, 0, 0]
                f_bbox = anno_instance["h2_bbox"]
                [fx, fy, fw, fh] = f_bbox if len(f_bbox) != 0 else [0, 0, 0, 0]
                if hw != 0 and hh != 0:
                    cv2.rectangle(img, (int(hx), int(hy)), (int(hx+hw), int(hy+hh)), (0,255,255), 1)  # yellow
                    cv2.line(img, (int(x), int(y)), (int(hx), int(hy)), (255,255,0), 2)  # cyan
                if fw != 0 and fh != 0:
                    cv2.rectangle(img, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0,255,255), 1)  # yellow
                    cv2.line(img, (int(x), int(y)), (int(fx), int(fy)), (255,255,0), 2)  # cyan
            cv2.imwrite("./debug_BodyHands/"+img_name[:-4]+".jpg", img)


        anno_new_dict_person['images'].append(imgs_dict)  # images w/o person bboxes, we should also save it for cocoAPI
        if len(new_body_instance_list) != 0:           
            left_images_count[0] += 1
            anno_new_dict_person['annotations'] += new_body_instance_list

        anno_new_dict_hand['images'].append(imgs_dict)  # images w/o head bboxes, we should also save it for cocoAPI
        if len(new_hand_instance_list) != 0:
            left_images_count[1] += 1
            anno_new_dict_hand['annotations'] += new_hand_instance_list
    
    print("The total images/instances number in new <person> annotation: %d / %d"%(
        left_images_count[0], len(anno_new_dict_person['annotations'])))
    print("The left images/instances number in new <hand> annotation: %d / %d"%(
        left_images_count[1], len(anno_new_dict_hand['annotations'])))

    return anno_new_dict_person, anno_new_dict_hand
    
    
if __name__ == '__main__':

    dataset_root_path = "/datasdc/zhouhuayi/dataset/BodyHands/"
    
    imgs_root = os.path.join(dataset_root_path, "VOC2007/JPEGImages")
    anno_path = os.path.join(dataset_root_path, "VOC2007/Annotations")
    train_split_txt = os.path.join(dataset_root_path, "VOC2007/ImageSets/Main/train.txt")
    val_split_txt = os.path.join(dataset_root_path, "VOC2007/ImageSets/Main/test.txt")
    
    if os.path.exists(os.path.join(dataset_root_path, "images")):
        shutil.rmtree(os.path.join(dataset_root_path, "images"))
    os.mkdir(os.path.join(dataset_root_path, "images"))
    save_imgs_train = os.path.join(dataset_root_path, "images/train")  # save soft-link of image path
    save_imgs_val = os.path.join(dataset_root_path, "images/val")  # save soft-link of image path

    save_anno_train = os.path.join(dataset_root_path, "JointBodyPart/bodyhands_coco_train_person.json")
    save_anno_val = os.path.join(dataset_root_path, "JointBodyPart/bodyhands_coco_val_person.json")
    save_anno_hand_train = os.path.join(dataset_root_path, "JointBodyPart/bodyhands_coco_train_hand.json")
    save_anno_hand_val = os.path.join(dataset_root_path, "JointBodyPart/bodyhands_coco_val_hand.json")


    img_anno_dict_train, img_anno_dict_hand_train = get_full_new_annotations(
        # imgs_root, anno_path, train_split_txt, save_imgs_train, debug=False)
        imgs_root, anno_path, train_split_txt, save_imgs_train, debug=True)  # checking bboxes
    with open(save_anno_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_train, dst_ann_file)
    with open(save_anno_hand_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_hand_train, dst_ann_file)

    img_anno_dict_val, img_anno_dict_hand_val = get_full_new_annotations(
        imgs_root, anno_path, val_split_txt, save_imgs_val, debug=False)
    with open(save_anno_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_val, dst_ann_file)
    with open(save_anno_hand_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_hand_val, dst_ann_file)

'''
Processing annotations of BodyHands [/datasdc/zhouhuayi/dataset/BodyHands/VOC2007/ImageSets/Main/train.txt]...
The original images/instances number in BodyHands: 18858 / 56060
100%|██████████████████████████████████████████████| 18858/18858 [05:45<00:00, 54.59it/s]
The total images/instances number in new <person> annotation: 18858 / 56060
The left images/instances number in new <hand> annotation: 18858 / 51901
Processing annotations of BodyHands [/datasdc/zhouhuayi/dataset/BodyHands/VOC2007/ImageSets/Main/test.txt]...
The original images/instances number in BodyHands: 1629 / 7048
100%|██████████████████████████████████████████████| 1629/1629 [00:32<00:00, 50.69it/s]
The total images/instances number in new <person> annotation: 1629 / 7048
The left images/instances number in new <hand> annotation: 1629 / 5983
'''
    