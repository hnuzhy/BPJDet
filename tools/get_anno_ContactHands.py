
import os
import json
import copy
import shutil
import cv2

from tqdm import tqdm

import xml.etree.ElementTree as ET


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

def get_full_new_annotations(imgs_root, anno_path, anno_path_aux, split_txt, save_imgs, debug=False):
        
    hand_iou_thre = 0.75
        
    if os.path.exists(save_imgs):
        shutil.rmtree(save_imgs)
    os.mkdir(save_imgs)
    
    '''check_citypersons_annotation_integrity'''
    print("Processing annotations of ContactHands [%s]..."%(split_txt))

    anno_new_dict_person = {"type": "instances", "categories": [{"id": 1, "name": "person"}]}
    anno_new_dict_person['images'] = []
    anno_new_dict_person['annotations'] = []
    anno_new_dict_hand = {"type": "instances", "categories": [{"id": 1, "name": "hand"}]}
    anno_new_dict_hand['images'] = []
    anno_new_dict_hand['annotations'] = []

    left_images_count = [0, 0]  # for person, hand
    lost_xmls_list = []
    hand_contact_states_cnt = [[0,0,0], [0,0,0], [0,0,0], [0,0,0]]
    
    image_names = open(split_txt, "r").readlines()
    annos_dict_list, total_body_cnt, totol_image_cnt = [], 0, 0
    for img_id, image_name in enumerate(image_names):
        image_name = image_name.strip()
        temp_anno_dict = {}
        
        img_anno_path = os.path.join(anno_path, image_name+".xml")  # the annotation of ContactHands
        tree = ET.parse(img_anno_path)
        root = tree.getroot()
        
        img_anno_path_aux = os.path.join(anno_path_aux, image_name+".xml")  # the annotation of BodyHands
        if os.path.exists(img_anno_path_aux):
            tree_aux = ET.parse(img_anno_path_aux)
            root_aux = tree_aux.getroot()
        else:
            lost_xmls_list.append(image_name)
            print("XML annotation lost in BodyHands!!! ", image_name)
            continue
        
        img_w = int(root.find('size').find('width').text)
        img_h = int(root.find('size').find('height').text)
        temp_anno_dict["width"] = img_w
        temp_anno_dict["height"] = img_h
        temp_anno_dict["file_name"] = image_name+".jpg"
        temp_anno_dict["id"] = img_id
        temp_anno_dict["anno"] = {}

        objects = root_aux.findall('object')  # the annotation of BodyHands
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

        objects = root.findall('object')  # the annotation of ContactHands
        for object in objects:
            ins_name = object.find('name').text  # only the "hand"
            assert ins_name == "hand", "all category in ContactHands should only be hand!!!" + image_name
            bndbox = object.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            contact_state_str = object.find('contact_state').text
            '''
            # The four numbers denote contact-state for four contact states: 
            # 0 denotes No, 1 denotes Yes and 2 denotes Unsure.
            # the order of contact states is [NC, SC, PC, OC]
            (1) No-Contact: the hand is not in contact with any object in the scene; 
            (2) Self-Contact: the hand is in contact with another body part of the same person; 
            (3) Other-Person-Contact: the hand is in contact with another person; and 
            (4) Object-Contact: the hand is holding or touching an object other than people.
            http://vision.cs.stonybrook.edu/~supreeth/ContactHands_data_website/
            '''
            contact_states = [int(contact_state_str.split(",")[i]) for i in range(4)]
            matched_flag = False  # search hands in BodyHands for on hand in ContactHands
            for body_id, body_part_list in temp_anno_dict["anno"].items():
                for index, cls_bbox in enumerate(body_part_list):  # may be 1 or 2 or 3 bboxes
                    [ins_name_aux, xmin_aux, ymin_aux, xmax_aux, ymax_aux] = cls_bbox[:5]
                    if ins_name_aux == "body":  # do not match body cls
                        temp_anno_dict["anno"][body_id][index] = cls_bbox[:5] + [-1,-1,-1,-1]
                    else:
                        bbox_aux = [xmin_aux, ymin_aux, xmax_aux, ymax_aux]
                        bbox = [xmin, ymin, xmax, ymax]
                        temp_iou = calculate_bbox_iou(bbox_aux, bbox)
                        if temp_iou > hand_iou_thre:
                            # temp_anno_dict["anno"][body_id][index] = cls_bbox[:5] + contact_states
                            temp_anno_dict["anno"][body_id][index] = ["hand"] + bbox + contact_states 
                            matched_flag = True  # matched hand instance
                        else:
                            continue
                if matched_flag == True:
                    break  # this bndbox of one hand in ContactHands has been matched
            assert matched_flag == True, "Each hand in ContactHands should be matched!!!" + image_name
            
        totol_image_cnt += 1
        total_body_cnt += len(temp_anno_dict["anno"])
        annos_dict_list.append(temp_anno_dict)
    print("The original images/instances number in BodyHands: %d / %d"%(
        totol_image_cnt, total_body_cnt))
    print("The lost XML annotation in BodyHands: ", len(lost_xmls_list), lost_xmls_list)
    
    body_id_cnt, hand_id_cnt = 0, 0
    not_labeled_instance_list = []
    for annos_dict in tqdm(annos_dict_list):
        img_name = annos_dict["file_name"]
        img_h, img_w = annos_dict["height"], annos_dict["width"]
        image_id = str(annos_dict['id'])
        
        imgs_dict = {}
        imgs_dict["height"], imgs_dict["width"] = img_h, img_w
        imgs_dict["id"] = int(image_id)
        
        img_path_src = os.path.join(imgs_root, img_name)
        assert os.path.exists(img_path_src), "original image missing :%s"%(img_path_src)
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
            "h1_state": [s1,s2,s3,s4],  # hand1 contact state, should be 0 or 1 or 2, format [NC, SC, PC, OC]
            "h2_state": [s1,s2,s3,s4],  # hand2 contact state, should be 0 or 1 or 2, format [NC, SC, PC, OC]
        }
        '''
        new_body_instance_list, new_hand_instance_list = [], []
        for body_id, body_hand_list in annos_dict["anno"].items():
        
            anno_instance_body = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id), "h1_bbox": [], "h2_bbox": [], "h1_state": [], "h2_state": [] }
            anno_instance_hand1 = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id) }
            anno_instance_hand2 = {"segmentation": [], "iscrowd": 0, "ignore": 0, "category_id": 1, 
                "image_id": int(image_id) }
            for body_hand in body_hand_list:  # "body" or "hand"
                if len(body_hand) == 5:
                    if body_hand[0] != "body":
                        # print("Not labeled hand instance by ContactHands dataset:", body_hand)
                        not_labeled_instance_list.append([img_name] + body_hand)
                        continue
                    else:
                        body_hand += [-1,-1,-1,-1]
                    
                [ins_name, xmin, ymin, xmax, ymax, state1, state2, state3, state4] = body_hand
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
                        anno_instance_hand1["state"] = [state1, state2, state3, state4]
                    else:
                        hand_id_cnt += 1
                        anno_instance_hand2["id"] = hand_id_cnt
                        anno_instance_hand2["bbox"] = [xmin, ymin, xmax-xmin, ymax-ymin]
                        anno_instance_hand2["area"] = (xmax-xmin)*(ymax-ymin)
                        anno_instance_hand2["state"] = [state1, state2, state3, state4]
            
            '''BodyHands does not label left-right hands, but we should distinguish between the two.
            We thus always set hand1 on the left-img-canvas and hand2 on the right-img-canvas'''
            if "bbox" in anno_instance_hand1 and "bbox" in anno_instance_hand2:
                if anno_instance_hand1["bbox"][0] > anno_instance_hand2["bbox"][0]:
                    temp_bbox = anno_instance_hand1["bbox"]
                    temp_area = anno_instance_hand1["area"]
                    temp_state = anno_instance_hand1["state"]
                    anno_instance_hand1["bbox"] = anno_instance_hand2["bbox"]
                    anno_instance_hand1["area"] = anno_instance_hand2["area"]
                    anno_instance_hand1["state"] = anno_instance_hand2["state"]
                    anno_instance_hand2["bbox"] = temp_bbox
                    anno_instance_hand2["area"] = temp_area
                    anno_instance_hand2["state"] = temp_state
                
            assert "bbox" in anno_instance_body, "Each body_id must have a body bbox!\n"+str(annos_dict)
            if "bbox" in anno_instance_hand1:
                anno_instance_body["h1_bbox"] = anno_instance_hand1["bbox"]  # update h1_bbox
                anno_instance_body["h1_state"] = anno_instance_hand1["state"]  # update h1_state
                new_hand_instance_list.append(anno_instance_hand1)
                for i in range(4):  # [NC, SC, PC, OC]
                    hand_contact_states_cnt[i][anno_instance_hand1["state"][i]] += 1
            if "bbox" in anno_instance_hand2:
                anno_instance_body["h2_bbox"] = anno_instance_hand2["bbox"]  # update h2_bbox
                anno_instance_body["h2_state"] = anno_instance_hand2["state"]  # update h2_state
                new_hand_instance_list.append(anno_instance_hand2)
                for i in range(4):  # [NC, SC, PC, OC]
                    hand_contact_states_cnt[i][anno_instance_hand2["state"][i]] += 1
            new_body_instance_list.append(anno_instance_body)
            
            
        if debug:
            if not os.path.exists("./debug_ContactHands/"):
                os.mkdir("./debug_ContactHands/")
            if len(os.listdir("./debug_ContactHands/")) > 30:
                debug = False
            for anno_instance in new_body_instance_list:
                [x, y, w, h] = anno_instance["bbox"]
                cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)  # green
                h_bbox = anno_instance["h1_bbox"]
                h_state = anno_instance["h1_state"]
                [hx, hy, hw, hh] = h_bbox if len(h_bbox) != 0 else [0, 0, 0, 0]
                f_bbox = anno_instance["h2_bbox"]
                f_state = anno_instance["h2_state"]
                [fx, fy, fw, fh] = f_bbox if len(f_bbox) != 0 else [0, 0, 0, 0]
                if hw != 0 and hh != 0:
                    cv2.rectangle(img, (int(hx), int(hy)), (int(hx+hw), int(hy+hh)), (0,255,255), 1)  # yellow
                    cv2.line(img, (int(x), int(y)), (int(hx), int(hy)), (255,255,0), 2)  # cyan
                    [NC, SC, PC, OC] = h_state
                    str_show = str(NC) + str(SC) + str(PC) + str(OC)
                    cv2.putText(img, str_show, (int(hx), int(hy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0,0,255), 1, cv2.LINE_AA)  # red
                if fw != 0 and fh != 0:
                    cv2.rectangle(img, (int(fx), int(fy)), (int(fx+fw), int(fy+fh)), (0,255,255), 1)  # yellow
                    cv2.line(img, (int(x), int(y)), (int(fx), int(fy)), (255,255,0), 2)  # cyan
                    [NC, SC, PC, OC] = f_state
                    str_show = str(NC) + str(SC) + str(PC) + str(OC)
                    cv2.putText(img, str_show, (int(fx), int(fy)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (0,0,255), 1, cv2.LINE_AA)  # red
            cv2.imwrite("./debug_ContactHands/"+img_name[:-4]+".jpg", img)


        anno_new_dict_person['images'].append(imgs_dict)  # images w/o person bboxes, we should also save it for cocoAPI
        if len(new_body_instance_list) != 0:           
            left_images_count[0] += 1
            anno_new_dict_person['annotations'] += new_body_instance_list

        anno_new_dict_hand['images'].append(imgs_dict)  # images w/o head bboxes, we should also save it for cocoAPI
        if len(new_hand_instance_list) != 0:
            left_images_count[1] += 1
            anno_new_dict_hand['annotations'] += new_hand_instance_list
    
    print("Hand instances in BodyHands but not labeled by ContactHands:", 
        len(not_labeled_instance_list), not_labeled_instance_list)
    print("hand_contact_states_cnt [NC, SC, PC, OC]:", hand_contact_states_cnt)
    
    print("The left images/instances number in new <person> annotation: %d / %d"%(
        left_images_count[0], len(anno_new_dict_person['annotations'])))
    print("The left images/instances number in new <hand> annotation: %d / %d"%(
        left_images_count[1], len(anno_new_dict_hand['annotations'])))

    return anno_new_dict_person, anno_new_dict_hand
    
    
if __name__ == '__main__':

    dataset_root_path = "/datasdc/zhouhuayi/dataset/ContactHands/"  # This dataset contians hand-contact states
    dataset_root_path_aux = "/datasdc/zhouhuayi/dataset/BodyHands/"  # body boxes are labeled by BodyHands 

    imgs_root = os.path.join(dataset_root_path, "JPEGImages")
    anno_path = os.path.join(dataset_root_path, "Annotations")
    train_split_txt = os.path.join(dataset_root_path, "ImageSets/Main/train.txt")
    val_split_txt = os.path.join(dataset_root_path, "ImageSets/Main/test.txt")
    
    # imgs_root_aux = os.path.join(dataset_root_path_aux, "VOC2007/JPEGImages")
    anno_path_aux = os.path.join(dataset_root_path_aux, "VOC2007/Annotations")
    # train_split_txt_aux = os.path.join(dataset_root_path_aux, "VOC2007/ImageSets/Main/train.txt")
    # val_split_txt_aux = os.path.join(dataset_root_path_aux, "VOC2007/ImageSets/Main/test.txt")
    
    if os.path.exists(os.path.join(dataset_root_path, "images")):
        shutil.rmtree(os.path.join(dataset_root_path, "images"))
    os.mkdir(os.path.join(dataset_root_path, "images"))
    save_imgs_train = os.path.join(dataset_root_path, "images/train")  # save soft-link of image path
    save_imgs_val = os.path.join(dataset_root_path, "images/val")  # save soft-link of image path
    
    if not os.path.exists(os.path.join(dataset_root_path, "JointBodyPart")):
        os.mkdir(os.path.join(dataset_root_path, "JointBodyPart"))
    save_anno_train = os.path.join(dataset_root_path, "JointBodyPart/bodyhandcontact_coco_train_person.json")
    save_anno_val = os.path.join(dataset_root_path, "JointBodyPart/bodyhandcontact_coco_val_person.json")
    save_anno_hand_train = os.path.join(dataset_root_path, "JointBodyPart/bodyhandcontact_coco_train_hand.json")
    save_anno_hand_val = os.path.join(dataset_root_path, "JointBodyPart/bodyhandcontact_coco_val_hand.json")


    img_anno_dict_train, img_anno_dict_hand_train = get_full_new_annotations(
        imgs_root, anno_path, anno_path_aux, train_split_txt, save_imgs_train, debug=False)
        # imgs_root, anno_path, anno_path_aux, train_split_txt, save_imgs_train, debug=True)  # checking bboxes
    with open(save_anno_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_train, dst_ann_file)
    with open(save_anno_hand_train, "w") as dst_ann_file:
        json.dump(img_anno_dict_hand_train, dst_ann_file)
    
    print("\n")
    
    img_anno_dict_val, img_anno_dict_hand_val = get_full_new_annotations(
        imgs_root, anno_path, anno_path_aux, val_split_txt, save_imgs_val, debug=False)
    with open(save_anno_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_val, dst_ann_file)
    with open(save_anno_hand_val, "w") as dst_ann_file:
        json.dump(img_anno_dict_hand_val, dst_ann_file)

'''
Processing annotations of ContactHands [/datasdc/zhouhuayi/dataset/ContactHands/ImageSets/Main/train.txt]...
The original images/instances number in BodyHands: 18861 / 56066
The lost XML annotation in BodyHands:  16 ['train_3910_festival_01321', 'train_4026_gathering_01446', 'train_2847_000000521819', 'train_0496_000000008571', 'train_3579_VOC2010_930', 'train_4441_students_00012', 'train_4443_students_00023', 'train_4272_sports_00065', 'train_2423_000000422155', 'train_2565_000000455486', 'train_4016_gathering_01128', 'train_4208_seminar_00932', 'train_3884_festival_00568', 'train_4289_sports_00265', 'train_3322_Poselet_378', 'train_0868_000000069270']
100%|██████████████████████████████████| 18861/18861 [05:14<00:00, 60.05it/s]
Hand instances in BodyHands but not labeled by ContactHands: 15 [['train_3804_dancing_00361.jpg', 'hand', 172, 559, 296, 684], ['train_3869_dancing_01847.jpg', 'hand', 252, 316, 309, 374], ['train_3928_friends_00204.jpg', 'hand', 153, 438, 202, 500], ['train_3916_friends_00016.jpg', 'hand', 299, 375, 398, 461], ['train_2756_000000499155.jpg', 'hand', 323, 293, 344, 322], ['train_3717_athletics_00711.jpg', 'hand', 138, 276, 181, 313], ['train_3717_athletics_00704.jpg', 'hand', 303, 316, 344, 356], ['train_0129_021343_008.jpg', 'hand', 238, -1, 437, 175], ['train_3579_VOC2010_929.jpg', 'hand', 377, 15, 432, 74], ['train_3402_VOC2007_476.jpg', 'hand', 268, 126, 287, 149], ['train_3246_Inria_201.jpg', 'hand', 202, 391, 223, 408], ['train_3807_dancing_00417.jpg', 'hand', 317, 253, 368, 318], ['train_3701_athletics_00521.jpg', 'hand', 371, 343, 416, 388], ['train_3852_dancing_01472.jpg', 'hand', 588, 170, 617, 209], ['train_4009_gathering_00968.jpg', 'hand', 294, 363, 346, 417]]
hand_contact_states_cnt [NC, SC, PC, OC]: [[34751, 16496, 645], [41795, 9396, 701], [48962, 2698, 232], [26296, 24994, 602]]
The left images/instances number in new <person> annotation: 18861 / 56066
The left images/instances number in new <hand> annotation: 18861 / 51892

Processing annotations of ContactHands [/datasdc/zhouhuayi/dataset/ContactHands/ImageSets/Main/test.txt]...
The original images/instances number in BodyHands: 1629 / 7048
The lost XML annotation in BodyHands:  0 []
100%|██████████████████████████████████| 1629/1629 [00:39<00:00, 41.55it/s]
Hand instances in BodyHands but not labeled by ContactHands: 1 [['test_4534_9.jpg', 'hand', 456, 680, 550, 775]]
hand_contact_states_cnt [NC, SC, PC, OC]: [[4329, 1345, 308], [4228, 1388, 366], [5773, 195, 14], [2866, 2976, 140]]
The left images/instances number in new <person> annotation: 1629 / 7048
The left images/instances number in new <hand> annotation: 1629 / 5982
'''
    