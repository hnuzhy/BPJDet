
import os
import json
import copy
import shutil

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

def convert_to_get_val_bh(anno_path, anno_path_bf):

    print("Processing annotations of CityPersons by BFJDet [%s]..."%(anno_path))

    anno_bh_new_dict = {"categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "head"}]}
    anno_bh_new_dict['images'] = []
    anno_bh_new_dict['annotations'] = []

    anno_dict = json.load(open(anno_path, "r"))
    
    imgs_dict_list = anno_dict['images']
    annos_dict_list = anno_dict['annotations']
    images_labels_dict = sort_labels_by_image_id(annos_dict_list)

    print("The original images/instances number in CityPersons(BFJDet): %d / %d"%(
        len(imgs_dict_list), len(annos_dict_list)))

    instance_id, body_instance_cnt, head_instance_cnt = 0, 0, 0
    for imgs_dict in tqdm(imgs_dict_list):
        # img_name = imgs_dict["file_name"]
        # img_h, img_w = imgs_dict["height"], imgs_dict["width"]
        image_id = str(imgs_dict['id'])
        if image_id not in images_labels_dict:
            continue  # this image has no person instances

        anno_BFJDet_list = images_labels_dict[image_id]
        assert len(anno_BFJDet_list) != 0, "Each image has at least one anno by BFJDet! --> "+img_path_src
        ''' coco format of an instance_id in BFJDet
        anno_BFJDet_instance= {
            "bbox": bbox,  # format [x0, y0, w, h]
            "vbox": vbox,
            "h_bbox": h_bbox,  # format [x0, y0, w, h]
            "f_bbox": f_bbox,  # format [x0, y0, w, h]
            "height": int(),
            "vis_ratio": float(),
            "image_id": image_id,
            "id": instance_id,
            "category_id": 1,
            "ignore": int(),
            "iscrowd": 0,
            "segmentation": [],
            "area": round(bbox[-1] * bbox[-2], 4),
        }
        '''
        new_anno_list = []
        for anno_BFJDet_instance in anno_BFJDet_list:
            # body annotation. following the format in anno_path_bf
            new_anno_list.append({
                "area": anno_BFJDet_instance["area"], 
                "iscrowd": anno_BFJDet_instance["iscrowd"], 
                "ignore": anno_BFJDet_instance["ignore"], 
                "image_id": anno_BFJDet_instance["id"], 
                "bbox": anno_BFJDet_instance["bbox"],   # body bbox
                "category_id": 1,  # class index of body is 1
                "id": instance_id, 
                "height": anno_BFJDet_instance["height"], 
                "vis_ratio": round(anno_BFJDet_instance["vis_ratio"], 6)
            })
            instance_id += 1
            body_instance_cnt += 1
            
            h_bbox = anno_BFJDet_instance["h_bbox"]
            # illegal head bbox are labeled as [1, 1, 1, 1] by BFJDet
            if h_bbox[0] == 1 and h_bbox[1] == 1 and h_bbox[2] == 1 and h_bbox[3] == 1:
                continue
            else:
                # head annotation. following the format in anno_path_bf
                new_anno_list.append({
                    "area": round(h_bbox[2] * h_bbox[3], 6),  # new area of head bbox 
                    "iscrowd": anno_BFJDet_instance["iscrowd"], 
                    "ignore": anno_BFJDet_instance["ignore"], 
                    "image_id": anno_BFJDet_instance["id"], 
                    "bbox": anno_BFJDet_instance["h_bbox"],  # head bbox
                    "category_id": 2,  # class index of head is 2
                    "id": instance_id, 
                    "height": h_bbox[3],  # new height of head bbox
                    "vis_ratio": round(anno_BFJDet_instance["vis_ratio"], 6)
                })
                instance_id += 1
                head_instance_cnt += 1

        anno_bh_new_dict['images'].append(imgs_dict)
        anno_bh_new_dict['annotations'] += new_anno_list

    print("The total images/instances number in new <person> and <head> annotation: %d / %d"%(
        len(anno_bh_new_dict['images']), len(anno_bh_new_dict['annotations']) ))
        
    print("The <person> and <head> instance annotation number are [%d] and [%d]"%(
        body_instance_cnt, head_instance_cnt))
        
    return anno_bh_new_dict


if __name__ == '__main__':
    
    dataset_root_path = "/datasdc/zhouhuayi/dataset/"
    
    # already provided
    anno_path_val = os.path.join(dataset_root_path, "CityPersons/BFJDet/instances_val_bhfmatch_new.json")
    anno_path_val_bf = os.path.join(dataset_root_path, "CityPersons/BFJDet/instances_val_bf_new.json")
    
    # not provided
    anno_path_val_bh = os.path.join(dataset_root_path, "CityPersons/BFJDet/instances_val_bh_new.json")
    
    anno_val_bh_new_dict = convert_to_get_val_bh(anno_path_val, anno_path_val_bf)
    with open(anno_path_val_bh, "w") as dst_ann_file:
        json.dump(anno_val_bh_new_dict, dst_ann_file)

'''

Processing annotations of CityPersons by BFJDet [/datasdc/zhouhuayi/dataset/CityPersons/BFJDet/instances_val_bhfmatch_new.json]...
The original images/instances number in CityPersons(BFJDet): 500 / 5185
100%|████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 30406.73it/s]
The total images/instances number in new <person> and <head> annotation: 361 / 8585
The <person> and <head> instance annotation number are [5185] and [3400]

'''
    