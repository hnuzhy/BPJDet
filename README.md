# BPJDet
Codes for my paper "[Body-Part Joint Detection and Association via Extended Object Representation](https://arxiv.org/abs/2212.07652)"

<table>
<tr>
<td><img src="./materials/000522_mpii_test_BPJDet.gif" height="240"></td>
<td><img src="./materials/002376_mpii_test_BPJDet.gif" height="240"></td> 
</tr>
</table>

## Paper Abstract
> The detection of human body and its related parts (e.g., face, head or hands) have been intensively studied and greatly improved since the breakthrough of deep CNNs. However, most of these detectors are trained independently, making it a challenging task to associate detected body parts with people. This paper focuses on the problem of joint detection of human body and its corresponding parts. Specifically, we propose a novel extended object representation that integrates the center location offsets of body or its parts, and construct a dense single-stage anchor-based Body-Part Joint Detector (BPJDet). Body-part associations in BPJDet are embedded into the unified representation which contains both the semantic and geometric information. Therefore, BPJDet does not suffer from error-prone association post-matching, and has a better accuracy-speed trade-off. Furthermore, BPJDet can be seamlessly generalized to jointly detect any body part. To verify the effectiveness and superiority of our method, we conduct extensive experiments on the CityPersons, CrowdHuman and BodyHands datasets. The proposed BPJDet detector achieves state-of-the-art association performance on these three benchmarks while maintains high accuracy of detection.

## Table of contents
<!--ts-->
- [Illustrations](#illustrations)
- [Installation](#installation)
- [Dataset Preparing](#dataset-preparing)
  * [CityPersons](#citypersons)
  * [CrowdHuman](#crowdhuman)
  * [BodyHands](#bodyhands)
- [Training and Testing](#training-and-testing)
  * [Configs](#configs)
  * [Body-Face Task](#body-face-task)
  * [Body-Hand Task](#body-hand-task)
  * [Body-Head Task](#body-head-task)
- [Inference](#inference)
- [References](#references)
- [Citation](#citation)
<!--te-->


## Illustrations


## Installation

* **Environment:** Anaconda, Python3.8, PyTorch1.10.0(CUDA11.2), wandb
```bash
$ git clone https://github.com/hnuzhy/BPJDet.git
$ pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Codes are only evaluated on GTX3090 + CUDA11.2 + PyTorch1.10.0.
$ pip3 install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

## Dataset Preparing

### CityPersons

### CrowdHuman

### BodyHands

## Training and Testing

### Configs
* **Yaml:** Please refer these `./data/*.yaml` files to config your own .yaml file. Such as the `JointBP_BodyHands.yaml` file for body-hand joint detection task.

* **Pretrained weights:** For YOLOv5 weights, please download the version 5.0 that we have used. And put them under the `./weights/` folder
```
yolov5s6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s6.pt]
yolov5m6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m6.pt]
yolov5l6.pt [https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l6.pt]
```

### Body-Face Task

### Body-Hand Task

### Body-Head Task

## Inference

* For single image or multiple images under one folder using `./demos/image.py`
```bash
# single image. Taking body-head joint detection as an example.
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/CrowdHuman/273271,1e59400094ef5d82.jpg --device 3
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/COCO/000000567640.jpg --device 3
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/BodyHands/test_4507_1.jpg --device 3

# multiple images. Taking body-head joint detection as an example.
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/CrowdHuman/ --device 3
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/COCO/ --device 3
$ python demos/image.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --imgsz 1536 --conf-thres 0.45 --iou-thres 0.75 \
    --match-iou 0.6 --img-path test_imgs/BodyHands/ --device 3
```

* For single video using `./demos/video.py`. Taking body-head joint detection as an example.
```bash
# save as .mp4 file
$ python demos/video.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --video-path test_imgs/path/to/file.mp4 \
    --imgsz 1536 --conf-thres 0.25 --iou-thres 0.75 --match-iou 0.6 --device 3 \
    --start 0 --end -1 --thickness 2 --alpha 0.2 --save-size 540

# save as .gif file
$ python demos/video.py --weights runs/JointBP/ch_head_l_1536_e150_mMR/weights/best_mMR.pt \
    --data data/JointBP_CrowdHuman_head.yaml --video-path test_imgs/PoseTrack/002376_mpii_test.mp4 \
    --imgsz 1536 --conf-thres 0.25 --iou-thres 0.75 --match-iou 0.6 --device 3 \
    --start 0 --end -1 --thickness 2 --alpha 0.2 --gif --gif-size 640 360
```


## References

* [YOLOv5 🚀 in PyTorch > ONNX > CoreML > TFLite](https://github.com/ultralytics/yolov5)
* [ICCV 2021 (BFJDet) - Body-Face Joint Detection via Embedding and Head Hook](https://github.com/AibeeDetect/BFJDet)
* [CVPR 2022 (BodyHands) - Whose Hands Are These? Hand Detection and Hand-Body Association in the Wild](https://github.com/cvlab-stonybrook/BodyHands)
* We also thank public datasets [CityPersons](https://www.cityscapes-dataset.com/), [CrowdHuman](https://www.crowdhuman.org/) and [COCOPersons](https://cocodataset.org/) for their excellent works.


## Citation

If you use our works in your research, please cite with:
```
@article{zhou2022body,
  title={Body-Part Joint Detection and Association via Extended Object Representation},
  author={Zhou, Huayi and Jiang, Fei and Lu, Hongtao},
  journal={arXiv preprint arXiv:2212.07652},
  year={2022}
}
```
