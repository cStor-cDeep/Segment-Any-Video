[English](README.md) | [简体中文](README.zh-CN.md)

**Introduction**

- The Segment Anything Model(SAM) proposed by facebook has made a great influence in computer vision, as it is a fundamental step in many tasks, such as edge detection, face recognition and autonomous driving. However, there are some weakness in SAM: (1) it can't return the semantic information about the regions, (2) in some cases an instance(eg. a car) may be segmented to different parts, (3) the model can't process video data.
- In this repository, we implement a segmentation and a tracking method using YOLOv8 and SAM, it can fix the weakness, we name this method Segment Any Video(SAV).
- In seg.py, our segmentation method is implemented by providing the boxes from YOLOv8 detector as prompts to SAM, and the masks with no semantic info will also be returned, this is the biggest difference with SAM.
In track.py, we modified the code from [ultralytics/tracker/track.py](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/tracker/track.py) which sported ByteTrack and BoTSORT, then apply instance segmentation to all frames.

**Installation**
```bash
pip install ultralytics
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**Model CheckPoints**
- [ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)
- [ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
- [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
- [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt)
- [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt)
- [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt)

**Usage**
```bash
python seg.py --img_path TestImages --save_dir SegOut --sam_checkpoint model/sam_vit_h_4b8939.pth --yolo_checkpoint model/yolov8x.pt
```
or
```bash
python track.py --video_path video.mp4 --save_path video_test.mp4 sam_checkpoint model/sam_vit_h_4b8939.pth --yolo_checkpoint model/yolov8x.pt --imgsz 1920
```

**Image Segment Results**
<div align=center>
<img src="./pic/image_seg_results.png" width="100%" alt="input"/>
</div>
<div align=center>We can see from the above result that our method can segment the bus, car and train to an intact object semantically while SAM segments to different parts. </div>

**Video Track Result**

<div align=center>
<img src="./video/1.gif" width="100%" alt="track" />
</div>
<div align=center>Segment and track</div>

<div align=center>
<img src="./video/2.gif" width="100%" alt="track" />
</div>
<div align=center>Segment and track</div>

<div align=center>
<img src="./video/3.gif" width="100%" alt="track" />
</div>
<div align=center>Segment and track</div>


**Demo**

- Our online demo is [here](http://sav.cstor.cn). 
- Note: Considering that video segmentation is time-consuming, so we didn't integrate this method. If you are interested, you can git clone this repository and run on your GPU machine.

**TODO**
- [ ] Train YOLOv8 models with object365 dataset
    - [x] [YOLO8m.pt](https://pan.baidu.com/s/1Lhbl_ez5sCC81j-s6RvP6A)  extract code: 65ge
    - [ ] YOLO8n.pt
    - [ ] YOLO8s.pt
    - [ ] YOLO8l.pt
    - [ ] YOLO8x.pt

**License**
- This code is licensed under the [AGPL-3.0 License](./LICENSE).
