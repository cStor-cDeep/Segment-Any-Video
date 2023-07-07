import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
import os
import argparse
   
   
def parse_args():
    parser = argparse.ArgumentParser()
    # 输入视频路径
    parser.add_argument("--video_path", type=str, default='video.mp4', help="input video path")
    # 输出视频路径
    parser.add_argument("--save_path", type=str, default='video_test.mp4', help="output video path")
    # model checkpoint
    parser.add_argument("--sam_checkpoint", type=str, default="model/sam_vit_h_4b8939.pth", help="sam model checkpoint")
    parser.add_argument("--yolo_checkpoint", type=str, default="model/yolov8x.pt", help="yolo model detection checkpoint")
    # imgsize
    parser.add_argument("--imgsz", type=int, default=1920, help="yolo track image size")
    # gpu
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda:[0,1,2,3,4] or cpu")
    return parser.parse_args()

 
def color_imgs(img, masks, boxes, ids, colors):

    grays = []
    for i in range(0, 256):
        g = i * 0.5
        grays.append(g)

    imw = img.shape[1]
    imh = img.shape[0]

    for k, (box) in enumerate(boxes):
        color = colors[int(ids[k])]
        c_b = color[0] * 0.5
        c_g = color[1] * 0.5
        c_r = color[2] * 0.5

        grays_r = []
        for i in range(0, 256):
            r = int(grays[i] + c_r)
            grays_r.append(r)

        grays_g = []
        for i in range(0, 256):
            g = int(grays[i] + c_g)
            grays_g.append(g)

        grays_b = []
        for i in range(0, 256):
            b = int(grays[i] + c_b)
            grays_b.append(b)

        mask = np.asarray(masks[k].cpu().reshape(imh, imw))
        h_array, w_array = np.where(mask == 1)
        for h_ind, w_ind in zip(h_array, w_array):
            img[h_ind, w_ind, 0] = grays_b[img[h_ind, w_ind, 0]]  # grays[img[h_ind,w_ind,0]] + grays[color[0]]
            img[h_ind, w_ind, 1] = grays_g[img[h_ind, w_ind, 1]]  # grays[img[h_ind,w_ind,1]] + grays[color[1]]
            img[h_ind, w_ind, 2] = grays_r[img[h_ind, w_ind, 2]]
            
    return img


def get_size_ratio(det_boxes):
    det_boxes = det_boxes.xyxy.cpu().numpy()
    size = [(det_box[2]-det_box[0])*(det_box[3]-det_box[1]) for det_box in det_boxes]
    max_size = max(size)
    size_ratio = np.array(size)/max_size
    return size_ratio


def get_det_centers(masks, w, h):
    det_centers = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask.cpu().reshape(h, w))    # 如果是fast_sam需要去掉
        y_array, x_array = np.where(mask == 1)
        center = (x_array.mean(), y_array.mean())
        det_centers.append(center)
    return  det_centers


def load_models(sam_checkpoint='model/sam_vit_h_4b8939.pth', 
                yolo_checkpoint='model/yolov8x.pt', 
                device='cuda:0'):
    '''
    sam以及yolo需要的模型配置
    '''
    # sam model type
    sam_file_name = os.path.basename(sam_checkpoint)        # 'sam_vit_h_4b8939.pth'
    sam_model_name = os.path.splitext(sam_file_name)[0]     # 'sam_vit_h_4b8939'
    sam_name_list = sam_model_name.split("_")               # ['sam', 'vit', 'h', '4b8939']
    model_type = sam_name_list[1] + '_' + sam_name_list[2]  # 'vit_h'
    print(model_type)
    # sam
    sam_device = torch.device(device)
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=sam_device)
    predictor = SamPredictor(sam) 
    # yolo
    model = YOLO(yolo_checkpoint)
    return model, predictor

def add_id_and_cls(frame, box, id, det_name, size_ratio, center, id_colors):
    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), id_colors[id], 2)   # (0, 255, 0),
    cv2.putText(frame, f"Id {id}", (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, id_colors[id], 2,)
    # 计算文本的宽高，baseLine
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = min(1*size_ratio+0.5, 1)
    thickness = min(int(1*size_ratio+0.5), 1)
    center = (int(center[0]), int(center[1]))
    retval, baseLine = cv2.getTextSize(det_name, fontFace=fontFace, fontScale=fontScale, thickness=thickness) #(width,height),bottom
    color = [0, 0, 0]
    topleft = (int(center[0]-retval[0]/2), int(center[1]-(retval[1]+baseLine)/2))
    topright = (int(center[0]+retval[0]/2), int(center[1]+(retval[1]+baseLine)/2))
    # 左边界限制
    limit_x_left = box[0]
    if topleft[0] < limit_x_left:
        delta_x_left = int(limit_x_left - topleft[0])
        # 整体往右移动
        topleft = (topleft[0] + delta_x_left, topleft[1])
    # 右边界限制
    limit_x_right = box[2]
    if topright[0] > limit_x_right:
        delta_x_right = int(limit_x_right - topright[0])
        # 整体向左移动
        topright = (topright[0] - delta_x_right, topright[1])

    cv2.rectangle(frame, topleft, topright, color, -1)
    cv2.putText(frame, f"{det_name}", (topleft[0], topleft[1]+retval[1]), 
                fontScale=fontScale, 
                fontFace=fontFace,
                thickness=thickness,
                color=[255, 255, 255])
    return frame   


def video_track_seg(model, predictor, id_colors,
                    video_path='2.mp4', save_path="video_test.mp4", 
                    imgsz=1920, device='cuda:0'):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.track(frame, persist=True, imgsz=imgsz, device=device)
        names = results[0].names
        det_boxes = results[0].boxes
        det_cls = det_boxes.cls
        det_names = [names[cls_ind.item()] for cls_ind in det_cls]
        size_ratios = get_size_ratio(det_boxes)
        ids = det_boxes.id.cpu().numpy().astype(int)
        predictor.set_image(frame)
        input_boxes = torch.tensor(det_boxes.xyxy.cpu().numpy().astype(int), device=device, dtype=torch.int32)
        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, frame.shape[:2])
        masks, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, 
                                              boxes=transformed_boxes, multimask_output=False,)
        det_centers = get_det_centers(masks, w, h)
        # 上色
        frame = color_imgs(frame, masks, 
                           det_boxes.xyxy.cpu().numpy().astype(int), 
                           ids, id_colors)
        # 添加标签
        for box, id, det_name, size_ratio, center in zip(det_boxes.xyxy.cpu().numpy().astype(int), ids, 
                                                         det_names, size_ratios, det_centers):
            frame = add_id_and_cls(frame, box, id, det_name, size_ratio, center, id_colors)
        vid_writer.write(frame)
    vid_writer.release() 
    
    
def process(sam_checkpoint='model/sam_vit_h_4b8939.pth', 
            yolo_checkpoint='model/yolov8x.pt',
            video_path='2.mp4',
            save_path="video_test.mp4",
            imgsz=1920,
            device='cuda:0'):
    id_colors = []
    for k in range(0, 1000):
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        c = [r, g, b]
        id_colors.append(c)
    model, predictor = load_models(sam_checkpoint, 
                                   yolo_checkpoint, 
                                   device)
    video_track_seg(model, predictor, id_colors,
                    video_path, save_path, 
                    imgsz, device)
  
    
def main(args):
    process(args.sam_checkpoint,
            args.yolo_checkpoint,
            args.video_path,
            args.save_path,
            args.imgsz,
            args.device
            )

       
if __name__ == '__main__':
    args = parse_args()
    main(args)