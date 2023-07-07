import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
import time
import os
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # 输入图片或图片目录
    parser.add_argument("--img_path", type=str, default='TestImages', help="input image path or image directory")
    # 输出图片目录
    parser.add_argument("--save_dir", type=str, default='SegOut', help="output image directory")
    # model checkpoint
    parser.add_argument("--sam_checkpoint", type=str, default="model/sam_vit_h_4b8939.pth", help="sam model checkpoint")
    parser.add_argument("--yolo_checkpoint", type=str, default="model/yolov8x.pt", help="yolo model detection checkpoint")
    # gpu
    parser.add_argument("--device", type=str, default='cuda:0', help="cuda:[0,1,2,3,4] or cpu")
    return parser.parse_args()


def combine_det_masks(masks, w, h):
    '''
    update detection mask by adding a mask position which equals to 1
    returns:
    det_centers: 每个mask的重心
    '''
    masks1 = []
    det_mask = np.zeros((h, w), dtype=int)
    det_centers = []
    for i, mask in enumerate(masks):
        mask = np.asarray(mask.cpu().reshape(h, w))    # 如果是fast_sam需要去掉
        y_array, x_array = np.where(mask == 1)
        center = (x_array.mean(), y_array.mean())
        det_centers.append(center)
        for y, x in zip(y_array, x_array):
            det_mask[y, x] = 1
        masks1.append(mask)
    return det_mask, masks1, det_centers


def pxl_overlap(det_mask, mask):
    olp = 0
    cnt = 0
    y_array, x_array = np.where(mask == 1)
    for y, x in zip(y_array, x_array):
        if (det_mask[y, x] == 1):
            olp += 1    
    cnt = len(y_array)
    return olp / cnt, cnt - olp


def color_imgs(img, mask_new, mask):  
    grays = []
    for i in range(0, 256):
        g = i * 0.5
        grays.append(g)
    c_b = np.random.randint(0, 255) * 0.5
    c_g = np.random.randint(0, 255) * 0.5
    c_r = np.random.randint(0, 255) * 0.5

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

    y_array, x_array = np.where(mask == 1)
    # 生成一个3通道的mask
    # mask_new = np.stack((mask,)*3, axis=-1)
    for y, x in zip(y_array, x_array):
        img[y, x, 0] = grays_b[img[y, x, 0]]
        img[y, x, 1] = grays_g[img[y, x, 1]]
        img[y, x, 2] = grays_r[img[y, x, 2]]
        # mask上色
        mask_new[y, x, 0] = int(c_b) 
        mask_new[y, x, 1] = int(c_g)
        mask_new[y, x, 2] = int(c_r)
        
        

    return img, mask_new.astype('uint8')


def color_imgs2(img, sam_img, det_mask, mask, mask_new):

    grays = []
    for i in range(0, 256):
        g = i * 0.5
        grays.append(g)      
    c_b = np.random.randint(0, 255) * 0.5
    c_g = np.random.randint(0, 255) * 0.5
    c_r = np.random.randint(0, 255) * 0.5

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

    y_array, x_array = np.where(mask == 1)
    for y, x in zip(y_array, x_array):
        if (det_mask[y, x] == 0):
            img[y, x, 0] = grays_b[img[y, x, 0]]
            img[y, x, 1] = grays_g[img[y, x, 1]]
            img[y, x, 2] = grays_r[img[y, x, 2]]
            sam_img[y, x, 0] = grays_b[sam_img[y, x, 0]]
            sam_img[y, x, 1] = grays_g[sam_img[y, x, 1]]
            sam_img[y, x, 2] = grays_r[sam_img[y, x, 2]]
            mask_new[y, x, 0] = int(c_b) 
            mask_new[y, x, 1] = int(c_g)
            mask_new[y, x, 2] = int(c_r)
            # det_mask[y, x] = 1

    return img, sam_img, det_mask, mask_new


def sam_color_img(img, mask, alpha=0.5):
    c_b = np.random.randint(0, 255) * alpha
    c_g = np.random.randint(0, 255) * alpha
    c_r = np.random.randint(0, 255) * alpha
    c = np.einsum('ijk, ij->ijk', img, mask) * alpha
    d = np.einsum('ijk, ij->ijk', img, ~mask)
    e = np.einsum('ij, k->ijk', mask, np.array([c_b, c_g, c_r]))
    return (c+d+e).astype(np.uint8)


def cal_single_area(contour):
    return cv2.contourArea(contour)


def cal_mask_area(mask):
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(list(map(cal_single_area, contours)))


def sam_color_mask(img, masks):
    # sort masks by area
    result_masks = sorted(masks, key=(lambda x: cal_mask_area(x)), reverse=True)
    # color mask
    for mask in result_masks:
        img = sam_color_img(img, mask, alpha=0.5)
    return img


def get_size_ratio(det_boxes):
    det_boxes = det_boxes.xyxy.cpu().numpy()
    size = [(det_box[2]-det_box[0])*(det_box[3]-det_box[1]) for det_box in det_boxes]
    max_size = max(size)
    size_ratio = np.array(size)/max_size
    return size_ratio


def add_label(img, det_box, det_name, size_ratio, center):
    ''''
    该函数为单一的mask检测框添加标签, 在图片上进行标注
    '''
    fontFace = cv2.FONT_HERSHEY_TRIPLEX
    fontScale = min(1*size_ratio+0.5, 1)
    thickness = min(int(1*size_ratio+0.5), 1)
    # center = (int((det_box[0]+det_box[2])/2), int((det_box[1]+det_box[3])/2))
    center = (int(center[0]), int(center[1]))
    retval, baseLine = cv2.getTextSize(det_name, fontFace=fontFace, fontScale=fontScale, thickness=thickness) #(width,height),bottom
    color = [0, 0, 0]
    topleft = (int(center[0]-retval[0]/2), int(center[1]-(retval[1]+baseLine)/2))
    topright = (int(center[0]+retval[0]/2), int(center[1]+(retval[1]+baseLine)/2))
    # 左边界限制
    limit_x_left = det_box[0]
    if topleft[0] < limit_x_left:
        delta_x_left = int(limit_x_left - topleft[0])
        # 整体往右移动
        topleft = (topleft[0] + delta_x_left, topleft[1])
    # 右边界限制
    limit_x_right = det_box[2]
    if topright[0]>limit_x_right:
        delta_x_right = int(limit_x_right - topright[0])
        # 整体向左移动
        topright = (topright[0] - delta_x_right, topright[1])

    cv2.rectangle(img, topleft, topright, color, -1)
    cv2.putText(img, f"{det_name}", (topleft[0], topleft[1]+retval[1]), 
                fontScale=fontScale, 
                fontFace=fontFace,
                thickness=thickness,
                color=[255, 255, 255])   
    return img


def add_label_to_det_box(img, results, det_box, det_centers):
    """
    该函数在上色后的图片上对每个mask检测框添加label(类别)
    results: yolo检测出目标的结果
    det_box: results里面的检查框坐标
    det_centers:中心点为每个mask检测框的中心
    """
    names = results[0].names
    det_boxes = results[0].boxes
    det_cls = det_boxes.cls # yolo所有的类别
    # 目标检测框的类别
    det_names = [names[cls_ind.item()] for cls_ind in det_cls]
    # 目标检测的框的size
    size_ratios = get_size_ratio(det_boxes)
    for det_box, det_name, size_ratio, center in zip(det_boxes.xyxy.cpu().numpy(), det_names, size_ratios, det_centers):
        img = add_label(img, det_box, det_name, size_ratio, center)     
    return img 


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
    mask_generator = SamAutomaticMaskGenerator(sam)  
    # yolo
    model = YOLO(yolo_checkpoint)
    return model, predictor, mask_generator


def segment(model, predictor, mask_generator, img, device='cuda:0'):
    # yolo 
    yolo_start = time.time()
    results = model.predict(img, device=device)
    yolo_predict_det_time = time.time()-yolo_start
    det_boxes = results[0].boxes

    masks = []
    sam_start = time.time()
    if len(det_boxes) >= 1:
        predictor.set_image(img)
        transformed_boxes = predictor.transform.apply_boxes_torch(
            det_boxes.xyxy, img.shape[:2])
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
    sam_predict_det_mask_time = time.time()-sam_start
    det_box_and_mask_predict_time = yolo_predict_det_time + sam_predict_det_mask_time
    print(f'det box and mask predict time: {det_box_and_mask_predict_time:.2f}, \
          yolo predict det time:{yolo_predict_det_time:.2f}, \
          sam predict det mask time:{sam_predict_det_mask_time:.2f}')

    w = img.shape[1]
    h = img.shape[0]
    img_fill = img.copy()
    sam_img = img.copy()
    det_mask, masks1, det_centers = combine_det_masks(masks, w, h) 
    # 检测框的mask上色
    start1 = time.time()
    mask_new = np.zeros_like(img) # mask_new是只有对应mask的位置进行上色
    for mask in masks1:
        img, mask_new = color_imgs(img, mask_new, mask)
    print(mask_new.shape)
    # cv2.imwrite('mask_new.jpg', mask_new)
    det_box_color_time = time.time()-start1
    sam_start1 = time.time()
    masks2 = mask_generator.generate(img_fill)
    sam_predict_all_mask_time = time.time() - sam_start1
    print(f'sam predict all mask time:{sam_predict_all_mask_time:.2f}')
    masks3 = [np.asarray(mask['segmentation'].reshape(h, w)) for mask in masks2]
    masks2 = sorted(masks3, key=(lambda x: cal_mask_area(x)), reverse=True)
    masks4 = []
    # 检测框以外的sam部分大目标上色+同时sam对应的位置上色（保持同种颜色）
    start2 = time.time()
    for mask_img in masks2:
        olp_rate, pxl_cnt = pxl_overlap(det_mask, mask_img)
        if (olp_rate < 0.3 and pxl_cnt > 100):
            img, sam_img, det_mask, mask_new = color_imgs2(img, sam_img, det_mask, mask_img, mask_new)
            
        else:
            masks4.append(mask_img)
    cv2.imwrite('mask_new.jpg', mask_new)
    ours_and_sam_other_color_time = time.time()-start2
    # sam剩余部分的mask上色
    start3 = time.time()
    sam_img = sam_color_mask(sam_img, masks4)
    sam_remain_color_time = time.time()-start3
    total_color_time = det_box_color_time+ours_and_sam_other_color_time+sam_remain_color_time
    
    print(f'total color time:{total_color_time:.2f}, \
          det box color time:{det_box_color_time:.2f}, \
          ours and sam other color time:{ours_and_sam_other_color_time:.2f}, \
          sam remain color time:{sam_remain_color_time:.2f}')
    # 对上色后的img添加标签
    img = add_label_to_det_box(img, results, det_boxes, det_centers)

    return sam_img, img, mask_new

def process(img_path, save_dir, sam_checkpoint, yolo_checkpoint, device):
    # 判断img_path是文件还是目录
    if '.jpg' in img_path:     # 说明路径是单张图片的路径
        img_files = [img_path]
    else:                      # 如果是目录
        # 取该目录下的所有图片路径
        img_files = glob.glob(os.path.join(img_path, '*.jpg'))
    sam_dir = os.path.join(save_dir, 'sam')
    our_dir = os.path.join(save_dir, 'our')
    mask_dir = os.path.join(save_dir, 'mask')
    if not os.path.exists(sam_dir):
        os.makedirs(sam_dir, exist_ok=True)
    if not os.path.exists(our_dir):
        os.makedirs(our_dir, exist_ok=True)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir, exist_ok=True)
    for img_file in img_files:
        # 读取该图片
        start = time.time()
        img = cv2.imread(img_file)
        model, predictor, mask_generator = load_models(sam_checkpoint, yolo_checkpoint, device)
        sam_img, img, mask_img = segment(model, predictor, mask_generator, img, device)
        # 保存该图片
        img_name = os.path.basename(img_file)      # ****.jpg
        img_name = img_name.split('.')[0]
        our_img_name = img_name +'_our.jpg'
        sam_img_name = img_name +'_sam.jpg'
        mask_img_name = img_name +'_mask.jpg'
        cv2.imwrite(os.path.join(our_dir, our_img_name), img)  # ours
        cv2.imwrite(os.path.join(sam_dir, sam_img_name), sam_img) 
        cv2.imwrite(os.path.join(mask_dir, mask_img_name), mask_img)
        print(f'{img_name}处理结束,用时:{time.time()-start} ')   
    print(f'运行结束')
    
     
def main(args):
    process(args.img_path,
            args.save_dir,
            args.sam_checkpoint,
            args.yolo_checkpoint,
            args.device)
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
