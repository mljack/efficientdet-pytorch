import json
import time
import os
import sys
from ensemble_boxes import *
import torch
import numpy as np
import pandas as pd
from glob import glob
from torch.utils.data import Dataset,DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import gc
from matplotlib import pyplot as plt
from effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from effdet.efficientdet import HeadNet

import warnings
warnings.simplefilter("ignore")

def get_valid_transforms(image_scale):
    return A.Compose([
            #A.Resize(height=image_scale, width=image_scale, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def load_net(model_name, image_scale, checkpoint_path):
    config = get_efficientdet_config(model_name)
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = image_scale
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    del checkpoint
    gc.collect()

    net = DetBenchPredict(net, config)
    net.eval();
    return net.cuda()

def make_predictions(images, net, score_threshold=0.22):
    images = torch.stack(images).cuda().float()
    predictions = []
    with torch.no_grad():
        img_scale = torch.tensor([1]*images.shape[0]).float().cuda()
        img_size = [image.shape[-2:] for image in images]
        img_size = torch.tensor(img_size).float().cuda()
        det = net(images, img_scale, img_size)
        for i in range(images.shape[0]):
            boxes = det[i].detach().cpu().numpy()[:,:4]    
            scores = det[i].detach().cpu().numpy()[:,4]
            indexes = np.where(scores > score_threshold)[0]
            boxes = boxes[indexes]
            boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
            predictions.append({
                'boxes': boxes[indexes],
                'scores': scores[indexes],
            })
    return [predictions]

def run_wbf(predictions, image_index, image_size, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist()  for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist()  for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).tolist() for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    if 0:
        cv2.imwrite("rotated.png", image*255)
        cv2.namedWindow("input", cv2.WINDOW_NORMAL);
        cv2.setWindowProperty("input", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.imshow("input", image)
        k = cv2.waitKey()
        if k == 27:
            exit(0)

    return image, M

class DatasetRetriever(Dataset):
    def __init__(self, crop_size, overlap_size, image_scale, angle=0.0, path=None, img_bytes=None, np_img=None, transform=None):
        super(DatasetRetriever, self).__init__()
        if isinstance(path, torch._six.string_classes):
            path = os.path.expanduser(path)
        self.path = path
        self.transform = transform
        if img_bytes:
            #open("drone/test/abcd.jpg", "wb").write(img_bytes)
            #print(dir(img_bytes))
            #print(img_bytes.hex())
            #print(type(img_bytes), img_bytes.count())
            self.img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            #cv2.imshow("post", self.img)
            #cv2.waitKey()
        if path:
            self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if np_img is not None:
            self.img = np_img
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.img /= 255.0
        
        self.original_img = self.img.copy()
        self.img, self.angle_transform = rotate_im(self.img, angle)
        
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        self.input_size = image_scale
        self.count_w = (self.width - overlap_size * 2 + crop_size - 1) // (crop_size - overlap_size)
        self.count_h = (self.height - overlap_size * 2 + crop_size - 1) // (crop_size - overlap_size)
        self.idx_w = 0
        self.idx_h = 0
        self.step = crop_size - overlap_size
        self.bbox_scale = float(crop_size) / float(image_scale)
        
        
        #print(self.width, self.height)
        #print(self.count_w, self.count_h)

    def __getitem__(self, index):
        base_x = self.idx_w * self.step
        base_y = self.idx_h * self.step
        idx = self.idx_w+self.idx_h*self.count_w
        self.idx_w += 1
        if self.idx_w == self.count_w:
            self.idx_w = 0
            self.idx_h += 1
        
        crop_img = np.zeros((self.crop_size, self.crop_size,3), np.float32)
        crop_img2 = self.img[base_y:min(base_y+self.crop_size,self.height), base_x:min(base_x+self.crop_size,self.width)]
        crop_img[0:crop_img2.shape[0], 0:crop_img2.shape[1]] = crop_img2
        if self.input_size != self.crop_size:
            crop_img = cv2.resize(crop_img, (self.input_size, self.input_size))

        if 0:
            cv2.imshow("input", crop_img)
            k = cv2.waitKey()
            if k == 27:
                exit(0)
        
        if self.transform:
            image = self.transform(image=crop_img)['image']
        else:
            image = crop_img

        return image, index, base_x, base_y, idx

    def __len__(self):
        return self.count_w * self.count_h

def collate_fn(batch):
    return tuple(zip(*batch))

def predict(net, config, angle, json_path = None, img_path = None, img_bytes = None, np_img = None, delay = 1):
    dataset = DatasetRetriever(crop_size=config.crop_size, overlap_size=config.overlap_size,
        image_scale=config.image_scale, path=img_path, img_bytes=img_bytes, np_img=np_img,
        transform=get_valid_transforms(config.image_scale), angle=angle)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)

    if config.show_img:
        img2 = dataset.img
        #img2 = np_img if path is None else cv2.imread(path, cv2.IMREAD_COLOR)
        color = [(255, 0, 0), (0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255), (0,0,0)]

    results = []
    inv_m = np.linalg.inv(np.vstack((dataset.angle_transform, [0.0, 0.0, 1.0])))[0:2, :]
    for j, (images, image_ids, base_x, base_y, crop_idx) in enumerate(data_loader):
        predictions = make_predictions(images, net)
        for i, img in enumerate(images):
            img = img.permute(1,2,0).cpu().numpy()
            boxes, scores, labels = run_wbf(predictions, image_index=i, image_size=config.image_scale)
            for k, label in enumerate(labels):
                bbox = boxes[k].tolist()
                inside = True
                inside = inside and (base_x[i] == dataset.count_w - 1 or (bbox[0]+bbox[2])*0.5*dataset.bbox_scale < config.crop_size - config.overlap_size // 2)
                inside = inside and (base_y[i] == dataset.count_h - 1 or (bbox[1]+bbox[3])*0.5*dataset.bbox_scale < config.crop_size - config.overlap_size // 2)
                inside = inside and (base_x[i] == 0 or (bbox[0]+bbox[2])*0.5*dataset.bbox_scale > config.overlap_size // 2)
                inside = inside and (base_y[i] == 0 or (bbox[1]+bbox[3])*0.5*dataset.bbox_scale > config.overlap_size // 2)
                if inside:
                    box = [bbox[0]*dataset.bbox_scale+base_x[i], bbox[1]*dataset.bbox_scale+base_y[i],
                        bbox[2]*dataset.bbox_scale+base_x[i], bbox[3]*dataset.bbox_scale+base_y[i]]
                    box_pts = np.array([[[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]]])

                    polygon = cv2.transform(box_pts, inv_m)[0,:,:]
                    obj = {"label":float(label), "score":float(scores[k]), "box":box, "polygon":polygon.tolist()}
                    if config.show_img:
                        obj["crop_idx"] = crop_idx[i]
                    results.append(obj)
                img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 3)
                
            if 0:
                cv2.imshow("result", img)
                k = cv2.waitKey()
                if k == 27:
                    exit(0)
            
            if config.show_img and config.box_color is None:
                img2 = cv2.rectangle(img2, (base_x[i]+120, base_y[i]+120), (base_x[i]+dataset.crop_size-120, base_y[i]+dataset.crop_size-120), color[crop_idx[i]%len(color)], 3)

    if config.show_img:
        for obj in results:
            bbox = obj["box"]
            score = obj["score"]
            polygon = obj["polygon"]
            start_point = (int(bbox[0]), int(bbox[1])) 
            end_point = (int(bbox[2]), int(bbox[3])) 
            c = color[obj["crop_idx"]%len(color)] if config.box_color is None else config.box_color
            if 1:
                c = (0, 0, 255)         #    red: (0.8, 1.0)
                if score < 0.5:         #  green: (0.6, 0.8)
                    c = (0, 255, 255)   # yellow: (0.0, 0.6)
                if score < 0.7:
                    c = (0, 255, 0)
            if 1:
                img2 = cv2.rectangle(img2, start_point, end_point, c, 2)
            else:
                contour = np.array(polygon)
                img2 = cv2.drawContours(img2, [np.int0(contour)], 0, c, 2)
            
        if config.save_img:
            cv2.imwrite(config.img_name, img2)
        height, width, _ = img2.shape
        if width > 1920:
            cv2.namedWindow("result", cv2.WINDOW_NORMAL);
            cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.imshow("result", img2)
        k = cv2.waitKey(delay)
        if k == 27:
            exit(0)

    return results

def run(path, angles):
    if len(angles) == 0:
        angles = [0.0]
    else:
        angles = [float(a) for a in range(90)]
        #angles = [float(s) for s in angles]

    class Config:
        box_color = None
        show_img = True
        save_img = False
        img_name = "save_16.png"
        save_result = True
    config = Config()

    config.box_color = (0,0,255)
    #config.box_color = (0,255,0)
    #config.box_color = (0,255,255)

    if 0:
        config.crop_size = 512
        config.image_scale = 512
        config.overlap_size = 200
        config.batch_size = 64
        model_name = 'tf_efficientdet_d2'
        net = load_net(model_name, config.image_scale, 'effdet-d2-drone_003_512_1024_bs8_epoch32/best-checkpoint-005epoch.bin')
    if 0:
        config.crop_size = 896
        config.image_scale = 896
        config.overlap_size = 200
        config.batch_size = 8
        model_name = 'tf_efficientdet_d3'
        net = load_net(model_name, config.image_scale, 'effdet-d3-drone_004_896_1792_bs2_epoch6/best-checkpoint-000epoch.bin')
        #net = load_net(model_name, config.image_scale, 'effdet-d3-drone_004_896_1792_bs2_epoch6/last-checkpoint.bin')
    if 1:
        config.crop_size = 768
        config.image_scale = 768
        config.overlap_size = 200
        config.batch_size = 32
        model_name = 'tf_efficientdet_d2'
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_005_768_1536_bs4_epoch6/best-checkpoint-000epoch.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_005_768_1536_bs4_epoch6/last-checkpoint.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_006_768_1536_rotated_obb_no_cutout_bs2_epoch3/best-checkpoint-002epoch.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_007_768_1536_rotated_obb_no_cutout_more_bus_bs4_epoch4/best-checkpoint-003epoch.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_010_768_1536_rotated_obb_no_cutout_more_bus_tongji_bs4_epoch16/best-checkpoint-015epoch.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_010_768_1536_rotated_obb_no_cutout_more_bus_tongji_bs4_epoch16/best-checkpoint-005epoch.bin')
        #net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_012_768_1536_rotated_obb_no_cutout_more_bus_tong_more_color_gray_blur_aug_lr1e-4_bs4_epoch32/best-checkpoint-005epoch.bin')
        net = load_net(model_name, config.image_scale, '_models/effdet-d2-drone_013_768_1536_rotated_obb_no_cutout_more_bus_tong_changtai_jinqiao_colorjitter0.2_lr1e-4_bs4_epoch32/best-checkpoint-001epoch.bin')
        
    ext = path[path.rfind(".")+1:]
    if ext.lower() in ("png", "jpg", "jpeg", "bmp"):
        start = time.time()
        json_path = path[0:path.rfind(".")]+".vehicles.json" if config.save_result else None
        for angle in angles:
            result = predict(net, config, img_path=path, delay=0, angle=angle, json_path = json_path)
            if json_path is not None:
                with open(json_path.replace(".json", "_%02d.json" % int(angle)), "w") as f:
                    json.dump(result, f, indent=4)
        print(time.time() - start)
    elif ext.lower() in ("mpg", "mpeg", "mov", "mp4"):
        torch.backends.cudnn.benchmark = True
        video = cv2.VideoCapture(path)
        fps = video.get(cv2.CAP_PROP_FPS)
        print("FPS:\t\t%6.2f" % fps)
        print("Frame Count:\t", int(video.get(cv2.CAP_PROP_FRAME_COUNT)))
        base = path.replace("."+ext, "")+"_objs"
        if not os.path.isdir(base):
            os.mkdir(base)
        count = -1
        has_frame = True
        while has_frame:
            count += 1
            has_frame, img = video.read()
            if not has_frame:
                break
            #if count < 34*30 :
            #if count < 29*30 :
            #    continue
            start = time.time()
            json_path = os.path.join(base, "%05d.json" % count) if config.save_result else None
            for angle in angles:
                result = predict(net, config, np_img = img, angle = angle, json_path = json_path)
                if json_path is not None:
                    with open(json_path.replace(".json", "_%02d.json" % int(angle)), "w") as f:
                        json.dump(result, f, indent=4)
            print("[%05d]: Found %3d vehicles in %.3fs" % (count, len(result), time.time() - start))

if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        print("Select [%s]" % torch.cuda.get_device_name(torch.cuda.current_device()))

    if len(sys.argv) < 2:
        print("Usage: python infer.py test.jpg [0 30 60]")
    else:
        run(sys.argv[1], sys.argv[2:])


