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

def get_valid_transforms():
    return A.Compose([
            A.Resize(height=image_scale, width=image_scale, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)

def load_net(checkpoint_path):
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

class DatasetRetriever(Dataset):
    def __init__(self, crop_size, overlap_size, path=None, img_bytes=None, np_img=None, transform=None):
        super(DatasetRetriever, self).__init__()
        if isinstance(path, torch._six.string_classes):
            path = os.path.expanduser(path)
        self.path = path
        self.transform = transform
        if img_bytes:
            self.img = cv2.imdecode(np.fromstring(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if path:
            self.img = cv2.imread(self.path, cv2.IMREAD_COLOR)
        if np_img is not None:
            self.img = np_img
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.img /= 255.0
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.crop_size = crop_size
        self.overlap_size = overlap_size
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

def predict(path = None, img_bytes = None, np_img = None, delay = 1):
    dataset = DatasetRetriever(image_scale, overlap_size, path=path, img_bytes=img_bytes, np_img=np_img, transform=get_valid_transforms())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, collate_fn=collate_fn)

    if show_img:
        img2 = np_img if path is None else cv2.imread(path, cv2.IMREAD_COLOR)
        color = [(255, 0, 0), (0, 255, 0),(0, 0, 255),(255, 255, 0),(255, 0, 255),(0, 255, 255), (0,0,0)]

    results = []
    for j, (images, image_ids, base_x, base_y, crop_idx) in enumerate(data_loader):
        predictions = make_predictions(images, net)
        for i, img in enumerate(images):
            img = img.permute(1,2,0).cpu().numpy()
            boxes, scores, labels = run_wbf(predictions, image_index=i, image_size=dataset.crop_size)
            for k, label in enumerate(labels):
                bbox = boxes[k].tolist()
                inside = True
                inside = inside and (base_x[i] == dataset.count_w - 1 or (bbox[0]+bbox[2])*0.5 < dataset.crop_size - dataset.overlap_size // 2)
                inside = inside and (base_y[i] == dataset.count_h - 1 or (bbox[1]+bbox[3])*0.5 < dataset.crop_size - dataset.overlap_size // 2)
                inside = inside and (base_x[i] == 0 or (bbox[0]+bbox[2])*0.5 > dataset.overlap_size // 2)
                inside = inside and (base_y[i] == 0 or (bbox[1]+bbox[3])*0.5 > dataset.overlap_size // 2)
                if inside:
                    box = [bbox[0]*dataset.bbox_scale+base_x[i], bbox[1]*dataset.bbox_scale+base_y[i],
                        bbox[2]*dataset.bbox_scale+base_x[i], bbox[3]*dataset.bbox_scale+base_y[i]]
                    obj = {"label":float(label), "score":float(scores[k]), "box":box}
                    if show_img:
                        obj["crop_idx"] = crop_idx[i]
                    results.append(obj)
            if show_img and box_color is None:
                img2 = cv2.rectangle(img2, (base_x[i]+120, base_y[i]+120), (base_x[i]+dataset.crop_size-120, base_y[i]+dataset.crop_size-120), color[crop_idx[i]%len(color)], 3)

    if show_img:
        for obj in results:
            bbox = obj["box"]
            start_point = (int(bbox[0]), int(bbox[1])) 
            end_point = (int(bbox[2]), int(bbox[3])) 
            c = color[obj["crop_idx"]%len(color)] if box_color is None else box_color
            img2 = cv2.rectangle(img2, start_point, end_point, c, 2)
        if save_img:
            cv2.imwrite(img_name, img2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL);
        cv2.setWindowProperty("result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.imshow("result", img2)
        k = cv2.waitKey(delay)
        #exit(0)
        if k == 27:
            exit(0)

    return results

box_color = None
show_img = True
save_img = False
img_name = "save_16.png"
save_result = False
#box_color = (0,0,255)
#box_color = (0,255,0)
#box_color = (0,255,255)

if 0:
    image_scale = 512
    overlap_size = 200
    batch_size = 64
    model_name = 'tf_efficientdet_d2'
    net = load_net('effdet-d2-drone_003_512_1024_bs8_epoch32/best-checkpoint-005epoch.bin')
if 0:
    image_scale = 896
    overlap_size = 200
    batch_size = 8
    model_name = 'tf_efficientdet_d3'
    net = load_net('effdet-d3-drone_004_896_1792_bs2_epoch6/best-checkpoint-000epoch.bin')
    #net = load_net('effdet-d3-drone_004_896_1792_bs2_epoch6/last-checkpoint.bin')
if 1:
    image_scale = 768
    overlap_size = 200
    batch_size = 32
    model_name = 'tf_efficientdet_d2'
    #net = load_net('effdet-d2-drone_005_768_1536_bs4_epoch6/best-checkpoint-000epoch.bin')
    #net = load_net('effdet-d2-drone_005_768_1536_bs4_epoch6/last-checkpoint.bin')
    #net = load_net('effdet-d2-drone_006_768_1536_rotated_obb_no_cutout_bs2_epoch3/best-checkpoint-002epoch.bin')
    #net = load_net('effdet-d2-drone_007_768_1536_rotated_obb_no_cutout_more_bus_bs4_epoch4/best-checkpoint-003epoch.bin')
    net = load_net('effdet-d2-drone_010_768_1536_rotated_obb_no_cutout_more_bus_tongji_bs4_epoch16/best-checkpoint-015epoch.bin')
    #net = load_net('effdet-d2-drone_010_768_1536_rotated_obb_no_cutout_more_bus_tongji_bs4_epoch16/best-checkpoint-005epoch.bin')
    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python infer.py test.jpg")
    else:
        path = sys.argv[1]
        ext = path[path.rfind(".")+1:]
        if ext.lower() in ("png", "jpg", "jpeg", "bmp"):
            start = time.time()
            result = predict(path, delay=0)
            print(time.time() - start)
            if save_result:
                with open(path[0:path.rfind(".")]+".vehicles.json", "w") as f:
                    #json.dump(result, f, indent=4)
                    json.dump(result, f)
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
                #if count < 150:
                #    continue
                start = time.time()
                result = predict(np_img = img)
                print("[%05d]: Found %3d vehicles in %.3fs" % (count, len(result), time.time() - start))
                #print(result)
                if save_result:
                    with open(os.path.join(base, "%05d.json" % count), "w") as f:
                        #json.dump(result, f, indent=4)
                        json.dump(result, f)

