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
from .effdet import get_efficientdet_config, EfficientDet, DetBenchPredict
from .effdet.efficientdet import HeadNet
import warnings
warnings.simplefilter("ignore")
import pprint
import pdb

DEBUG_MODE = False

def dbg_show_img(img, bbox=[]):
    for box in bbox:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,0,255), 3)
    cv2.imwrite('./out.png', img)

class VideoDataset(torch.utils.data.IterableDataset):
    def __init__(self, rotate_every_n_frames, rotate_every_n_degree, video_file, tile_size, pad_size, tile_content_size, mask_image, down_sample_interval):
        self.rotate_every_n_frames = rotate_every_n_frames
        self.rotate_every_n_degree = rotate_every_n_degree
        self.tile_size = tile_size
        self.tile_pad_size = pad_size
        self.tile_content_size = tile_content_size
        self.mask_image_file = mask_image
        self.down_sample_interval = down_sample_interval
        self.angle_num = int(90 / self.rotate_every_n_degree)
        
        self.video = cv2.VideoCapture(video_file)
        self.fps = self.video.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.init_tile_need_infer_table()
        
    def get_tile_img(self, padded_img, x_tile_idx, y_tile_idx):
        x = self.tile_content_size * x_tile_idx
        y = self.tile_content_size * y_tile_idx
        tile_img = padded_img[y: y + self.tile_size, x: x + self.tile_size]

        return tile_img.astype(np.float32)

    def rotate_img(self, image, angle):
        if (angle == 0):
            return image, np.identity(3)

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

        M = np.vstack((M, [0,0,1]))
        M = np.linalg.inv(M)

        return image, M 

    def init_tile_need_infer_table(self):
        if (self.mask_image_file):
            mask_image = cv2.imread(self.mask_image_file, cv2.IMREAD_GRAYSCALE)
            _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
            mask_image = cv2.resize(mask_image, (self.video_width, self.video_height))

        self.tile_need_infer_table = []

        sample_pts = np.zeros((self.tile_size, self.tile_size, 2))
        for y in range(0, self.tile_size):
            for x in range(0, self.tile_size):
                sample_pts[y, x] = [x, y]
        sample_pts = sample_pts.reshape((1, self.tile_size * self.tile_size, 2))
    
        self.valueable_tiles_num = []

        for angle_idx in range(self.angle_num):
            print('init_tile_need_infer_table angle_idx = %d' % angle_idx)

            sample_img = np.ones((self.video_height, self.video_width, 3)) * 255
            padded_rotated_img, x_tile_count, y_tile_count, inv_matrix = self.rotate_pad_img(sample_img, angle_idx * self.rotate_every_n_degree)

            tile_need_infer_table = np.zeros((y_tile_count, x_tile_count)).astype(np.bool)
            valueable_tiles_num = 0

            for x_tile_idx in range(0, x_tile_count):   
                for y_tile_idx in range(0, y_tile_count):
                    tile_img = self.get_tile_img(padded_rotated_img, x_tile_idx, y_tile_idx)
                    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2GRAY)
                    _, tile_img = cv2.threshold(tile_img, 127, 255, cv2.THRESH_BINARY)

                    tile_need_infer = (tile_img != 0).any()

                    if (tile_need_infer and self.mask_image_file):
                        pts = sample_pts.copy()

                        pts[:,:,1] += self.tile_content_size * y_tile_idx - self.tile_pad_size
                        pts[:,:,0] += self.tile_content_size * x_tile_idx - self.tile_pad_size

                        ori_pts = cv2.transform(pts, inv_matrix[0:2,:])[0,:,:].astype(np.int)

                        ori_pts = ori_pts[::self.down_sample_interval]
                        ori_pts = ori_pts[ori_pts[:,0] >= 0]
                        ori_pts = ori_pts[ori_pts[:,0] < self.video_width]
                        ori_pts = ori_pts[ori_pts[:,1] >= 0]
                        ori_pts = ori_pts[ori_pts[:,1] < self.video_height]
                        
                        pts_in_mask = [mask_image[int(pt[1]), int(pt[0])] for pt in ori_pts]
                        tile_need_infer = (np.array(pts_in_mask) != 0).any()

                    tile_need_infer_table[y_tile_idx, x_tile_idx] = tile_need_infer
                    if (tile_need_infer):
                        valueable_tiles_num += 1

            self.tile_need_infer_table.append(tile_need_infer_table)
            self.valueable_tiles_num.append(valueable_tiles_num)

        rotated_frame_tile_count = sum(self.valueable_tiles_num)
        normal_frame_tile_count = self.valueable_tiles_num[0]

        self.tile_count = 0
        for i in range(self.frame_count):
            c = rotated_frame_tile_count if (i % self.rotate_every_n_frames == 0) else normal_frame_tile_count
            self.tile_count += c

    def __iter__(self):      
        frame_idx = -1

        while True:
            succ, ori_frame_img = self.video.read()
            frame_idx += 1
            if (not succ):
                return StopIteration

            angle_num = self.angle_num if (frame_idx % self.rotate_every_n_frames == 0) else 1
            
            for angle_idx in range(angle_num):
                rotated_padded_frame_img, x_tile_count, y_tile_count, inv_matrix = self.rotate_pad_img(ori_frame_img, angle_idx * self.rotate_every_n_degree)
                for y in range(y_tile_count):
                    for x in range(x_tile_count):
                        if (not self.tile_need_infer_table[angle_idx][y, x]):
                            continue

                        tile_img = self.get_tile_img(rotated_padded_frame_img, x, y)
                        model_input_image = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        model_input_image = A.Compose([ToTensorV2(p=1.0)], p=1.0)(image = model_input_image)['image']

                        yield {
                            'frame_idx': frame_idx,
                            'angle_idx': angle_idx,
                            'inv_matrix': inv_matrix,
                            'rotated_padded_frame_img': rotated_padded_frame_img,
                            'ori_frame_img': ori_frame_img,
                            'x_tile_count': x_tile_count,
                            'y_tile_count': y_tile_count,
                            'model_input_image': model_input_image,
                            'tile_np_img': tile_img,
                            'x_tile_idx': x,
                            'y_tile_idx': y,
                            'size': self.tile_size
                        }

    def rotate_pad_img(self, img, angle):        
        img, inv_matrix = self.rotate_img(img, angle)

        height, width = img.shape[:2]

        x_tile_count = (width + self.tile_content_size - 1) // self.tile_content_size
        width = x_tile_count * self.tile_content_size + self.tile_pad_size * 2
        
        y_tile_count = (height + self.tile_content_size - 1) // self.tile_content_size
        height = y_tile_count * self.tile_content_size + self.tile_pad_size * 2

        padded_img = np.zeros((height, width, img.shape[2]))
        padded_img[self.tile_pad_size: self.tile_pad_size + img.shape[0], self.tile_pad_size: self.tile_pad_size + img.shape[1], :] = img
 
        return padded_img, x_tile_count, y_tile_count, inv_matrix

class VideoDetection():
    def __init__(self, video_file, mask_image_file, out_json_path):
        self.video_file = video_file
        self.mask_image_file = mask_image_file

        self.tile_size = 768
        self.net_input_size = 768
        self.tile_pad_size = 100
        self.tile_content_size = self.tile_size - self.tile_pad_size * 2
        self.batch_size = 32
        self.rotate_every_n_frames = 20
        self.rotate_every_n_degree = 5
        self.down_sample_interval = 250
        self.score_threshold = 0.22
        self.iou_thr = 0.55
        self.skip_box_thr = 0.1
        self.out_json_path = out_json_path
        self.model_name = 'tf_efficientdet_d2'
        self.net = self.load_net(self.model_name, self.net_input_size, 1,
            '_models/model-023-best-checkpoint-000epoch.bin')

        self.dataset = VideoDataset(self.rotate_every_n_frames, self.rotate_every_n_degree, self.video_file, 
            self.tile_size, self.tile_pad_size, self.tile_content_size, self.mask_image_file, self.down_sample_interval)

        self.data_loader = DataLoader(self.dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=lambda x: x)


    def load_net(self, model_name, net_input_size, num_classes, checkpoint_path):
        config = get_efficientdet_config(model_name)
        net = EfficientDet(config, pretrained_backbone=False)

        config.num_classes = num_classes
        config.image_size = net_input_size
        net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

        checkpoint_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])

        del checkpoint
        gc.collect()

        net = DetBenchPredict(net, config)
        net.eval()
        return net.cuda()

    def run(self, progress_callback = None):
        cur_frame_idx = 0
        cur_angle_idx = 0
        cur_tile_count = 0

        result = []

        for tile_img_batch in self.data_loader:
            model_input_imgs = [tile_img_info['model_input_image'] for tile_img_info in tile_img_batch]
            model_input_imgs = torch.stack(model_input_imgs).cuda().float()            

            with torch.no_grad():
                model_input_img_scale = torch.tensor([1] * model_input_imgs.shape[0]).float().cuda()
                model_input_img_size = [img.shape[-2:] for img in model_input_imgs]
                model_input_img_size = torch.tensor(model_input_img_size).float().cuda()
                
                det = self.net(model_input_imgs, model_input_img_scale, model_input_img_size)

                for i, tile_img_info in enumerate(tile_img_batch):
                    d = det[i].detach().cpu().numpy()
                    tile_img_info['box'] = d[:,:4]
                    tile_img_info['score'] = d[:,4]
                    tile_img_info['class'] = d[:,5]

                    tile_img_info['box'][:, 2] = tile_img_info['box'][:, 2] + tile_img_info['box'][:, 0]
                    tile_img_info['box'][:, 3] = tile_img_info['box'][:, 3] + tile_img_info['box'][:, 1]
                    tile_img_info['box'] = tile_img_info['box'] / tile_img_info['size']

                    if (tile_img_info['frame_idx'] == cur_frame_idx and tile_img_info['angle_idx'] == cur_angle_idx):
                        result.append(tile_img_info)
                    else:
                        self.process_frame_result(result)
                        result.clear()
                        cur_frame_idx = tile_img_info['frame_idx']
                        cur_angle_idx = tile_img_info['angle_idx']
                        result.append(tile_img_info)

            
            cur_tile_count += len(tile_img_batch)
            p = cur_tile_count / self.dataset.tile_count
            print('processing %f' % p)
            if (progress_callback):
                progress_callback(p)

        self.process_frame_result(result)

    def dbg_show_final_result(self, final_result, angle_idx, rotated_padded_frame_img, ori_frame_img):
        tile_need_infer_table = self.dataset.tile_need_infer_table[angle_idx]
       
        # show in rotated frame image
        rotated_padded_frame_img = rotated_padded_frame_img.copy()
        polygon_in_rotated_img = [r['polygon_in_rotated_img'] for r in final_result]
        contour = np.array(polygon_in_rotated_img)
        img = cv2.drawContours(rotated_padded_frame_img, np.int0(contour), -1, (255,0,0), 2)

        for x in range(tile_need_infer_table.shape[1]):
            for y in range(tile_need_infer_table.shape[0]):
                color = (0, 255, 0) if tile_need_infer_table[y,x] else (255, 0, 0)
                
                offset = 2
                left = self.tile_content_size * x + self.tile_pad_size + offset
                top = self.tile_content_size * y + self.tile_pad_size + offset
                right = left + self.tile_content_size
                down = top + self.tile_content_size

                img = cv2.rectangle(img, (left, top), (right, down), color, 3)

        dbg_show_img(img)

        pdb.set_trace()

        # show in original frame image
        ori_frame_img = ori_frame_img.copy()
        polygon_in_ori_img = [r['polygon_in_ori_img'] for r in final_result]
        contour = np.array(polygon_in_ori_img)
        img = cv2.drawContours(ori_frame_img, np.int0(contour), -1, (255,0,0), 2)
        dbg_show_img(img)
     
    def process_frame_result(self, frame_result):
        final_result = []
        dbg_result = []

        for tile_img_info in frame_result:
            index = tile_img_info['score'] > self.score_threshold
            if (not index.any()):
                continue

            boxes = tile_img_info['box'][index].tolist()
            scores = tile_img_info['score'][index].tolist()
            labels = tile_img_info['class'][index].tolist()

            boxes, scores, labels = weighted_boxes_fusion([boxes], [scores], [labels], weights=None, iou_thr=self.iou_thr, skip_box_thr=self.skip_box_thr)
            boxes = boxes * (self.tile_size -1)

            for i, box in enumerate(boxes):
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0

                if (cx < self.tile_pad_size or cx > self.tile_size - self.tile_pad_size or cy < self.tile_pad_size or cy > self.tile_size - self.tile_pad_size):
                    continue

                box[0] += tile_img_info['x_tile_idx'] * self.tile_content_size
                box[2] += tile_img_info['x_tile_idx'] * self.tile_content_size
                box[1] += tile_img_info['y_tile_idx'] * self.tile_content_size
                box[3] += tile_img_info['y_tile_idx'] * self.tile_content_size

                box_pts = np.array([[[box[0], box[1]], [box[0], box[3]], [box[2], box[3]], [box[2], box[1]]]])
                polygon_in_rotated_img = box_pts[0]

                box_pts -= self.tile_pad_size
                polygon_in_ori_img = cv2.transform(box_pts, tile_img_info['inv_matrix'][0:2,:])[0,:,:]
                
                final_result.append({
                    'polygon': polygon_in_ori_img.tolist(),
                    'score': scores[i],
                    'label': labels[i],
                })

                if DEBUG_MODE:
                    dbg_result.append({
                        'polygon_in_rotated_img': polygon_in_rotated_img,
                        'polygon_in_ori_img': polygon_in_ori_img
                    })

        first_tile_img_info = frame_result[0]
        frame_idx = first_tile_img_info['frame_idx']
        angle_idx = first_tile_img_info['angle_idx']

        if DEBUG_MODE:
            self.dbg_show_final_result(dbg_result, first_tile_img_info['angle_idx'],
                first_tile_img_info['rotated_padded_frame_img'], tile_img_info['ori_frame_img'])

        # save json file
        json_file = os.path.join(self.out_json_path, "%05d_%02d.json" % (frame_idx, int(angle_idx * self.rotate_every_n_degree)))
        with open(json_file, 'w') as f:
            f.write(pprint.pformat(final_result, width=200, indent=1).replace("'", "\""))
    
        return final_result

if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        print("Select [%s]" % torch.cuda.get_device_name(torch.cuda.current_device()))

    video_detection = VideoDetection('/home/yao/work/eagle/data/1/1.mp4', None, '/home/yao/work/eagle/data/1/1_objs')
    video_detection.run()
