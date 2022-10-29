import os
import sys
from shutil import copyfile
from datetime import datetime
from collections import OrderedDict
import time
import math
import random
import cv2
import pandas as pd
import numpy as np
import torch
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
import warnings

import effdet

warnings.filterwarnings("ignore")
dataset_path_base = "_datasets"
model_path_base = "_dataset_issues"

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def obb_to_aabb(bbox, **kwargs):
    a = bbox[4] / 180 * 3.14159265
    x = (bbox[0] + bbox[2]) * 0.5
    y = (bbox[1] + bbox[3]) * 0.5
    length = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    c = math.cos(a)
    s = math.sin(a)
    
    xx1 = x + c * length * 0.5 - s * width * 0.5
    yy1 = y + s * length * 0.5 + c * width * 0.5
    
    xx2 = x + c * length * 0.5 + s * width * 0.5
    yy2 = y + s * length * 0.5 - c * width * 0.5
    
    xx3 = x - c * length * 0.5 - s * width * 0.5
    yy3 = y - s * length * 0.5 + c * width * 0.5
    
    xx4 = x - c * length * 0.5 + s * width * 0.5
    yy4 = y - s * length * 0.5 - c * width * 0.5
    bbox2 = (min(xx1, xx2, xx3, xx4), min(yy1, yy2, yy3, yy4), max(xx1, xx2, xx3, xx4), max(yy1, yy2, yy3, yy4))
    return bbox2

def get_train_transforms():
    return A.Compose(
        [
            #A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), use_obb=True, p=0.5),
            A.Lambda(bbox=obb_to_aabb, always_apply=True, use_obb=True, p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_train_transforms2(img_scale):
    return A.Compose(
        [
            #A.CLAHE(p=0.3),
            #A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.3, p=0.5),
            #A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3, val_shift_limit=0.3, p=0.9),
            #A.OneOf([
            #    A.HueSaturationValue(hue_shift_limit=0.5, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
            #    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            #],p=0.9),
            #A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            #A.JpegCompression(quality_lower=15, quality_upper=75, p=0.3),
            #A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
            #A.RandomGamma(gamma_limit=(80, 140), p=0.3),
            #A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=0,
            #    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, value=(0,0,0), p=0.5),
            #A.IAAPerspective(scale=(0.05, 0.05), keep_size=True, p=0.5),
            #A.OneOf([
            #    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.95),
            #    A.ToGray(p=0.05),
            #],p=0.9),
            #A.GaussianBlur(p=0.2),
            #A.ChannelShuffle(p=1.0),
            #A.ToGray(p=0.3),
            #A.HorizontalFlip(p=0.5),
            #A.VerticalFlip(p=0.5),
            #A.Resize(height=img_scale, width=img_scale, p=1.0),
            #A.Cutout(num_holes=4, max_h_size=32, max_w_size=32, fill_value=0, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

def get_valid_transforms():
    return A.Compose(
        [
            A.Lambda(bbox=obb_to_aabb, always_apply=True, use_obb=True, p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
def get_valid_transforms2(img_scale):
    return A.Compose(
        [
            #A.CLAHE(p=1.0),
            A.Resize(height=img_scale, width=img_scale, p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )

class_id_maps = {5:2, 7:2, 8:1}
class_name_maps = ["bg", "ped", "bike", "motor"]

class DatasetRetriever(Dataset):
    def __init__(self, root, box_scale, transform=None, transform2=None, test=False):
        super(DatasetRetriever, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.transform2 = transform2
        self.box_scale = box_scale
        data_path = os.path.join(root, "obj")
        if test:
            list_path = os.path.join(root, "test.txt")
        else:
            list_path = os.path.join(root, "train.txt")
        with open(list_path) as f1:
            lines = f1.readlines()
            lines = [line.replace("\n", "") for line in lines]
        self.img_ids = list(range(len(lines)))
        self.img_names = list(lines)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        path = self.img_names[index]
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #image /= 255.0

        with open(path.replace(".jpg", ".txt")) as f:
            lines = f.readlines()
        boxes = []
        cls_ids = []
        for i, line in enumerate(lines):
            values = [float(token) for token in line.replace("\n", "").split(" ")]
            #if int(values[0]) != 8 and i+1 != len(lines):
            #    continue
            bbox = [(values[1]-values[3]*0.5)*self.box_scale, (values[2]-values[4]*0.5)*self.box_scale,
                    (values[1]+values[3]*0.5)*self.box_scale, (values[2]+values[4]*0.5)*self.box_scale, values[5]+360.0 if values[5] < 0 else values[5]]
            boxes.append(bbox)
            cls_ids.append(1)
            #cls_ids.append(class_id_maps[int(values[0])])
            
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes = torch.min(boxes, torch.tensor([float(self.box_scale)]))
        boxes = torch.max(boxes, torch.tensor([0.0]))
        
        if 0:
            # there is only one class
            labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        else:
            labels = torch.tensor(cls_ids, dtype=torch.int64)
        
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transform:
            for i in range(10):
                sample = self.transform(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels,
                    'use_obb': True
                })
                sample = self.transform2(**{
                    'image': sample['image'],
                    'bboxes': sample['bboxes'],
                    'labels': sample['labels'],
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    #yxyx: be warning
                    #print(target['boxes'])
                    #print(target['boxes'].shape)
                    target['boxes'][:,[0,1,2,3]] = target['boxes'][:,[1,0,3,2]]
                    #print(target['boxes'].shape)
                    target['labels'] = torch.stack(sample['labels']) # <--- add this!
                    #print(target['boxes'].shape, target['labels'].shape)
                    #assert len(sample['bboxes']) == labels.shape[0], 'not equal!'
                    break

            image2 = image.permute(1,2,0).cpu().numpy()
            for i, box in enumerate(target['boxes'].cpu().numpy().astype(np.int32)):
                cv2.rectangle(image2, (box[1], box[0]), (box[3],  box[2]), (255, 0, 0), 3)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            cv2.imshow("image", image2)
            k = cv2.waitKey()
            if k == 27:
                exit(0)
            if k == 13:
                print(self.img_names[index])
                cv2.imwrite(self.img_names[index].replace("/", "_").replace(".._..__datasets", "_dataset_issues/"), image2)

        return image, target, img_id

    def __len__(self):
        return len(self.img_ids)
        
    def get_img(self, index):
        img_id = self.img_ids[index]
        path = os.path.join(self.root,  self.img_names[index])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    

class TrainGlobalConfig:
    num_workers = 0
    batch_size = 4
    n_epochs = 40
    samples_per_virtual_epoch = 10000
    #lr = 0.01
    #lr = 0.001
    lr = 0.0001
    #lr = 0.00001
    #lr = 0.00003
    # -------------------
    verbose = True
    verbose_step = 1
    eval_mAP_on_test_sets = True
    #test_sets = ["private_dataset_no_crop_aabb", "web-collection-001-002_dataset_no_crop_aabb"]
    test_sets = ["private170_dataset_no_crop_aabb"]
    # -------------------

    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False, 
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0, 
        min_lr=1e-8,
        eps=1e-08
    )
    

def collate_fn(batch):
    return tuple(zip(*batch))

def run_training(output_folder):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=SequentialSampler(train_dataset),
        #sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
        #shuffle=False,
    )

    config = TrainGlobalConfig
    for e in range(config.n_epochs):
        for step, (images, targets, image_ids) in enumerate(train_loader):
            pass

def build_dataset(names, filters, output_name):
    all_img = []
    for name, ratio in names.items():
        path = os.path.join(dataset_path_base, name)
        count = 0
        items = []
        for item in os.listdir(path):
            ext = item[item.rfind("."):]
            if ext not in filters:
                continue
            count += 1
            back_path = [".."]*(output_name.replace("\\", "/").count("/"))
            file_path = os.path.join(*back_path, dataset_path_base, name, item).replace("\\", "/")
            items.append(file_path)
        random.shuffle(items)
        items = items[:int(ratio*len(items))]
        all_img += items
        print("%7d/%7d\t%s" % (len(items), count, name))
    random.shuffle(all_img)

    total = len(all_img)
    train_n = int(total)
    #train_n = total - 8
    train_set = []
    test_set = []

    for i, item in enumerate(all_img):
        if i < train_n:
            train_set.append(item + "\n")
        else:
            test_set.append(item + "\n")

    random.shuffle(train_set)
    random.shuffle(test_set)

    output_path = model_path_base
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    if 0:
        train_path = os.path.join(output_path, "train.txt")
        with open(train_path, "w") as f:
            for line in train_set:
                f.write(line)

        test_path = os.path.join(output_path, "test.txt")
        with open(test_path, "w") as f:
            for line in test_set:
                f.write(line)

    print("Training set: %9d" % len(train_set))
    print("Test set:     %9d" % len(test_set))

    return output_path

if __name__ == '__main__':
    if len(sys.argv) > 2:
        SEED = int(sys.argv[2])
    else:
        SEED = 42
    seed_everything(SEED)

    img_scale = 768
    box_scale = 768
    num_classes = 1

    datasets = OrderedDict({
        "0009_dataset_20200901M2_20200907_1202_200m_fixed_768_768_obb":     1.0,
        "0010_dataset_20200901M2_20200907_1202_200m_fixed_1536_768_obb":    1.0,
        "0011_dataset_20200901M2_20200907_1202_200m_fixed_768_768_obb_bus": 1.0,
        "0012_dataset_20200901M2_20200907_1202_200m_fixed_1536_768_obb_bus":1.0,
        "0013_dataset_tongji_011_768_768_obb":                              1.0,
        "0014_dataset_20200901M2_20200903_1205_250m_fixed_768_768_obb":     1.0,
        "0015_dataset_20200901M2_20200907_1104_200m_fixed_768_768_obb":     1.0,
        "0016_dataset_ysq1_768_768_obb":                                    1.0,
        "0017_dataset_ysq1_1440_768_obb":                                   1.0,
        #"a004_dataset_changan001_ped_bike_motor_384_768_3classes":          1.0
        "0018_syq4_dataset_768_768_obb_bus":                                1.0,
        "0019_gm7_dataset_768_768_obb_bus":                                 1.0,
        #"0020_web-collection-003_888_768_768_obb":                          1.0,
        "0020_web-collection-003_1184_768_768_obb":                         1.0,
    })
    output_name = '_dataset_issues'
    if len(sys.argv) > 1:
        output_name = sys.argv[1]
    if not os.path.exists(output_name):
        os.mkdir(output_name)

    dataset_path = build_dataset(datasets, {".jpg", ".jpeg", ".png", ".bmp"}, output_name)
    train_dataset = DatasetRetriever(dataset_path, box_scale, transform=get_train_transforms(), transform2=get_train_transforms2(img_scale), test=False)

    if 0:
        for i, (img, target, img_id) in enumerate(train_dataset):
            img = img.permute(1,2,0).cpu().numpy()
            boxes = target['boxes'].cpu().numpy().astype(np.int32)
            class_ids = target['labels'].cpu().numpy().astype(np.int32)
            for i, box in enumerate(boxes):
                cv2.rectangle(img, (box[1], box[0]), (box[3],  box[2]), (255, 0, 0), 3)
                #img = cv2.putText(img, class_name_maps[class_ids[i]], (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv2.LINE_AA)
            print(img_id)
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            k = cv2.waitKey()
            if k == 27:
                exit(0)
            if k == 13:
                break
        cv2.destroyWindow("image")

    run_training(output_name)

