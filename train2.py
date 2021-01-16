import os
import sys
from shutil import copyfile
from datetime import datetime
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
model_path_base = "_models"

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

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
            #A.ColorJitter(brightness=0.2, contrast=0.5, saturation=0.2, hue=0.3, p=0.5),
            #A.HueSaturationValue(hue_shift_limit=0.3, sat_shift_limit=0.3, val_shift_limit=0.3, p=0.9),
            #A.OneOf([
            #    A.HueSaturationValue(hue_shift_limit=0.5, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
            #    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            #],p=0.9),
            #A.OneOf([
            #    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05, p=0.95),
            #    A.ToGray(p=0.05),
            #],p=0.9),
            #A.GaussianBlur(p=0.2),
            #A.ChannelShuffle(p=1.0),
            #A.ToGray(p=0.3),
            A.HorizontalFlip(p=0.5),
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
        path = os.path.join(self.root,  self.img_names[index])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        with open(path.replace(".jpg", ".txt")) as f:
            lines = f.readlines()
        boxes = []
        cls_ids = []
        for i, line in enumerate(lines):
            values = [float(token) for token in line.replace("\n", "").split(" ")]
            if int(values[0]) != 8 and i+1 != len(lines):
                continue
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

        return image, target, img_id

    def __len__(self):
        return len(self.img_ids)
        
    def get_img(self, index):
        img_id = self.img_ids[index]
        path = os.path.join(self.root,  self.img_names[index])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

class Logger:
    def __init__(self, config, output_folder):
        self.config = config
        self.log_path = f'./{output_folder}/log.txt'
        with open(self.log_path, 'a+') as f:
            f.write("="*80 + "\n")
        
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as f:
            f.write(f'{message}\n')

class Fitter:
    def __init__(self, model, device, config, output_folder, logger):
        self.config = config
        self.logger = logger
        self.epoch = 0

        self.base_dir = f'./{output_folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10**5

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.logger.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        if 0:
            from torch_lr_finder import LRFinder
            criterion = torch.nn.CrossEntropyLoss()
            lr_finder = LRFinder(self.model, self.optimizer, criterion, self.device)
            lr_finder.range_test(train_loader, start_lr=0.000001, end_lr=0.01, num_iter=1000)
            lr_finder.plot()
            plt.savefig("LRvsLoss.png")
            plt.close()

        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.utcnow().isoformat()
                self.logger.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.logger.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {((time.time() - t)/60.0):.1f} mins                  ')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss = self.validation(validation_loader)

            self.logger.log(f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {((time.time() - t)/60.0):.1f} mins                   ')
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
            self.model.eval()
            self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')
            #for path in sorted(glob(f'{self.base_dir}/best-checkpoint-*epoch.bin'))[:-3]:
            #    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {((time.time() - t)/60.0):.1f} mins ' + \
                        f'remaining: {(time.time() - t)/(step+1)*(len(val_loader)-step-1)/60:.1f} mins           ', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                target_res = {}
                target_res['bbox'] = boxes
                target_res['cls'] = labels 
                target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
                target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)

                outputs = self.model(images, target_res)
                loss = outputs['loss']
                
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {((time.time() - t)/60.0):.1f} mins ' + \
                        f'remaining: {((time.time() - t)/(step+1)*(len(train_loader)-step-1)/60.0):.1f} mins            ', end='\r'
                    )
            
            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            target_res = {}
            target_res['bbox'] = boxes
            target_res['cls'] = labels 
            target_res["img_scale"] = torch.tensor([1.0] * batch_size, dtype=torch.float).to(self.device)
            target_res["img_size"] = torch.tensor([images[0].shape[-2:]] * batch_size, dtype=torch.float).to(self.device)

            self.optimizer.zero_grad()
            
            outputs = self.model(images, target_res)
            loss = outputs['loss']
            
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            self.optimizer.step()

            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss
    
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

class TrainGlobalConfig:
    num_workers = 4
    batch_size = 4
    n_epochs = 256
    lr = 0.01
    #lr = 0.001
    #lr = 0.0001
    #lr = 0.00003

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    '''
    step_scheduler = True  # do scheduler.step after optimizer.step
    validation_scheduler = False  # do scheduler.step after validation stage loss
    SchedulerClass = torch.optim.lr_scheduler.OneCycleLR
    scheduler_params = dict(
        max_lr=0.0007,
        epochs=n_epochs,
        steps_per_epoch=int(18899 / batch_size),
        pct_start=0.1,
        anneal_strategy='cos', 
        final_div_factor=10**5
    )

    '''
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

def run_training(net, output_folder, logger):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        validation_dataset, 
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig, output_folder=output_folder, logger=logger)
    fitter.fit(train_loader, val_loader)

def build_net(type, img_scale, num_classes):
    config = effdet.get_efficientdet_config(type)
    net = effdet.EfficientDet(config, pretrained_backbone=False)
    checkpoint = torch.load(model_path_base + '/' + type + '.pth')
    net.load_state_dict(checkpoint)
    config.num_classes = num_classes
    config.image_size = img_scale
    net.class_net = effdet.efficientdet.HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    return effdet.DetBenchTrain(net, config)

def build_dataset(names, filters, output_name, logger):
    all_img = []
    logger.log("Build dataset:")
    for name in names:
        path = os.path.join(dataset_path_base, name)
        logger.log("\t" + name)
        for item in os.listdir(path):
            ext = item[item.rfind("."):]
            if ext not in filters:
                continue
            back_path = [".."]*(output_name.replace("\\", "/").count("/")+2)
            file_path = os.path.join(*back_path, dataset_path_base, name, item).replace("\\", "/")
            all_img.append(file_path)
    random.shuffle(all_img)

    total = len(all_img)
    train_n = int(total * 0.8)
    train_set = []
    test_set = []

    for i, item in enumerate(all_img):
        if i < train_n:
            train_set.append(item + "\n")
        else:
            test_set.append(item + "\n")

    random.shuffle(train_set)
    random.shuffle(test_set)

    output_path = os.path.join(model_path_base, output_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    train_path = os.path.join(output_path, "train.txt")
    with open(train_path, "w") as f:
        for line in train_set:
            f.write(line)

    test_path = os.path.join(output_path, "test.txt")
    with open(test_path, "w") as f:
        for line in test_set:
            f.write(line)

    logger.log("Training set: %7d" % len(train_set))
    logger.log("Test set:     %7d" % len(test_set))

    return output_path

if __name__ == '__main__':
    SEED = 42
    seed_everything(SEED)

    img_scale = 768
    box_scale = 768
    num_classes = 3


    datasets = [
        #"0009_dataset_20200901M2_20200907_1202_200m_fixed_768_768_obb",
        #"0010_dataset_20200901M2_20200907_1202_200m_fixed_1536_768_obb",
        #"0011_dataset_20200901M2_20200907_1202_200m_fixed_768_768_obb_bus",
        #"0012_dataset_20200901M2_20200907_1202_200m_fixed_1536_768_obb_bus",
        #"0013_dataset_tongji_011_768_768_obb",
        #"0014_dataset_20200901M2_20200903_1205_250m_fixed_768_768_obb",
        #"0015_dataset_20200901M2_20200907_1104_200m_fixed_768_768_obb",
        #"0016_dataset_ysq1_768_768_obb",
        #"0017_dataset_ysq1_1440_768_obb",
        "a004_dataset_changan001_ped_bike_motor_384_768_3classes"
        "0018_syq4_dataset_768_768_obb_bus",
        "0019_gm7_dataset_768_768_obb_bus",
        ]
    output_name = 'effdet-d2-drone_'
    model_type = 'tf_efficientdet_d2'

    output_path = os.path.join(model_path_base, output_name)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    logger = Logger(TrainGlobalConfig, output_path)
    logger.log("seed:\t\t%d" % SEED)
    logger.log("img_scale:\t%d" % img_scale)
    logger.log("box_scale:\t%d" % box_scale)
    dataset_path = build_dataset(datasets, {".jpg", ".jpeg", ".png", ".bmp"}, output_name, logger)
    train_dataset = DatasetRetriever(dataset_path, box_scale, transform=get_train_transforms(), transform2=get_train_transforms2(img_scale), test=False)
    validation_dataset = DatasetRetriever(dataset_path, box_scale, transform=get_valid_transforms(), transform2=get_train_transforms2(img_scale), test=True)

    if 1:
        for i, (img, target, img_id) in enumerate(train_dataset):
            img = img.permute(1,2,0).cpu().numpy()
            boxes = target['boxes'].cpu().numpy().astype(np.int32)
            class_ids = target['labels'].cpu().numpy().astype(np.int32)
            for i, box in enumerate(boxes):
                cv2.rectangle(img, (box[1], box[0]), (box[3],  box[2]), (0, 1, 0), 1)
                img = cv2.putText(img, class_name_maps[class_ids[i]], (box[1], box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) , 2, cv2.LINE_AA) 
            cv2.imshow("image", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            k = cv2.waitKey()
            if k == 27:
                exit(0)
            if k == 13:
                break
        cv2.destroyWindow("image")

    net = build_net(model_type, img_scale, num_classes)
    device = torch.device('cuda:0')
    net.to(device)
    copyfile(sys.argv[0], os.path.join(output_path, os.path.split(sys.argv[0])[-1]))
    run_training(net, output_path, logger)

