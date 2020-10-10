#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import os
import json
import time
import logging
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import numpy as np
import cv2

try:
    from apex import amp
    has_amp = True
except ImportError:
    has_amp = False

from effdet import create_model
from data import create_loader, MyDetection
from timm.utils import AverageMeter, setup_default_logging

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--anno', default='val2017',
                    help='mscoco annotation set (one of val2017, train2017, test-dev2017)')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default='mean', type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')


def infer(args):
    setup_default_logging()

    # might as well try to infer something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    # create model
    model = create_model(
        args.model,
        bench_task='predict',
        pretrained=args.pretrained,
        redundant_bias=args.redundant_bias,
        checkpoint_path=args.checkpoint,
        checkpoint_ema=args.use_ema,
    )
    input_size = model.config.image_size

    param_count = sum([m.numel() for m in model.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    model = model.cuda()
    if has_amp:
        print('Using AMP mixed precision.')
        model = amp.initialize(model, opt_level='O1')
    else:
        print('AMP not installed, running network in FP32.')

    #if args.num_gpu > 1:
    #    model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    annotation_path = os.path.join(args.data, 'annotations', f'instances_{args.anno}.json')
    image_dir = args.anno

    dataset = MyDetection(args.data, annotation_path)

    loader = create_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=args.interpolation,
        fill_color=args.fill_color,
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    img_ids = []
    results = []
    model.eval()

    #import pdb
    to_img = transforms.ToPILImage()
    
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output = model(input, target['img_scale'], target['img_size'])
            output = output.cpu()
            sample_ids = target['img_id'].cpu()
            for index, sample in enumerate(output):
                image_id = int(sample_ids[index])
                cv_img = dataset.get_img(image_id)
                
                #pdb.set_trace()
                for det in sample:
                    score = float(det[4])
                    if score < .2:  # stop when below this threshold, scores in descending order
                        break
                    my_det = dict(
                        image_id=image_id,
                        bbox=det[0:4].tolist(),
                        score=score,
                        category_id=int(det[5]))
                    img_ids.append(image_id)
                    results.append(my_det)
                    
                    bbox = det[0:4].tolist()
                    start_point = (int(bbox[0]), int(bbox[1])) 
                    end_point = (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])) 
                    color = (255, 0, 0) 
                    cv_img = cv2.rectangle(cv_img, start_point, end_point, color, 1)
                    
                cv2.imshow("input", cv_img)
                k = cv2.waitKey()
                if k == 27:
                    exit(0)

    json.dump(results, open(args.results, 'w'), indent=4)

    return results


def main():
    args = parser.parse_args()
    infer(args)


if __name__ == '__main__':
    main()

