#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module: sb_speed_test
Created: 2020-12-05

Description:

    copying the provided eval scripts, but to run on our downloaded content

Usage:

    $> python sb_speed_test.py

"""
import argparse
import datetime
import glob
import os
import pickle
import sys

import torch
import torch.hub
import yaml
from torch.backends import cudnn

HERE = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(HERE)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (invert_affine, postprocess, preprocess)

DATA = os.path.join(ROOT, 'data')
MODEL_DIR = os.path.join(ROOT, 'model')

with open(os.path.join(ROOT, 'projects', 'coco.yml')) as fp:
    PARAMS = yaml.safe_load(fp)


def get_weights_url(c):
    """this list of urls is hand-created from
    https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/readme.md as of the
    time of writing this file (2020-12-03). as the model is updated, these urls *could* (but
    likely won't) change much. of course we may need to update this over time

    this could be automated with the github releases api, but that's overkill rn
    https://developer.github.com/v3/repos/releases/

    Args
        c: the compound coefficient (read the efficientnet/det papers for details)

    Returns
        url of the weights file to download

    """
    url_map = {
        0: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d0.pth',
        1: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d1.pth',
        2: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d2.pth',
        3: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d3.pth',
        4: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d4.pth',
        5: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d5.pth',
        6: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0'
           '/efficientdet-d6.pth',
        7: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
           '/efficientdet-d7.pth',
        8: 'https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2'
           '/efficientdet-d8.pth', }

    try:
        return url_map[c]
    except KeyError:
        raise ValueError(
            "compound coefficient not found, cannot download model weights")


def model_fn(model_dir, compound_coef=0, use_cuda=False, use_float16=False):
    # based entirely off of
    # https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/coco_eval.py
    print(f'building and loading efficientdet d{compound_coef}')
    model = EfficientDetBackbone(compound_coef=compound_coef,
                                 num_classes=len(PARAMS['obj_list']),
                                 ratios=eval(PARAMS['anchors_ratios']),
                                 scales=eval(PARAMS['anchors_scales']))
    state_dict = torch.hub.load_state_dict_from_url(
        url=get_weights_url(c=compound_coef),
        model_dir=model_dir,
        map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model.cuda(0)

    if use_float16:
        model.half()

    return model


def image_path_batches(image_paths, batch_size=2):
    i0 = 0
    L = len(image_paths)
    while i0 < L:
        yield image_paths[i0: i0 + batch_size]
        i0 += batch_size


def main(compound_coef=0, model_dir=MODEL_DIR, nms_threshold=0.5, use_cuda=False, use_float16=False,
         image_batch_size=2):
    threshold = 0.05

    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef]

    model = model_fn(model_dir=model_dir, compound_coef=compound_coef, use_cuda=use_cuda,
                     use_float16=use_float16)

    image_paths = glob.glob(os.path.join(DATA, '*.jpg'))

    L = len(image_paths)
    print(f'processing {L} in batches of {image_batch_size}')
    results = {}
    loop_start = datetime.datetime.now()
    for image_batch in image_path_batches(image_paths, image_batch_size):
        batch_start = datetime.datetime.now()
        ori_images, framed_images, framed_metas = preprocess(*image_batch, max_size=input_size)

        # build tensor from framed images
        x = torch.stack([(torch.from_numpy(fi).cuda()
                          if use_cuda
                          else torch.from_numpy(fi))
                         for fi in framed_images],
                        0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        with torch.no_grad():
            features, regression, classification, anchors = model(x)

            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, nms_threshold)

            out = invert_affine(framed_metas, out)

        batch_end = datetime.datetime.now()
        batch_time = (batch_end - batch_start).total_seconds()
        print(f"batch_time = {batch_time} (s)")
        print(f"batch_size = {image_batch_size}")
        print(f"FPS = {image_batch_size / batch_time:0.4f}")
        print(f"SPF = {batch_time / image_batch_size:0.4f}")

        results.update(dict(zip(image_batch, out)))

    loop_end = datetime.datetime.now()
    loop_time = (loop_end - loop_start).total_seconds()
    print('\nfinal summary:')
    print(f"total processing time: {loop_time} (s)")
    print(f"number of frames processed: {len(image_paths)}")
    print(f"batch_size = {image_batch_size}")
    print(f"FPS: {L / loop_time:0.4f}")
    print(f"SPF: {loop_time / L:0.4f}")

    with open(f'results.{compound_coef}.pkl', 'wb') as fp:
        pickle.dump(results, fp)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--compound_coef', type=int, default=0,
                    help='coefficients of efficientdet')
    ap.add_argument('--nms_threshold', type=float, default=0.5,
                    help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--gpu', action='store_true')
    ap.add_argument('--float16', action='store_true')
    ap.add_argument('--image-batch-size', type=int, default=2)
    return ap.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("=" * 80)
    print(f'compound_coef = {args.compound_coef}')
    print(f'nms_threshold = {args.nms_threshold}')
    print(f'gpu = {args.gpu}')
    print(f'float16 = {args.float16}\n')
    print(f'image_batch_size = {args.image_batch_size}')
    main(compound_coef=args.compound_coef,
         nms_threshold=args.nms_threshold,
         use_cuda=args.gpu,
         use_float16=args.float16,
         image_batch_size=args.image_batch_size)
