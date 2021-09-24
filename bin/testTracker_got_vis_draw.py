# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np
import math
import sys

sys.path.append('../')
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from pysot.core.config import cfg
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder_gfl import ModelBuilder
from toolkit.datasets import DatasetFactory
from pysot.tracker.siamgfl_tracker import SiamGATTracker


parser = argparse.ArgumentParser(description='siamgat tracking')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--dataset', type=str, default='GOT-10k',
        help='datasets') # OTB100 LaSOT UAV123 GOT-10k
parser.add_argument('--vis', action='store_true', default=True,
        help='whether visualzie result')
parser.add_argument('--snapshot', type=str, default='snapshot/got10k_model.pth',
        help='snapshot of models to eval')
parser.add_argument('--config', type=str, default='../experiments/siamgat_googlenet_got10k/config.yaml',
        help='config file')
args = parser.parse_args()

torch.set_num_threads(1)


def main():
    # load config
    cfg.merge_from_file(args.config)

    # Test dataset
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '/media/david/000AA762000315F3/Datesets/GOT_10k/val')
    #//home/david/Desktop/pysot-master/testing_dataset

    # set hyper parameters
    params = getattr(cfg.HP_SEARCH, 'GOT10k')
    cfg.TRACK.LR = params[0]
    cfg.TRACK.PENALTY_K = params[1]
    cfg.TRACK.WINDOW_INFLUENCE = params[2]

    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval()

    # build tracker
    tracker = SiamGATTracker(model)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    model_name = args.snapshot.split('/')[-1].split('.')[-2]

    box_dir = "results/others"

    # OPE tracking
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            # test one special video
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        track_times = []

        name = video.name
        record_file = os.path.join(box_dir,'Ours_VAL',name,'%s_001.txt'%video.name)
        pact_box = np.loadtxt(record_file, delimiter=',')

        record_file1 = os.path.join(box_dir,'SiamGAT_val',name,'%s_001.txt'%video.name)
        pact_box1 = np.loadtxt(record_file1, delimiter=',')

        record_file2 = os.path.join(box_dir,'siamfcpp_val',name,'%s_001.txt'%video.name)
        pact_box2 = np.loadtxt(record_file2, delimiter=',')

        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()

            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                #print(outputs['ltrb'])
                pred_bboxes.append(pred_bbox)
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                if not any(map(math.isnan,gt_bbox)):
                    x1, y1, w1, h1 = pact_box[idx]
                    x11, y11, w11, h11 = pact_box1[idx]
                    x12, y12, w12, h12 = pact_box2[idx]
                    gt_bbox = list(map(int, gt_bbox))
                    pred_bbox = list(map(int, pred_bbox))
                    '''out = cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                  (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)'''   # green
                    out = cv2.rectangle(img, (int(x1), int(y1)),
                                  (int(x1 + w1), int(y1 + h1)), (255, 0, 0), 3)  # red
                    out = cv2.rectangle(out, (int(x11), int(y11)),
                                  (int(x11 + w11), int(y11 + h11)), (0, 255, 0), 3)  # green
                    out = cv2.rectangle(out, (int(x12), int(y12)),
                                  (int(x12 + w12), int(y12 + h12)), (0, 0, 255), 3)  # blue
                    #out = cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                  #(pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)    # yellow
                    out = cv2.putText(out, str(idx), (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.imshow(video.name, out)
                    cv2.waitKey(1)
                    out_path = os.path.join('output_box', video.name)
                    if not os.path.isdir(out_path):
                        os.makedirs(out_path)
                    cv2.imwrite("output_box/{seq_name}/{seq_name}_{id}.jpg".format(seq_name=video.name, id=idx), out)
        toc /= cv2.getTickFrequency()

        # save results
        if 'GOT-10k' == args.dataset:
            video_path = os.path.join('results', 'GOT-10K-Val', model_name, video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x]) + '\n')
            result_path = os.path.join(video_path,
                                       '{}_time.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in track_times:
                    f.write("{:.6f}\n".format(x))
        else:
            model_path = os.path.join('results', args.dataset, model_name)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
